"""VLNDAPORayTrainer — VLN full-episode DAPO with real Dynamic Sampling (group filtering).

This subclasses the stock ``RayPPOTrainer`` and overrides ``fit()`` to add DAPO's
**Dynamic Sampling** on top of the VLN online-GRPO flow. Clip-Higher / dual-clip /
token-level loss are pure actor-loss config (set in the run scripts); the missing
DAPO piece — group filtering + resampling — lives *here* because the stock trainer's
``fit()`` has no such loop.

Under binary success reward, an episode
whose N rollouts all succeed (or all fail) yields advantage=0, pg_loss=0 for that whole
group — a zero-gradient sample that wasted an expensive Habitat rollout. Dynamic Sampling
drops those groups and keeps generating **new** episodes until we have ``train_batch_size``
groups with signal (without refilling from filtered zero-gradient groups).

Filtering unit = the flattened batch's ``uid`` (== ``group_uid``; the N trajectories of
one original episode share it). A uid is kept iff the std of its rows' ``trajectory_reward``
is > eps (i.e. its N trajectories are not all-equal). See
``online_rollout_manager.py`` for how the flattened batch carries ``uid`` /
``trajectory_uid`` / ``trajectory_reward`` / ``is_padding``.

Isolation: nothing here is imported unless ``recipe.vln_simplevln.dapo_patch`` is on
``VERL_USE_EXTERNAL_MODULES`` (only the DAPO run script sets it), which swaps
``verl.trainer.main_ppo.RayPPOTrainer`` to this class. The GRPO / plain-DAPO paths are
byte-for-byte untouched, and Dynamic Sampling itself is gated on
``algorithm.filter_groups.enable`` — off ⇒ this fit() reduces to the stock VLN step.

Assumptions (asserted at fit() start, fail-loud): adv_estimator=grpo, no critic, no
REMAX, no rollout_correction bypass, no reference policy required. These match the VLN
recipe; anything else should use the stock trainer.
"""

import hashlib
import io
import json
from pathlib import Path
import uuid

import numpy as np
import torch

# Resolve trainer-level helpers through the ray_trainer *module* (not by value-import) so
# the monkey-patches installed by recipe.vln_simplevln.patches — which replace
# ray_trainer.compute_advantage (trajectory-level GRPO) and ray_trainer.compute_data_metrics
# (padding-free / traj-level) as module attributes — are the ones actually called, exactly
# as the stock fit() resolves them.
import verl.trainer.ppo.ray_trainer as _rt
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from recipe.vln_simplevln.batch_utils import padding_multiple

# dapo_patch copies recipe methods onto the stock class. Keep the original
# checkpoint methods explicitly; super() would recurse after that installation.
_base_save_checkpoint = RayPPOTrainer._save_checkpoint
_base_load_checkpoint = RayPPOTrainer._load_checkpoint


class VLNDAPORayTrainer(RayPPOTrainer):
    """RayPPOTrainer + DAPO Dynamic Sampling for the VLN full-episode flatten flow."""

    # ── small helpers ────────────────────────────────────────────────────────
    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        folder.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_vln_last_saved_step", None) == self.global_steps:
            # Re-registering the same actor path can make upstream retention delete
            # it. Only the sampler advanced through the filtered tail of this epoch.
            temporary_data = folder / "data.pt.tmp"
            torch.save(self.train_dataloader.state_dict(), temporary_data)
            temporary_data.replace(folder / "data.pt")
        else:
            _base_save_checkpoint(self)
        progress = {
            "version": 1, "global_steps": self.global_steps,
            "epoch": self._vln_epoch, "batches_in_epoch": self._vln_batches_in_epoch,
            "batches_per_epoch": len(self.train_dataloader),
            "data_sha256": hashlib.sha256((folder / "data.pt").read_bytes()).hexdigest(),
        }
        temporary = folder / "vln_progress.json.tmp"
        temporary.write_text(json.dumps(progress) + "\n")
        temporary.replace(folder / "vln_progress.json")
        self._vln_last_saved_step = self.global_steps

    def _load_checkpoint(self):
        self._vln_epoch = 0
        self._vln_batches_in_epoch = 0
        self._vln_last_saved_step = None
        cfg = self.config.trainer
        if cfg.resume_mode == "disable":
            return
        if cfg.get("default_hdfs_dir") is not None:
            raise NotImplementedError("Checkpoint resume requires a local directory")
        if cfg.resume_mode == "auto":
            latest = _rt.find_latest_ckpt_path(cfg.default_local_dir)
            if latest is None:
                print("Training from scratch")
                return
            folder = Path(latest)
        elif cfg.resume_mode == "resume_path":
            folder = Path(cfg.resume_from_path)
        else:
            raise ValueError(f"Unknown resume mode: {cfg.resume_mode}")
        if not folder.name.startswith("global_step_"):
            raise ValueError("Resume path must name a global_step_<number> checkpoint")
        checkpoint_step = int(folder.name.removeprefix("global_step_"))
        val_only = cfg.get("val_only", False)
        if not val_only:
            progress_file = folder / "vln_progress.json"
            if not progress_file.is_file():
                raise ValueError(
                    "Checkpoint lacks vln_progress.json; optimizer steps cannot identify DAPO's data epoch. "
                    "Export its actor as MODEL_PATH and set trainer.resume_mode=disable for a new run."
                )
            progress = json.loads(progress_file.read_text())
            if (progress.get("version") != 1 or progress.get("global_steps") != checkpoint_step
                    or progress.get("batches_per_epoch") != len(self.train_dataloader)):
                raise ValueError("Checkpoint sampling progress is incompatible with this training dataloader")
            epoch, batches = progress.get("epoch"), progress.get("batches_in_epoch")
            if (type(epoch) is not int or epoch < 0 or type(batches) is not int
                    or not 0 <= batches <= len(self.train_dataloader)):
                raise ValueError("Checkpoint has invalid DAPO sampling counters")
            data_bytes = (folder / "data.pt").read_bytes()
            expected_hash = progress.get("data_sha256")
            if expected_hash and hashlib.sha256(data_bytes).hexdigest() != expected_hash:
                raise ValueError("Checkpoint data.pt and sampling progress do not match; the save may be incomplete")
            state = torch.load(io.BytesIO(data_bytes), weights_only=False)

        # Validate before loading weights. Pin auto-resume to the checked directory,
        # and normalize explicit paths that have a trailing slash.
        mode, original_path = cfg.resume_mode, cfg.get("resume_from_path")
        try:
            cfg.resume_mode, cfg.resume_from_path = "resume_path", str(folder)
            _base_load_checkpoint(self)
        finally:
            cfg.resume_mode, cfg.resume_from_path = mode, original_path
        if val_only:
            return
        # Stock restore may skip data.pt based on optimizer steps. DAPO's data
        # cursor is independent, so always restore the validated state here.
        self.train_dataloader.load_state_dict(state)
        self._vln_epoch, self._vln_batches_in_epoch = epoch, batches
        output_folder = Path(cfg.default_local_dir) / f"global_step_{self.global_steps}"
        if folder.resolve() == output_folder.resolve():
            self._vln_last_saved_step = self.global_steps

    def _pad_multiple(self) -> int:
        return padding_multiple(self.config)

    def _vln_repad(self, batch: DataProto, mult: int) -> DataProto:
        """Pad ``batch`` up to a multiple of ``mult`` with padding rows that EXACTLY match
        the manager's convention:

          * a single pad token at the last prompt position, 1 attended token, empty response;
          * NO image placeholder tokens and null multimodal fields (image_buffer_ref=None,
            raw_prompt="", mm_processor_kwargs={}, image_indices=[]) → these rows carry no
            multimodal content and never touch the image merge in the actor forward;
          * decision_loss_weight=0, rm_scores=0, response_mask=0 → zero loss contribution;
          * unique ``__vln_padding__{i}`` uid & trajectory_uid → zero reward and weight ⇒ adv 0.

        We reuse ``select_idxs`` only to inherit the exact tensor schema (keys/shape/dtype/
        device), then overwrite every field. We must NOT use ``pad_dataproto_to_divisor``:
        it duplicates *real* rows, re-using real uids and biasing GRPO group-normalization.
        """
        n = len(batch)
        if n == 0:
            return batch
        rem = n % mult
        if rem == 0:
            return batch
        pad = mult - rem

        idxs = (np.arange(pad) % n).astype(np.int64)
        tmpl = batch.select_idxs(idxs)

        # pad token id (matches manager's meta_info.get("pad_token_id", 0) or 0 fallback)
        pad_id = 0
        tok = getattr(self, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            pad_id = int(tok.pad_token_id)

        # Zero all tensors, then place a single attended pad token at the last prompt position
        # (left-padded convention). input_ids == cat(prompts, responses), so its prompt region
        # is [:, :prompt_len]. Everything else (responses, position_ids, response_mask, rm_scores)
        # stays zero — identical to a manager padding row.
        bt = tmpl.batch
        for k in list(bt.keys()):
            bt[k] = torch.zeros_like(bt[k])
        if "prompts" in bt.keys():
            prompt_len = bt["prompts"].shape[1]
            bt["prompts"][:, -1] = pad_id
            if "input_ids" in bt.keys():
                bt["input_ids"][:, prompt_len - 1] = pad_id
            if "attention_mask" in bt.keys():
                bt["attention_mask"][:, prompt_len - 1] = 1

        def _fill(val):
            a = np.empty(pad, dtype=object)
            for i in range(pad):
                a[i] = val
            return a

        pad_uids = np.array([f"__vln_padding__{i}" for i in range(pad)], dtype=object)
        overrides = {
            "uid": pad_uids,
            "trajectory_uid": pad_uids.copy(),
            "trajectory_reward": np.zeros(pad, dtype=np.float32),
            "trajectory_success": np.zeros(pad, dtype=np.float32),
            "decision_loss_weight": np.zeros(pad, dtype=np.float32),
            "turn_id": np.full(pad, -1, dtype=np.int32),
            "action_text": _fill(""),
            "image_buffer_ref": _fill(None),
            "image_indices": _fill([]),
            "raw_prompt": _fill(""),
            "mm_processor_kwargs": _fill({}),
            "is_padding": np.ones(pad, dtype=bool),
        }
        ntb = tmpl.non_tensor_batch
        for k, v in overrides.items():
            if k in ntb:
                ntb[k] = v

        return DataProto.concat([batch, tmpl])

    @staticmethod
    def _valid_uids(flat: DataProto, eps: float):
        """Return (kept_row_mask, ordered_kept_uids, n_filtered) for a flattened batch.

        A uid is kept iff std(its rows' trajectory_reward) > eps, i.e. its trajectories are
        not all-equal (zero-gradient group otherwise). Padding rows are assumed already
        removed by the caller. Order of kept uids follows first appearance.
        """
        uids = np.asarray(flat.non_tensor_batch["uid"], dtype=object)
        rewards = np.asarray(flat.non_tensor_batch["trajectory_reward"], dtype=np.float64)

        kept_uids, all_uids_order, seen = [], [], set()
        for u in uids:
            if u not in seen:
                seen.add(u)
                all_uids_order.append(u)
        for u in all_uids_order:
            r = rewards[uids == u]
            if r.std() > eps:
                kept_uids.append(u)
        kept_set = set(kept_uids)
        kept_mask = np.array([u in kept_set for u in uids], dtype=bool)
        n_filtered = len(all_uids_order) - len(kept_uids)
        return kept_mask, kept_uids, n_filtered

    def _vln_metrics_from_batch(self, step_batch: DataProto) -> dict:
        """Recompute trajectory-level SR/reward over the *trained* batch (non-padding,
        one sample per trajectory_uid). Replaces the manager's per-gen-batch vln_metrics,
        which is only partial after accumulating multiple gen batches."""
        ntb = step_batch.non_tensor_batch
        if "trajectory_uid" not in ntb or "trajectory_reward" not in ntb:
            return {}
        is_pad = np.asarray(ntb.get("is_padding", np.zeros(len(step_batch), dtype=bool)), dtype=bool)
        tuid = np.asarray(ntb["trajectory_uid"], dtype=object)
        rew = np.asarray(ntb["trajectory_reward"], dtype=np.float64)
        success = np.asarray(ntb["trajectory_success"], dtype=np.float64)
        seen, r_per_traj, s_per_traj = set(), [], []
        for i in range(len(tuid)):
            if is_pad[i] or tuid[i] in seen:
                continue
            seen.add(tuid[i])
            r_per_traj.append(rew[i])
            s_per_traj.append(success[i])
        if not r_per_traj:
            return {}
        r = np.array(r_per_traj)
        return {
            "vln/traj_reward/mean": float(r.mean()),
            "vln/traj_reward/std": float(r.std()),
            "vln/traj_reward/max": float(r.max()),
            "vln/traj_reward/min": float(r.min()),
            "vln/traj_sr": float(np.mean(s_per_traj)),
            "vln/traj_count": int(len(r)),
        }

    # ── main loop ─────────────────────────────────────────────────────────────
    def fit(self):
        try:
            return self._fit()
        finally:
            self._shutdown_dump_executor()

    def _fit(self):
        """PPO/GRPO loop + DAPO Dynamic Sampling (VLN full-episode flatten flow)."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # Fail loud on features this lean fit() intentionally does not replicate.
        assert self.config.algorithm.adv_estimator in (_rt.AdvantageEstimator.GRPO, "grpo"), (
            "VLNDAPORayTrainer only supports adv_estimator=grpo"
        )
        assert not self.use_critic, "VLNDAPORayTrainer does not support a critic"
        rc = self.config.algorithm.get("rollout_correction", None)
        assert not (rc and rc.get("bypass_mode", False)), "rollout_correction bypass unsupported"
        assert self.config.algorithm.adv_estimator not in (
            _rt.AdvantageEstimator.REMAX,
        ), "REMAX unsupported here"

        if self._dump_executor._shutdown:
            self._init_dump_executor()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        val_only = self.config.trainer.get("val_only", False)
        if not val_only and (self.global_steps >= self.total_training_steps
                            or self._vln_epoch >= self.config.trainer.total_epochs):
            return
        self.checkpoint_manager.update_weights(self.global_steps)
        current_epoch = self._vln_epoch

        if val_only or self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            _rt.pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if val_only:
                return

        progress_bar = _rt.tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0   # base initializes this inside fit(); replicate

        rollout_n = self.config.actor_rollout_ref.rollout.n

        # ── DAPO Dynamic Sampling config ──────────────────────────────────────
        fg = self.config.algorithm.get("filter_groups", None)
        ds_enable = bool(fg and fg.get("enable", False))
        target_uids = self.config.data.train_batch_size          # valid-uid target = #prompts
        max_gen = int(fg.get("max_num_gen_batches", 0)) if ds_enable else 0
        eps = 1e-8
        print(
            f"[VLN-DAPO] dynamic_sampling={ds_enable} target_valid_uids={target_uids} "
            f"gen_batch_size={self.config.data.get('gen_batch_size', target_uids)} "
            f"max_num_gen_batches={max_gen} (<=0 means unlimited)"
        )

        # Accumulators persist ACROSS dataloader iterations (and epoch boundaries) until a
        # real training step fires — this is what lets "not enough ⇒ fetch new episodes".
        accum: list[DataProto] = []
        accum_uid_order: list = []
        accum_uid_set: set = set()
        num_gen_batches = 0
        gen_filter_total = 0
        timing_raw = {}

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            self._vln_epoch = epoch
            for batch_dict in self.train_dataloader:
                self._vln_batches_in_epoch += 1
                metrics = {}

                # ── 1) generate + flatten (VLN path, mirrors stock fit 1385-1453) ──
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                data_exhausted = (epoch + 1 == self.config.trainer.total_epochs
                                  and self._vln_batches_in_epoch == len(self.train_dataloader))
                is_last_step = self.global_steps >= self.total_training_steps or data_exhausted

                with _rt.marked_timer("step", timing_raw):
                    with _rt.marked_timer("gen", timing_raw, color="red"):
                        gen_out = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        # NOTE: do NOT sleep_replicas() here. sleep frees the rollout replicas'
                        # weight+KV-cache device memory; with Dynamic Sampling a step may generate
                        # multiple times (resampling), and each generate needs replicas awake.
                        # Nothing wakes them between two gens in a step (stock relies on the
                        # end-of-step update_weights to re-sync before the next step's gen). So we
                        # sleep exactly once, right before the training forward/backward below.
                        timing_raw.update(gen_out.meta_info.get("timing", {}))
                        gen_out.meta_info.pop("timing", None)

                    # Training consumes decision rows directly; episode-level inputs
                    # have a different row count and cannot be joined with this batch.
                    if not gen_out.meta_info.get("vln_flattened"):
                        raise ValueError("VLN training requires flattened decision rollouts")
                    flat = gen_out
                    flat.meta_info = {**batch.meta_info, **gen_batch_output.meta_info, **flat.meta_info}
                    if "response_mask" not in flat.batch.keys():
                        flat.batch["response_mask"] = _rt.compute_response_mask(flat)

                    # ── 2) DAPO filter + accumulate (or straight-through if disabled) ──
                    if not ds_enable:
                        step_batch = flat
                    else:
                        num_gen_batches += 1
                        # drop the manager's padding rows before grouping
                        is_pad = np.asarray(flat.non_tensor_batch["is_padding"], dtype=bool)
                        real = flat.select_idxs(~is_pad) if is_pad.any() else flat
                        kept_mask, kept_uids, n_filtered = self._valid_uids(real, eps)
                        gen_filter_total += n_filtered
                        if kept_mask.any():
                            frag = real.select_idxs(kept_mask)
                            # DataProto.concat asserts meta_info consistency across items; frags
                            # from different gen batches carry differing vln_metrics /
                            # seqlen_sorted_indices, which would trip that assert. We rebuild
                            # step_batch.meta_info from the current `flat` after concat, so the
                            # frags' meta_info is unneeded — clear it to keep concat conflict-free.
                            frag.meta_info = {}
                            accum.append(frag)
                            for u in kept_uids:
                                if u not in accum_uid_set:
                                    accum_uid_set.add(u)
                                    accum_uid_order.append(u)
                        print(
                            f"[VLN-DAPO] gen#{num_gen_batches}: +{len(kept_uids)} valid uids "
                            f"(filtered {n_filtered} zero-grad), total "
                            f"{len(accum_uid_order)}/{target_uids}"
                        )

                        # ── 3) not enough ⇒ fetch NEW episodes (never refill zero-grad) ──
                        if len(accum_uid_order) < target_uids:
                            if data_exhausted or (max_gen > 0 and num_gen_batches >= max_gen):
                                print(
                                    "[VLN-DAPO] sampling limit reached with only "
                                    f"{len(accum_uid_order)}/{target_uids} valid uids; training on partial batch"
                                )
                                if not accum:
                                    # nothing to train on this round; reset and keep going
                                    num_gen_batches, gen_filter_total = 0, 0
                                    continue
                            else:
                                continue  # pull the next dataloader batch (new episodes)

                        # ── 4) enough (or capped): concat → truncate by uid → re-pad ──
                        merged = DataProto.concat(accum) if len(accum) > 1 else accum[0]
                        chosen = accum_uid_order[:target_uids]
                        muids = np.asarray(merged.non_tensor_batch["uid"], dtype=object)
                        chosen_set = set(chosen)
                        sel = np.array([u in chosen_set for u in muids], dtype=bool)
                        step_batch = merged.select_idxs(sel)
                        step_batch = self._vln_repad(step_batch, self._pad_multiple())
                        step_batch.meta_info = dict(flat.meta_info)
                        # refresh length-dependent meta (flat's was for a different row count)
                        step_batch.meta_info["seqlen_sorted_indices"] = list(range(len(step_batch)))
                        # DAPO diagnostics for this trained step
                        metrics["dapo/num_gen_batches"] = num_gen_batches
                        metrics["dapo/valid_uids"] = len(chosen)
                        metrics["dapo/filtered_uids"] = gen_filter_total
                        denom = gen_filter_total + len(accum_uid_order)
                        metrics["dapo/filter_ratio"] = float(gen_filter_total / denom) if denom else 0.0
                        # reset accumulators for the next training step
                        accum, accum_uid_order, accum_uid_set = [], [], set()
                        num_gen_batches, gen_filter_total = 0, 0

                    # Free rollout replicas' device memory for the training forward/backward.
                    # Reached only when we are about to train (both ds and non-ds paths); the
                    # end-of-step update_weights() re-syncs/wakes replicas for the next step's gen.
                    self.checkpoint_manager.sleep_replicas()

                    # ── 5) train one step (VLN path, mirrors stock fit 1461-1631) ──
                    if self.config.trainer.balance_batch:
                        self._balance_batch(step_batch, metrics=metrics)
                    step_batch.meta_info["global_token_num"] = torch.sum(
                        step_batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    images_seqlens_all = []
                    if "multi_modal_inputs" in step_batch.non_tensor_batch:
                        for mmi in step_batch.non_tensor_batch["multi_modal_inputs"]:
                            if "image_grid_thw" not in mmi.keys():
                                continue
                            images_seqlens_all.extend(mmi["images_seqlens"].tolist())
                    step_batch.meta_info["images_seqlens"] = images_seqlens_all

                    with _rt.marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in step_batch.batch.keys():
                            step_batch = step_batch.union(self._compute_reward_colocate(step_batch))
                        reward_tensor, reward_extra_infos_dict = _rt.extract_reward(step_batch)

                    with _rt.marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(step_batch)
                        entropys = old_log_prob.batch["entropys"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = _rt.agg_loss(
                            loss_mat=entropys,
                            loss_mask=step_batch.batch["response_mask"],
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        metrics["perf/mfu/actor_infer"] = old_log_prob_mfu
                        old_log_prob.batch.pop("entropys")
                        step_batch = step_batch.union(old_log_prob)
                    assert "old_log_probs" in step_batch.batch, f"{step_batch.batch.keys()=}"

                    if self.use_reference_policy:
                        with _rt.marked_timer(str(_rt.Role.RefPolicy), timing_raw, color="olive"):
                            step_batch = step_batch.union(self._compute_ref_log_prob(step_batch))

                    with _rt.marked_timer("adv", timing_raw, color="brown"):
                        step_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            step_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                        if self.config.algorithm.use_kl_in_reward:
                            step_batch, kl_metrics = _rt.apply_kl_penalty(
                                step_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            step_batch.batch["token_level_rewards"] = step_batch.batch["token_level_scores"]
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        step_batch = _rt.compute_advantage(
                            step_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=rollout_n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.config.trainer.critic_warmup > self.global_steps:
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        with _rt.marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(step_batch)

                        esi_close = _rt.should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close
                        ):
                            with _rt.marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()
                        with _rt.marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)
                        metrics.update(_rt.reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(step_batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # ── validate ──
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with _rt.marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])

                # ── 6) metrics / log / bookkeeping (mirrors stock fit 1665-1715) ──
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(_rt.compute_data_metrics(batch=step_batch, use_critic=self.use_critic))
                vln_metrics = self._vln_metrics_from_batch(step_batch)
                if vln_metrics:
                    metrics.update(vln_metrics)
                metrics.update(_rt.compute_timing_metrics(batch=step_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    _rt.compute_throughout_metrics(batch=step_batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                logger.log(data=metrics, step=self.global_steps)
                timing_raw = {}
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    _rt.pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=step_batch)

            self._vln_epoch = epoch + 1
            self._vln_batches_in_epoch = 0

        # The final batches may all have been filtered. Preserve the last actor
        # update and exhausted data position even if no final training step fired.
        self.global_steps -= 1
        if self.global_steps > 0 and self.config.trainer.save_freq > 0:
            self._save_checkpoint()
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
        progress_bar.close()
        if self.global_steps == 0:
            raise RuntimeError("No training update was possible: all sampled episode groups were filtered")
