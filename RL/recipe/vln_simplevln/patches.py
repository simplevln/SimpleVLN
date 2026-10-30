"""Runtime patches injected into every verl process via VERL_USE_EXTERNAL_MODULES.

Runtime patches, applied in order:

1. (critical) Drop ``mm_token_type_ids`` from verl's training multimodal collation
   (``verl.utils.model.extract_multi_modal_inputs``). Qwen3-VL emits a per-token
   ``mm_token_type_ids`` whose length varies per decision, so verl's naive
   ``torch.cat`` fails once micro_batch>1. It is only used to derive (m)rope position
   ids, which the training forward already receives explicitly, so we drop it. This
   lets micro_batch>1 work. (Validate ppo_kl≈0 / entropy sane vs a micro=1 baseline.)

2. (optional, best-effort) Progress + ETA for the two otherwise-silent training
   phases, ``old_log_prob`` and ``update_actor``:
     - ``TrainingWorker.infer_batch`` / ``train_mini_batch`` set the phase label and an
       (approximate) total micro-batch count = ceil(rows_per_dp * epochs / micro_bs).
     - ``FSDPEngineWithLMHead.forward_step`` (one call per micro-batch) ticks the
       counter and prints ``done/total | elapsed | ETA | rate`` on rank 0 only.
   Plain periodic prints, not tqdm — Ray prefixes worker stdout per line and the launch
   pipes through ``tee``, which both break tqdm's ``\\r`` animation. (Rollout ETA lives
   in online_rollout_manager._generate_sequences_sliding.)

3. (recipe) Trajectory-level GRPO advantage. Default verl GRPO pools every decision
   row of an episode together, so the group baseline is weighted by decision count
   (a longer trajectory dominates). The VLN recipe wants each *trajectory* to count
   once (WMPO-style, cf. verl's own ``compute_advantage_for_multi_trajectories``):
   one reward per trajectory builds the group mean/std, and that advantage is assigned
   to every step of the trajectory. It then scales each step's advantage by
   ``decision_loss_weight`` (= 1/num_decisions) to reduce weighting by decision count.
   Response token lengths still affect token-mean loss aggregation. Toggle the
   trajectory baseline off with ``VLN_ADV_TRAJ_LEVEL=0`` or decision weighting off
   with ``VLN_TRAJ_LOSS_WEIGHT=0``.
"""

import math
import os
import sys
import time


# ──────────────────────────────────────────────────────────────────────────────
# Patch 1 (critical): drop mm_token_type_ids in training multimodal collation
# ──────────────────────────────────────────────────────────────────────────────
def _install_mm_token_type_ids_fix() -> None:
    import verl.utils.model as _model

    orig = _model.extract_multi_modal_inputs
    if getattr(orig, "_vln_patched", False):
        return

    def extract_multi_modal_inputs(batch_data, indices=None):
        cleaned = []
        for d in batch_data:
            base = d.data if (not isinstance(d, dict) and hasattr(d, "data")) else d
            if isinstance(base, dict) and "mm_token_type_ids" in base:
                cleaned.append({k: v for k, v in base.items() if k != "mm_token_type_ids"})
            else:
                cleaned.append(d)
        return orig(cleaned, indices)

    extract_multi_modal_inputs._vln_patched = True
    _model.extract_multi_modal_inputs = extract_multi_modal_inputs

    for name, mod in list(sys.modules.items()):
        if name.startswith("verl.workers.engine.") and getattr(mod, "extract_multi_modal_inputs", None) is orig:
            mod.extract_multi_modal_inputs = extract_multi_modal_inputs

    print(f"[vln-patch] extract_multi_modal_inputs patched (drop mm_token_type_ids), pid={os.getpid()}")


# ──────────────────────────────────────────────────────────────────────────────
# Patch 2 (optional): per-phase progress + ETA for old_log_prob / update_actor
# ──────────────────────────────────────────────────────────────────────────────
_PHASE = {"label": None, "total": None, "done": 0, "t0": 0.0, "last": 0.0}
_PRINT_EVERY_N = 8       # print at least every N micro-batches
_PRINT_EVERY_S = 20.0    # ...or every S seconds, whichever comes first


def _fmt(seconds: float) -> str:
    s = int(max(0, seconds))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _phase_start(label: str, total) -> None:
    now = time.time()
    _PHASE.update(label=label, total=total, done=0, t0=now, last=now)


def _progress_tick(rank: int) -> None:
    if rank != 0 or _PHASE["label"] is None:
        return
    st = _PHASE
    st["done"] += 1
    now = time.time()
    if st["done"] % _PRINT_EVERY_N != 0 and (now - st["last"]) < _PRINT_EVERY_S:
        return
    done, el = st["done"], now - st["t0"]
    rate = done / el if el > 0 else 0.0
    if st["total"]:
        total = st["total"]
        eta = (total - done) / rate if rate > 0 else 0.0
        pct = min(100.0, 100.0 * done / total)
        print(f"[vln-progress] {st['label']}: {done}/{total} ({pct:.0f}%) "
              f"| elapsed {_fmt(el)} | ETA {_fmt(eta)} | {rate:.1f} mb/s")
    else:
        print(f"[vln-progress] {st['label']}: {done} mb | elapsed {_fmt(el)} | {rate:.1f} mb/s")
    st["last"] = now


def _approx_total(self, data, micro_attr: str, epochs: int):
    """Approximate #micro-batches = ceil(rows_per_dp * epochs / micro_bs).

    Returns None when it can't be estimated (e.g. dynamic bsz), so we fall back to a
    count-only line with no ETA.
    """
    try:
        if getattr(self.engine_config, "use_dynamic_bsz", False):
            return None
        rows = int(data.shape[0])
        mbs = int(getattr(self.engine_config, micro_attr))
        if mbs <= 0:
            return None
        return max(1, math.ceil(rows * epochs / mbs))
    except Exception:
        return None


def _install_progress() -> None:
    import verl.workers.engine.fsdp.transformer_impl as _impl
    import verl.workers.engine_workers as _ew

    # (a) phase boundaries + totals on the TrainingWorker
    TW = _ew.TrainingWorker
    if not getattr(TW.infer_batch, "_vln_progress", False):
        _orig_infer = TW.infer_batch
        _orig_train = TW.train_mini_batch

        def infer_batch(self, data):
            _phase_start("old_log_prob/ref", _approx_total(self, data, "infer_micro_batch_size_per_gpu", 1))
            return _orig_infer(self, data)

        def train_mini_batch(self, data):
            # epochs defaults to ppo_epochs (1 in this recipe); ETA scales with it.
            _phase_start("actor-update", _approx_total(self, data, "micro_batch_size_per_gpu", 1))
            return _orig_train(self, data)

        infer_batch._vln_progress = True
        train_mini_batch._vln_progress = True
        TW.infer_batch = infer_batch
        TW.train_mini_batch = train_mini_batch

    # (b) per-micro-batch tick on the engine forward
    cls = _impl.FSDPEngineWithLMHead
    if not getattr(cls.forward_step, "_vln_progress", False):
        _orig_fs = cls.forward_step

        def forward_step(self, micro_batch, loss_function, forward_only):
            try:
                _progress_tick(getattr(self, "rank", -1))
            except Exception:
                pass
            return _orig_fs(self, micro_batch, loss_function, forward_only)

        forward_step._vln_progress = True
        cls.forward_step = forward_step

    print(f"[vln-patch] progress+ETA instrumented (old_log_prob / actor-update), pid={os.getpid()}")


# ──────────────────────────────────────────────────────────────────────────────
# Patch 3 (recipe): trajectory-level GRPO advantage
# ──────────────────────────────────────────────────────────────────────────────
def _install_traj_grpo_advantage() -> None:
    """Replace verl's row-pooled GRPO advantage with a trajectory-level one.

    The VLN recipe flattens one episode into N decision rows that all share the
    same episode ``uid``. Stock ``compute_grpo_outcome_advantage`` pools every row
    of a uid group and normalizes by row count, so a trajectory with more decisions
    contributes more samples to the baseline (length-weighted). Here we instead:

      * collapse each ``trajectory_uid`` to a single reward (mean of its rows),
      * build the group mean/std over *trajectories* (each trajectory == 1 sample),
      * assign ``(r_traj - mean_g) / (std_g + eps)`` to every step of that trajectory.

    Padding rows have unique uids, zero rewards and zero weights, so their advantages
    are zero. Non-GRPO estimators and batches without ``trajectory_uid`` fall through
    to the original implementation untouched. Disable with ``VLN_ADV_TRAJ_LEVEL=0``.
    """
    if os.environ.get("VLN_ADV_TRAJ_LEVEL", "1") != "1":
        return

    from collections import defaultdict

    import torch

    from verl.trainer.ppo import ray_trainer as _rt
    from verl.trainer.ppo.core_algos import AdvantageEstimator

    _orig_compute_advantage = _rt.compute_advantage

    def _traj_grpo_advantage(token_level_rewards, response_mask, index, traj_index,
                             decision_weight=None, epsilon=1e-6, norm_adv_by_std_in_grpo=True):
        scores = token_level_rewards.sum(dim=-1)  # (bs,) per-row outcome reward
        bsz = scores.shape[0]

        # group(uid) -> traj_uid -> list of that trajectory's row scores
        group_traj = defaultdict(lambda: defaultdict(list))
        for i in range(bsz):
            group_traj[index[i]][traj_index[i]].append(scores[i])

        group_mean, group_std = {}, {}
        for g, trajs in group_traj.items():
            # one reward per trajectory (rows of a traj are equal when KL is off;
            # mean is the robust aggregation otherwise)
            traj_rewards = torch.stack([torch.stack(v).mean() for v in trajs.values()])
            if traj_rewards.numel() == 1:
                # singleton group: no contrast — match stock GRPO (mean 0, std 1)
                group_mean[g] = scores.new_tensor(0.0)
                group_std[g] = scores.new_tensor(1.0)
            else:
                group_mean[g] = traj_rewards.mean()
                group_std[g] = traj_rewards.std()

        out = scores.clone()
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                out[i] = (scores[i] - group_mean[index[i]]) / (group_std[index[i]] + epsilon)
            else:
                out[i] = scores[i] - group_mean[index[i]]

        # Weight by inverse decision count, then normalize weights to mean one over
        # real rows. Token-mean loss still weights longer responses more heavily.
        if decision_weight is not None:
            w = torch.as_tensor(decision_weight, dtype=out.dtype, device=out.device)
            real = w > 0
            if real.any():
                w = w / w[real].mean()
            out = out * w

        adv = out.unsqueeze(-1) * response_mask
        return adv, adv

    def compute_advantage(data, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                          norm_adv_by_std_in_grpo=True, config=None):
        is_grpo = adv_estimator in (AdvantageEstimator.GRPO, "grpo")
        ntb = data.non_tensor_batch
        if is_grpo and "trajectory_uid" in ntb and "uid" in ntb:
            if "response_mask" not in data.batch.keys():
                data.batch["response_mask"] = _rt.compute_response_mask(data)
            use_w = os.environ.get("VLN_TRAJ_LOSS_WEIGHT", "1") == "1"
            decision_weight = ntb.get("decision_loss_weight") if use_w else None
            adv, ret = _traj_grpo_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=ntb["uid"],
                traj_index=ntb["trajectory_uid"],
                decision_weight=decision_weight,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            data.batch["advantages"] = adv
            data.batch["returns"] = ret
            return data
        return _orig_compute_advantage(
            data, adv_estimator=adv_estimator, gamma=gamma, lam=lam,
            num_repeat=num_repeat, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    _rt.compute_advantage = compute_advantage
    print(f"[vln-patch] trajectory-level GRPO advantage installed, pid={os.getpid()}")


# ──────────────────────────────────────────────────────────────────────────────
# Patch 4 (recipe): trajectory-level / padding-free data metrics
# ──────────────────────────────────────────────────────────────────────────────
def _install_correct_data_metrics() -> None:
    """Make verl's logged data metrics match the VLN flatten+pad batch semantics.

    The trainer logs ``compute_data_metrics`` over the flattened batch, which has two
    distortions for this recipe:

      1. Padding rows (appended in ``online_rollout_manager._build_output`` to pad the
         decision count to a multiple of the mini-batch) are real tensor rows with empty
         responses. They inflate ``response/aborted_ratio`` and drag down
         ``response_length/*`` and ``prompt_length/*``.
      2. reward/score/advantage are averaged per *decision row*, so a long episode is
         counted ``num_decisions`` times — inconsistent with the trajectory-level GRPO
         baseline (``_install_traj_grpo_advantage``), which treats each trajectory as one
         sample. With binary success reward (failures run longer → more rows),
         ``critic/rewards/mean`` ends up length-weighted and systematically below the true
         success rate reported by ``vln/traj_reward/mean``.

    This wraps ``compute_data_metrics``:
      * drop padding rows before delegating to the stock implementation — fixes #1 and
        keeps length/abort/prompt stats over real decisions only;
      * overwrite ``critic/{rewards,score,advantages}/{mean,max,min}`` with
        trajectory-collapsed values: each ``trajectory_uid`` contributes one reward (mean
        of its rows), then mean/max/min are taken over trajectories. Advantage is the
        per-trajectory GRPO advantage ``(r_traj - group_mean)/(group_std + eps)`` grouped
        by episode ``uid`` — the unweighted signal, i.e. without the decision weighting
        ``decision_loss_weight``, mirroring ``_traj_grpo_advantage`` (sample std;
        singleton groups use mean 0 and std 1).

    Falls through untouched for batches without ``is_padding`` (non-VLN / stock-GRPO runs).
    """
    from collections import defaultdict

    import numpy as np

    from verl.trainer.ppo import ray_trainer as _rt

    _orig_compute_data_metrics = _rt.compute_data_metrics

    def _traj_level_stats(batch):
        """Trajectory-collapsed reward/score/advantage stat dict, or None if unavailable."""
        ntb = batch.non_tensor_batch
        bb = batch.batch
        if "trajectory_uid" not in ntb or "uid" not in ntb:
            return None
        if "token_level_rewards" not in bb.keys() or "token_level_scores" not in bb.keys():
            return None

        uid = ntb["uid"]
        traj_uid = ntb["trajectory_uid"]
        seq_reward = bb["token_level_rewards"].sum(-1).detach().cpu().numpy()
        seq_score = bb["token_level_scores"].sum(-1).detach().cpu().numpy()

        # collapse rows → one reward/score per trajectory; remember each traj's group(uid)
        acc_r, acc_s, traj_group = defaultdict(list), defaultdict(list), {}
        for i in range(len(uid)):
            t = traj_uid[i]
            acc_r[t].append(float(seq_reward[i]))
            acc_s[t].append(float(seq_score[i]))
            traj_group[t] = uid[i]
        traj_reward = {t: float(np.mean(v)) for t, v in acc_r.items()}
        traj_score = {t: float(np.mean(v)) for t, v in acc_s.items()}

        # per-trajectory GRPO advantage grouped by episode uid (sample std ddof=1,
        # singleton group → mean 0/std 1, matching _traj_grpo_advantage)
        group_rewards = defaultdict(list)
        for t, r in traj_reward.items():
            group_rewards[traj_group[t]].append(r)
        group_stat = {}
        for g, rs in group_rewards.items():
            a = np.asarray(rs, dtype=np.float64)
            group_stat[g] = (0.0, 1.0) if a.size == 1 else (float(a.mean()), float(a.std(ddof=1)))
        traj_adv = []
        for t, r in traj_reward.items():
            m, s = group_stat[traj_group[t]]
            traj_adv.append((r - m) / (s + 1e-6))

        def _stat(prefix, vals):
            arr = np.asarray(vals, dtype=np.float64)
            return {f"{prefix}/mean": float(arr.mean()),
                    f"{prefix}/max": float(arr.max()),
                    f"{prefix}/min": float(arr.min())}

        out = {}
        out.update(_stat("critic/rewards", list(traj_reward.values())))
        out.update(_stat("critic/score", list(traj_score.values())))
        out.update(_stat("critic/advantages", traj_adv))
        return out

    def compute_data_metrics(batch, use_critic=True):
        ntb = getattr(batch, "non_tensor_batch", None) or {}
        if "is_padding" not in ntb:
            return _orig_compute_data_metrics(batch=batch, use_critic=use_critic)

        real = ~np.asarray(ntb["is_padding"], dtype=bool)
        real_batch = batch if real.all() else batch.select_idxs(real)
        metrics = _orig_compute_data_metrics(batch=real_batch, use_critic=use_critic)
        try:
            traj = _traj_level_stats(real_batch)
            if traj:
                metrics.update(traj)
        except Exception as exc:  # pragma: no cover - keep stock metrics on failure
            print(f"[vln-patch] traj-level metrics skipped: {exc!r}")
        return metrics

    _rt.compute_data_metrics = compute_data_metrics
    print(f"[vln-patch] trajectory-level / padding-free data metrics installed, pid={os.getpid()}")


# critical patch first (must always apply); progress is best-effort
_install_mm_token_type_ids_fix()
try:
    _install_progress()
except Exception as exc:  # pragma: no cover - progress is non-essential
    print(f"[vln-patch] progress instrumentation skipped: {exc!r}")
# Trajectory-level advantages are part of the objective; failure must stop training.
_install_traj_grpo_advantage()
try:
    _install_correct_data_metrics()
except Exception as exc:  # pragma: no cover - fall back to stock data metrics
    print(f"[vln-patch] traj-level data metrics skipped: {exc!r}")
