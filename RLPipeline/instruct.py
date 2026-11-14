#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Label sampled routes using dataset-specific references and waypoint observations.
"""Write instructions and pack them back into the trajectory files.

The fixed prompt is prompt.txt. Same house, at most eight style lines:
R2R train, English RxR train, or the original ScaleVLN corpus.

  python3 instruct.py label --dataset r2r
  python3 instruct.py assemble --dataset r2r
"""
import argparse
import base64
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

import nav

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("VLN_DATA", "/root/code/data")
PROMPT_BASE = open(os.path.join(_HERE, "prompt.txt"), encoding="utf-8").read().strip()
IMG_W, IMG_H = 640, 480
TAIL = "\n\n---\n\n# Process this route\n\nInput:"
STYLE_HINT = (
    "\n\nFirst read the same-scene dataset references above, noting their wording, landmark selection, directions, and stopping descriptions. "
    "Follow that style while writing this instruction so it reads like a same-scene reference annotation; learn the style without copying its content."
)

def dataset_spec(name, data_root):
    root = os.path.join(data_root, "datasets")
    specs = {
        "r2r": dict(src=os.environ.get("R2R_SRC", f"{root}/r2r_rl_v3/r2r"),
                    out=os.environ.get("R2R_OUT", f"{root}/r2r_rl_final"),
                    sim="mp3d", mode="instance", refs="r2r"),
        "rxr": dict(src=os.environ.get("RXR_SRC", f"{root}/r2r_rl_v3/rxr"),
                    out=os.environ.get("RXR_OUT", f"{root}/rxr_rl_final"),
                    sim="mp3d", mode="instance", refs="rxr"),
        "scalevln": dict(src=os.environ.get("SCALEVLN_SRC", f"{root}/scalevln_rl_v3"),
                         out=os.environ.get("SCALEVLN_OUT", f"{root}/scalevln_rl_final"),
                         sim="hm3d", mode="rgb", refs="scalevln"),
    }
    if name not in specs:
        raise SystemExit(f"dataset must be one of {', '.join(specs)}")
    return specs[name]


def scene_key(scene_id):
    parts = str(scene_id).replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else parts[-1]


def instr_text(ep):
    ins = ep.get("instruction", "")
    if isinstance(ins, dict):
        return str(ins.get("instruction_text", "")).strip()
    return str(ins).strip()


def load_levels(src):
    """Return [(lvl, rl_dict, gt_dict)] from a root directory or lvl_* subdirectories."""
    if os.path.exists(os.path.join(src, "rl.json.gz")):
        dirs = [src]
    else:
        dirs = sorted(p for p in glob.glob(os.path.join(src, "lvl_*")) if os.path.isdir(p))
    out = []
    for d in dirs:
        rl_path, gt_path = os.path.join(d, "rl.json.gz"), os.path.join(d, "rl_gt.json.gz")
        if not (os.path.exists(rl_path) and os.path.exists(gt_path)):
            raise FileNotFoundError(f"Incomplete dataset level: {d}; need rl.json.gz and rl_gt.json.gz")
        lvl = "" if d == src else os.path.basename(d)
        with gzip.open(rl_path, "rt") as f:
            rl = json.load(f)
        with gzip.open(gt_path, "rt") as f:
            gt = json.load(f)
        out.append((lvl, rl, gt))
    if not out:
        raise SystemExit(f"In {src}, no rl.json.gz + rl_gt.json.gz pair was found")
    return out


def index_episodes(levels):
    by_scene = defaultdict(list)
    for lvl, rl, gt in levels:
        for ep in rl["episodes"]:
            eid = str(ep["episode_id"])
            if eid not in gt:
                raise ValueError(f"Missing GT: {lvl}/{eid}")
            by_scene[scene_key(ep["scene_id"])].append((lvl, ep, gt[eid]))
    return by_scene


def load_corpus_refs(data_root, rels):
    """Load same-house style references from the first existing path in rels; keep English entries when language is specified."""
    path = None
    for rel in rels:
        cand = rel if os.path.isabs(rel) else os.path.join(data_root, "datasets", rel)
        if os.path.exists(cand):
            path = cand
            break
    refs = defaultdict(list)
    if path is None:
        print(f"[refs] Missing {' / '.join(rels)}; same-scene references will be empty", flush=True)
        return refs
    with gzip.open(path, "rt") as f:
        episodes = json.load(f)["episodes"]
    for ep in episodes:
        ins = ep.get("instruction", {})
        if isinstance(ins, dict):
            lang = str(ins.get("language", "en")).lower()
            if lang not in ("en", "en-us", "en-in", ""):
                continue
        text = instr_text(ep)
        if text:
            refs[scene_key(ep["scene_id"])].append(text)
    print(f"[refs] {path}  {sum(len(v) for v in refs.values())} instructions / {len(refs)} scenes", flush=True)
    return refs


def load_self_refs(by_scene):
    refs = defaultdict(list)
    for scene, rows in by_scene.items():
        for _, ep, _ in rows:
            text = instr_text(ep)
            if text:
                refs[scene].append(text)
    return refs


def header_for(scene, self_ref, refs, k=8):
    seen, picked = set(), []
    for text in refs.get(scene, []):
        if not text or text == self_ref or text in seen:
            continue
        seen.add(text)
        picked.append(text)
        if len(picked) >= k:
            break
    block = "\n".join(f"- {t}" for t in picked) if picked else "(No dataset references available for this scene)"
    return (PROMPT_BASE
            + "\n\n# Same-scene dataset references (learn the style; do not copy)\n"
            + block + STYLE_HINT + TAIL)


def jpeg_b64(bgr):
    if bgr.shape[1] != IMG_W or bgr.shape[0] != IMG_H:
        bgr = cv2.resize(bgr, (IMG_W, IMG_H))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def img_msg(bgr):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64(bgr)}}


def node_text(node):
    inst = ", ".join(node["instances"]) if node["instances"] else "(none)"
    loc = str(node.get("Location") or "unknown").split(" / ")[0]
    return f"Action: {node['Action']}\nLocation: {loc}\nInstances: {inst}"


def simple_nodes(gt, turn_thresh=30.0, step_deg=15.0, fwd_seg=8):
    """Split nodes by GT actions when semantics are unavailable; ignore turns below the net-angle threshold."""
    actions = gt["actions"]
    nodes = [{"step": 0, "Action": "Start", "Location": "unknown"}]
    net, fwd = [0.0], [0]

    def flush(step):
        if abs(net[0]) >= turn_thresh:
            if abs(net[0]) >= 135:
                name = "Turn around"
            else:
                name = "Turn left" if net[0] > 0 else "Turn right"
            nodes.append({"step": step, "Action": name, "Location": "unknown"})
        net[0] = 0.0

    step = 0
    for act in actions:
        if act == 0:
            break
        step += 1
        if act == 2:
            net[0] += step_deg
            fwd[0] = 0
        elif act == 3:
            net[0] -= step_deg
            fwd[0] = 0
        else:
            flush(step - 1)
            fwd[0] += 1
            if fwd[0] >= fwd_seg:
                nodes.append({"step": step, "Action": "Walk forward", "Location": "unknown"})
                fwd[0] = 0
    last = step
    flush(last)
    if nodes[-1]["step"] != last:
        nodes.append({"step": last, "Action": "Walk forward and Stop", "Location": "unknown"})
    else:
        nodes[-1]["Action"] += " and Stop"
    return nodes


def request_vlm(content, key, base_url, model):
    err = "no attempt"
    for i in range(4):
        try:
            resp = requests.post(
                base_url,
                json={"model": model, "messages": [{"role": "user", "content": content}],
                      "max_tokens": 2048, "temperature": 0.2},
                headers={"Authorization": f"Bearer {key}"},
                timeout=300,
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip().strip('"').strip()
                if not text or text == "__ERR__":
                    raise ValueError("Empty or invalid VLM response")
                return text
            err = f"http {resp.status_code}: {resp.text[:160]}"
        except Exception as exc:
            err = f"exc: {exc}"
        time.sleep(2 * (i + 1))
    print(f"  [vlm] {err}", flush=True)
    return "__ERR__"


def replay(sim, ep, gt, need_rgb, node_steps, grab_sem):
    ag = sim.get_agent(0)
    sr = ep["start_rotation"]
    st = ag.get_state()
    st.position = np.array(ep["start_position"], np.float32)
    st.rotation = np.quaternion(sr[3], sr[0], sr[1], sr[2])
    ag.set_state(st)
    frames, at = {}, {}

    def grab(k):
        obs = sim.get_sensor_observations()
        if k in need_rgb:
            frames[k] = cv2.cvtColor(np.asarray(obs["rgb"])[:, :, :3], cv2.COLOR_RGB2BGR)
        if grab_sem and k in node_steps:
            at[k] = (np.asarray(obs["sem"]), ag.get_state().position, np.asarray(obs["depth"]))

    grab(0)
    k = 0
    for act in gt["actions"]:
        if act == 0:
            break
        sim.step(nav.AMAP[act])
        k += 1
        grab(k)
    return frames, at, k


def build_content(sim, ep, gt, maps, id2label, header, turn_deg, use_instances):
    if maps is not None:
        nodes = nav.process_episode(sim, ep, gt, maps, 4, 8.0, 0.006, True, turn_deg=turn_deg)[0]
    else:
        nodes = simple_nodes(gt, turn_thresh=turn_deg)
    if not nodes:
        return None, 0
    node_steps = {n["step"] for n in nodes}
    last = max(node_steps)
    need_rgb = set(node_steps) | {max(0, last - 1), max(0, last - 2)}
    frames, at, k = replay(sim, ep, gt, need_rgb, node_steps, maps is not None and use_instances)
    blank = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    for fid in list(need_rgb) + [k]:
        frames.setdefault(min(fid, k), blank)
    id2cat = id2ctr = None
    if maps is not None:
        id2cat, id2ctr = maps[0], maps[1]
    content = [{"type": "text", "text": header}]
    for i, node in enumerate(nodes):
        step = min(node["step"], k)
        insts = []
        if use_instances and step in at:
            sem, cam, dep = at[step]
            insts = nav.visible_instances(sem, cam, id2cat, id2ctr, id2label, depth=dep)
        node = dict(node, instances=insts)
        content.append({"type": "text", "text": node_text(node)})
        frame_ids = [max(0, step - 2), max(0, step - 1), step] if i == len(nodes) - 1 else [step]
        for fid in frame_ids:
            content.append(img_msg(frames[min(fid, k)]))
    content.append({"type": "text", "text": "\nOutput:"})
    return content, len(nodes)


def progress_rows(path):
    with open(path, "rb") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                continue  # A process may have died in the middle of its final write.
            raise ValueError(f"Malformed progress record: {path}:{i + 1}")


def repair_progress_tail(path):
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
        except (json.JSONDecodeError, UnicodeDecodeError):
            f.truncate(start)
        else:
            f.seek(0, os.SEEK_END)
            f.write(b"\n")


def load_done(out_dir):
    done = set()
    for path in glob.glob(os.path.join(out_dir, "_progress*.jsonl")):
        for row in progress_rows(path):
            if str(row.get("instruction") or "").strip() and row["instruction"] != "__ERR__":
                done.add((row.get("lvl", ""), str(row["eid"])))
    return done


def open_sim(sim_kind, scene, gpu, need_sem, hm3d_root, data_root=None):
    import habitat_sim
    if sim_kind == "mp3d":
        glb = (os.path.join(data_root, "scene_datasets", "mp3d", scene, f"{scene}.glb")
               if data_root is not None else nav.scene_path(scene))
        if not os.path.exists(glb):
            raise FileNotFoundError(glb)
        if need_sem:
            sim = habitat_sim.Simulator(nav.make_sem_cfg(glb, with_rgb=True, gpu=gpu))
            maps, id2label, _ = nav.build_maps_inst(sim)
            house = glb.replace(".glb", ".house")
            if os.path.exists(house):
                nav.inject_raw_landmarks(sim, maps[0], id2label, house)
            return sim, maps, id2label
        cfg = nav.make_cfg(glb, with_rgb=True)
        try:
            cfg.sim_cfg.gpu_device_id = gpu
        except Exception:
            pass
        return habitat_sim.Simulator(cfg), None, None

    hits = glob.glob(os.path.join(hm3d_root, "*", scene, "*.basis.glb"))
    if not hits:
        raise FileNotFoundError(f"hm3d glb for {scene}")
    glb = hits[0]
    navs = glob.glob(os.path.join(os.path.dirname(glb), "*.basis.navmesh"))
    cfg = nav.make_cfg(glb, with_rgb=True)
    try:
        cfg.sim_cfg.gpu_device_id = gpu
    except Exception:
        pass
    sim = habitat_sim.Simulator(cfg)
    if navs and not sim.pathfinder.is_loaded:
        sim.pathfinder.load_nav_mesh(navs[0])
    # This HM3D configuration has RGB only. ScaleVLN uses action-based nodes even when instance mode is requested.
    return sim, None, None


def label(args):
    key = os.environ.get("VLM_API_KEY", "").strip()
    base_url = os.environ.get("VLM_BASE_URL", "").strip()
    model = os.environ.get("VLM_MODEL", "").strip()
    if not key or not base_url or not model:
        raise SystemExit("Set VLM_API_KEY, VLM_BASE_URL, and VLM_MODEL first")
    spec = dataset_spec(args.dataset, args.data_root)
    src = args.src or spec["src"]
    out = args.out or spec["out"]
    mode = args.mode or spec["mode"]
    use_instances = mode == "instance"
    os.makedirs(out, exist_ok=True)

    levels = load_levels(src)
    by_scene = index_episodes(levels)
    if spec["refs"] == "r2r":
        refs = load_corpus_refs(args.data_root, ["r2r/train/train.json.gz"])
    elif spec["refs"] == "rxr":
        refs = load_corpus_refs(args.data_root, [
            "rxr/train/train_guide_en.json.gz",
            "rxr/train/train_guide.json.gz",
        ])
    else:
        ref_path = os.environ.get("SCALEVLN_REFS", "scalevln/ScaleVLN/scalevln_subset_150k.json.gz")
        refs = load_corpus_refs(args.data_root, [ref_path])
        if not refs:
            raise SystemExit("No ScaleVLN style references; set SCALEVLN_REFS to a source corpus JSON.gz")
    done = load_done(out)
    total = sum(len(v) for v in by_scene.values())
    worker = args.worker
    nworkers = args.workers if worker >= 0 else 1
    gpu = int(os.environ.get("GPU_ID", args.gpu))
    hm3d_root = os.path.join(args.data_root, "scene_datasets", "hm3d")
    prog = os.path.join(out, "_progress.jsonl" if worker < 0 else f"_progress.w{worker}.jsonl")
    print(f"[label {args.dataset}] {total} eps / {len(by_scene)} scenes, completed {len(done)}, "
          f"mode={mode}, turn_deg={args.turn_deg}, worker={worker}/{nworkers}", flush=True)

    scenes = sorted(by_scene, key=lambda s: -len(by_scene[s]))
    n_ok = n_err = 0
    repair_progress_tail(prog)
    with open(prog, "a", encoding="utf-8") as pf:
        for si, scene in enumerate(scenes):
            if worker >= 0 and si % nworkers != worker:
                continue
            pending = [(lvl, ep, gt) for lvl, ep, gt in by_scene[scene]
                       if (lvl, str(ep["episode_id"])) not in done]
            if args.limit and len(pending) > args.limit:
                pending = pending[:args.limit]
            if not pending:
                continue
            print(f"[scene {si + 1}/{len(scenes)}] {scene}: {len(pending)} pending", flush=True)
            try:
                sim, maps, id2label = open_sim(spec["sim"], scene, gpu, use_instances, hm3d_root, args.data_root)
            except Exception:
                n_err += len(pending)
                print(f"  [{scene}] Cannot open scene\n{traceback.format_exc()[-400:]}", flush=True)
                continue
            if use_instances and maps is None:
                print(f"  [{scene}] No semantics; using RGB for this scene", flush=True)
            try:
                for start in range(0, len(pending), args.batch):
                    chunk = pending[start:start + args.batch]
                    jobs = []
                    for lvl, ep, gt in chunk:
                        try:
                            header = header_for(scene, instr_text(ep), refs)
                            content, _ = build_content(
                                sim, ep, gt, maps, id2label, header, args.turn_deg,
                                use_instances and maps is not None)
                            if content is None:
                                raise ValueError("no nodes")
                            jobs.append((lvl, str(ep["episode_id"]), content))
                        except Exception:
                            pf.write(json.dumps({
                                "lvl": lvl, "eid": str(ep["episode_id"]), "scene": scene,
                                "instruction": "__ERR__", "err": traceback.format_exc()[-200:],
                            }, ensure_ascii=False) + "\n")
                            n_err += 1
                    with ThreadPoolExecutor(max_workers=args.conc) as pool:
                        futs = {pool.submit(request_vlm, content, key, base_url, model): (lvl, eid)
                                for lvl, eid, content in jobs}
                        for fut in futs:
                            lvl, eid = futs[fut]
                            text = fut.result()
                            pf.write(json.dumps({
                                "lvl": lvl, "eid": eid, "scene": scene, "instruction": text,
                            }, ensure_ascii=False) + "\n")
                            n_err += text == "__ERR__"
                            n_ok += text != "__ERR__"
                    pf.flush()
                    print(f"    {scene} +{len(jobs)}  ok={n_ok} err={n_err}", flush=True)
            finally:
                sim.close()
    print(f"[label] DONE ok={n_ok} err={n_err}", flush=True)
    if n_err:
        raise SystemExit(f"Labeling failed for {n_err} episodes; retry label before assembly")


def assemble(args):
    spec = dataset_spec(args.dataset, args.data_root)
    src = args.src or spec["src"]
    out = args.out or spec["out"]
    source = "synthetic"
    gen = {}
    for path in sorted(glob.glob(os.path.join(out, "_progress*.jsonl"))):
        for row in progress_rows(path):
            if str(row.get("instruction") or "").strip() and row["instruction"] != "__ERR__":
                gen[(row.get("lvl", ""), str(row["eid"]))] = row["instruction"]
    n_ok = n_keep = 0
    all_eps, vocab = [], None
    all_gt = {}
    levels = load_levels(src)
    for lvl, rl, _gt in levels:
        if vocab is None:
            vocab = rl.get("instruction_vocab")
        out_dir = out if lvl == "" else os.path.join(out, lvl)
        os.makedirs(out_dir, exist_ok=True)
        level_gt = {}
        for ep in rl["episodes"]:
            old_id = str(ep["episode_id"])
            new_id = old_id
            if new_id in all_gt:
                raise ValueError(f"Duplicate episode ID: {new_id}")
            if old_id not in _gt:
                raise ValueError(f"Missing GT: {new_id}")
            text = gen.get((lvl, str(ep["episode_id"])))
            ins = ep.setdefault("instruction", {})
            if not isinstance(ins, dict):
                ins = {"instruction_text": str(ins)}
                ep["instruction"] = ins
            if text:
                ins["original_text"] = ins.get("instruction_text", "")
                ins["instruction_text"] = text
                ins["source"] = source
                n_ok += 1
            else:
                n_keep += 1
            if not str(ins.get("instruction_text", "")).strip():
                raise ValueError(f"Empty instruction: {new_id}; rerun label first")
            level_gt[new_id] = _gt[old_id]
            all_gt[new_id] = _gt[old_id]
        all_eps.extend(rl["episodes"])
        print(f"  Assemble {lvl or '.'}: {len(rl['episodes'])} eps", flush=True)
    # Rebuild tokens after text replacement, with one vocabulary per dataset.
    if any("instruction_tokens" in ep["instruction"] for ep in all_eps):
        vocab = {"word_list": nav.fill_vocab(all_eps)}
    for lvl, rl, _gt in levels:
        if vocab is not None:
            rl["instruction_vocab"] = vocab
        out_dir = out if lvl == "" else os.path.join(out, lvl)
        with gzip.open(os.path.join(out_dir, "rl.json.gz"), "wt") as f:
            json.dump(rl, f, ensure_ascii=False)
        level_gt = {str(ep["episode_id"]): all_gt[str(ep["episode_id"])] for ep in rl["episodes"]}
        with gzip.open(os.path.join(out_dir, "rl_gt.json.gz"), "wt") as f:
            json.dump(level_gt, f, ensure_ascii=False)
    with gzip.open(os.path.join(out, "rl_all.json.gz"), "wt") as f:
        json.dump({"episodes": all_eps, "instruction_vocab": vocab}, f, ensure_ascii=False)
    with gzip.open(os.path.join(out, "rl_all_gt.json.gz"), "wt") as f:
        json.dump(all_gt, f, ensure_ascii=False)
    print(f"[assemble] {out}  replaced {n_ok}, retained source instructions {n_keep}, total {len(all_eps)}", flush=True)


def launch(args):
    n_gpu = 1
    try:
        n_gpu = max(1, len(subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()))
    except Exception:
        pass
    out = args.out or dataset_spec(args.dataset, args.data_root)["out"]
    os.makedirs(out, exist_ok=True)
    procs = []
    for i in range(args.workers):
        env = dict(os.environ, GPU_ID=str(i % n_gpu))
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "label",
               "--dataset", args.dataset, "--data-root", args.data_root,
               "--worker", str(i), "--workers", str(args.workers),
               "--conc", str(args.conc), "--batch", str(args.batch),
               "--turn-deg", str(args.turn_deg), "--gpu", str(i % n_gpu)]
        if args.mode:
            cmd += ["--mode", args.mode]
        if args.src:
            cmd += ["--src", args.src]
        if args.out:
            cmd += ["--out", args.out]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        logf = open(os.path.join(out, f"_label_w{i}.log"), "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT))
    print(f"[launch] {args.workers} workers, {n_gpu} gpu", flush=True)
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker exit codes {codes}")


def main():
    ap = argparse.ArgumentParser(description="R2R / RxR / ScaleVLN instruction labeling")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(dataset={"choices": ["r2r", "rxr", "scalevln"], "required": True})
    for name in ("label", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", **common["dataset"])
        p.add_argument("--data-root", default=DATA_ROOT)
        p.add_argument("--src", default="", help="Override the default trajectory directory")
        p.add_argument("--out", default="", help="Override the default output directory")
        p.add_argument("--mode", default="", choices=["", "instance", "rgb"])
        p.add_argument("--turn-deg", type=int, default=45)
    lab = sub.choices["label"]
    lab.add_argument("--workers", type=int, default=1)
    lab.add_argument("--worker", type=int, default=-1, help="Set by launch; leave unset for a single process")
    lab.add_argument("--conc", type=int, default=8)
    lab.add_argument("--batch", type=int, default=40)
    lab.add_argument("--gpu", type=int, default=0)
    lab.add_argument("--limit", type=int, default=0, help="Maximum labels per scene; 0 means unlimited")
    args = ap.parse_args()
    if args.cmd == "assemble":
        assemble(args)
    elif args.workers > 1 and args.worker < 0:
        launch(args)
    else:
        label(args)


if __name__ == "__main__":
    main()
