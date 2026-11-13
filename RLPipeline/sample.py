#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample trajectories, then pack them.

  python3 sample.py mp3d   ...
  python3 sample.py merge  ...
  python3 sample.py hm3d   ...
  python3 sample.py dedup  ...
  python3 sample.py pack   ...
"""
import argparse
import hashlib
import glob
import gzip
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import habitat_sim
import habitat_sim.utils.common as hsc

from quotas import ratio_caps

import nav as C

# =========================================================================
# rl: room-to-room RL 数据 (导航图 + 多跳 + 物体 goal + 地标指令 + 去重)
# =========================================================================
GOAL_RADIUS_RL = 0.5
STEP_MIN, STEP_MAX = 10, 400
# reference_path 重采样: 对齐原始 R2R/RxR 的 ~1.8m 相邻点间距与点数 (别用 RDP 抽稀成 2 点)
REF_STEP = 1.8          # 相邻 waypoint 目标弧长间距 (m)
REF_MIN_PTS = 5         # 每条轨迹最少 waypoint 数 (短轨迹兜底)
RL_MIN_GEO = 3.0        # geodesic 下限(m); 提高可滤掉超短轨迹, 右移步数分布补中等桶
RL_STALE_MAX = 800      # 连续无新增即放弃; rxr 多房 walk 慢, 调低(如150)加速饱和场景放弃
LEVELS = [(i, i + 50) for i in range(0, 400, 50)]   # 0-50,50-100,...,350-400


def level_name(steps):
    for lo, hi in LEVELS:
        if lo <= steps < hi or (hi == LEVELS[-1][1] and steps == hi):
            return f"lvl_{lo:03d}-{hi:03d}"
    return None


# 每类经验步/米 (waypoint 轨迹: rxr 转弯多 -> steps/m 更高), 用于"目标桶 -> 目标 geodesic 带"的定向提案
SPM = {"r2r": 5.8, "rxr": 9.5}


def parse_target_dist(s):
    """解析 --target-dist: 'lo:count' 逗号分隔, lo=桶下界(0,50,100,...).
    返回 {lvl_name: count}; 空串返回 None(退化为原始无配额行为)。"""
    if not s:
        return None
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        lo, cnt = part.split(":")
        lo, cnt = int(lo), int(cnt)
        name = next((f"lvl_{a:03d}-{b:03d}" for a, b in LEVELS if a == lo), None)
        if name and cnt > 0:
            out[name] = cnt
    return out or None


def load_existing_trajs(dataset_dir, want_scenes=None):
    """从已生成数据集(lvl_*/rl.json.gz + rl_gt.json.gz)按场景读出 (start, goal, locations),
    用于跨 run 去重: 把现有轨迹指纹种进 Dedup。want_scenes 限定只读这些场景(分片加速)。"""
    out = defaultdict(list)
    if not dataset_dir:
        return out
    efs = glob.glob(os.path.join(dataset_dir, "lvl_*", "rl.json.gz"))
    if not efs:                                   # 兼容父目录(含 r2r/ rxr/ 子目录)布局
        efs = glob.glob(os.path.join(dataset_dir, "*", "lvl_*", "rl.json.gz"))
    for ef in sorted(efs):
        gf = ef.replace("rl.json.gz", "rl_gt.json.gz")
        d = json.load(gzip.open(ef, "rt"))
        gt = json.load(gzip.open(gf, "rt")) if os.path.exists(gf) else {}
        for ep in d["episodes"]:
            sc = ep["scene_id"].split("/")[1]
            if want_scenes and sc not in want_scenes:
                continue
            g = gt.get(str(ep["trajectory_id"]))
            locs = g["locations"] if g and "locations" in g else None
            out[sc].append((ep["start_position"], ep["goals"][0]["position"], locs))
    return out


def load_real_refpaths(want_scenes=None):
    """读真实 R2R train + RxR guide 的 (start, goal, reference_path), 按场景分组。
    reference_path 稀疏(几点), 用法侧用 navmesh 加密后做轨迹指纹去重。want_scenes 限定加速。"""
    out = defaultdict(list)
    for ds in ["r2r/train/train.json.gz", "rxr/train/train_guide.json.gz"]:
        p = os.path.join(C.DATA, "datasets", ds)
        if not os.path.exists(p):
            continue
        for e in json.load(gzip.open(p, "rt"))["episodes"]:
            sc = e["scene_id"].split("/")[1]
            if want_scenes and sc not in want_scenes:
                continue
            rp = e.get("reference_path")
            if rp and len(rp) >= 2:
                out[sc].append((e["start_position"], e["goals"][0]["position"], rp))
    return out


def _rl_instruction(wp_pos, wp_room, regs, goal_obj, goal_room):
    lms = []
    for i, p in enumerate(wp_pos):
        lm, rn = C.nearest_landmark(p, regs, wp_room[i], exclude={lms[-1]} if lms else set())
        lms.append(lm or rn)
    parts = [f"Begin near the {lms[0]}"]
    for i in range(1, len(wp_pos) - 1):
        tw = C.turn_word(C.heading(wp_pos[i - 1], wp_pos[i]), C.heading(wp_pos[i], wp_pos[i + 1]))
        lead = (tw + " at" if tw else "continue past") + f" the {lms[i]}"
        nxt = lms[i + 1] if i + 1 < len(lms) - 1 else goal_obj
        parts.append(f"{lead}, then head toward the {nxt}")
    parts.append(f"stop at the {goal_obj} in the {goal_room}")
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."


def _rl_scene_waypoint(scene, sim, regs, nodes, edges, node_room, dedup, quota, rng, category,
                       lvl_quota=None, spm=None, walk_fn=None, band_hi_mult=1.30):
    """与 _rl_scene 并行的多 waypoint 版本: 沿 start->goal 的房间链, 每个途经房间选一个
    物体/地标作 waypoint, 逐段最短路串联, 最后一个 waypoint=goal。按段各自算最短路,
    不限制总绕路比。r2r(want=1)无中间房 -> 退化为单段; rxr(want>=2)产生真正多房途经轨迹。
    walk_fn: 走轨迹的函数(默认 greedy follower); 传 C.walk_pathfollow 用快通道。
    band_hi_mult: geo 预筛上界系数(收紧减少走完才因步数丢弃的浪费)。"""
    pf, agent = sim.pathfinder, sim.get_agent(0)
    walk_fn = walk_fn or C.walk_follower
    follower = sim.make_greedy_follower(agent_id=0, goal_radius=GOAL_RADIUS_RL, stop_key="stop",
                                        forward_key="move_forward", left_key="turn_left",
                                        right_key="turn_right")
    adj = defaultdict(list); radj = defaultdict(set)
    for a, b in edges:
        w = float(np.linalg.norm(nodes[a] - nodes[b]))
        adj[a].append((b, w)); adj[b].append((a, w))
        ra, rb = int(node_room[a]), int(node_room[b])
        if ra != rb and ra >= 0 and rb >= 0:
            radj[ra].add(rb); radj[rb].add(ra)
    rooms = sorted({int(node_room[i]) for i in range(len(nodes))
                    if node_room[i] >= 0 and regs[int(node_room[i])]["objs"]})
    if len(rooms) < 2:
        return []
    nin = lambda rm: [i for i in range(len(nodes)) if node_room[i] == rm]

    def room_of(p):
        for ri, rr in enumerate(regs):
            if np.all(p >= rr["lo"]) and np.all(p <= rr["hi"]):
                return ri
        return -1

    def room_waypoint(r):
        """中间房选一个非墙物体转可走站点; 失败回退房心; 再失败返回 None(跳过该房)。"""
        objs = [(cn, oc) for cn, oc in regs[r]["objs"] if cn not in C.WALL_OBJECTS]
        for oi in rng.permutation(len(objs)):
            gp = C.reachable_goal(pf, objs[oi][1])
            if gp is not None:
                return np.asarray(gp, np.float32)
        s = pf.snap_point(np.asarray(regs[r]["c"], np.float32))
        if not np.any(np.isnan(s)) and pf.is_navigable(s):
            return np.asarray(s, np.float32)
        return None

    def try_pair(s_node, g_room, goal_obj, gp, waypoints, planned):
        """沿 waypoints 逐段走 start->...->goal 并过滤; 通过返回 record, 否则 None。"""
        sp = np.array(nodes[s_node], np.float32)
        if dedup.is_dup(sp, gp):              # 端点/数据集/goal聚集 先廉价拦
            return None
        yaw = float(rng.uniform(0, 2 * math.pi))
        actions, locations = walk_fn(sim, follower, agent, sp, yaw, waypoints)
        n_steps = len([x for x in actions if x != 0])
        if not (STEP_MIN <= n_steps <= STEP_MAX):
            return None
        if np.linalg.norm(np.array(locations[-1]) - gp) > 0.4:      # 必须真到达最终 goal
            return None
        locs = np.array(locations, np.float32)
        gdist = C.geo(pf, sp, gp)
        if gdist < RL_MIN_GEO:                # 太近的丢弃; 多段不再限制总绕路比
            return None
        if dedup.is_dup(sp, gp, traj=locs):   # 走完后再查轨迹重合
            return None
        keep = C.rdp(locs, eps=0.6)                                 # 角点(保留转弯结构)
        ref_path = C.resample_waypoints(locs, keep, step=REF_STEP,  # 按弧长 ~1.8m 重采样
                                        min_pts=REF_MIN_PTS)        # 禁止退化成 2 点
        # 指令用规划好的 waypoint(房间索引已知, 避免 room_of 在 AABB 重叠处误判)
        instr_pos = [sp.tolist()] + [pt.tolist() for _, pt in planned] + [gp.tolist()]
        instr_room = [room_of(sp)] + [r for r, _ in planned] + [g_room]
        chain = []
        for r in instr_room:
            if r >= 0 and (not chain or chain[-1] != r):
                chain.append(r)
        if (category == "rxr" and len(set(chain)) < 3) or \
           (category == "r2r" and len(set(chain)) < 2):
            return None
        if len(instr_pos) < 3:                # r2r 不足 3 点时插中点(保持指令丰富度)
            mid = locs[len(locs) // 2]
            instr_pos.insert(-1, mid.tolist()); instr_room.insert(-1, room_of(mid))
        instr = _rl_instruction(instr_pos, instr_room, regs, goal_obj, regs[g_room]["name"])
        dedup.add(sp, gp, traj=locs)
        return {"scene": scene, "start": sp.tolist(), "yaw": yaw,
                "goal": gp.tolist(), "goal_obj": goal_obj,
                "rooms": [regs[r]["name"] for r in chain], "reference_path": ref_path,
                "instruction": instr, "geodesic": float(gdist),
                "n_steps": n_steps, "actions": actions, "locations": locations}

    # 分级重试: 外层选可达 start, 内层试多个 goal (复用好起点)
    # targeted=True: 按 lvl_quota(每桶剩余) 定向生成 -> 目标桶加权选 -> 映射目标 geodesic 带 ->
    #                提案前用 geo 廉价预筛(避免走满桶/空走), 落桶已满则拒(已入 dedup 不重复)。
    LVL_STEP = {f"lvl_{lo:03d}-{hi:03d}": (lo, hi) for lo, hi in LEVELS}
    targeted = lvl_quota is not None
    total_need = sum(lvl_quota.values()) if targeted else quota
    max_tries = max(int(total_need * 60), 400)
    out, tries, stale = [], 0, 0
    while tries < max_tries:
        if targeted:
            needed = {l: r for l, r in lvl_quota.items() if r > 0}
            if not needed:
                break
        elif len(out) >= quota:
            break
        tries += 1
        stale += 1
        if stale > RL_STALE_MAX:
            break
        s_room = int(rng.choice(rooms))
        sc = nin(s_room)
        if not sc:
            continue
        s_node = int(rng.choice(sc))
        sp = np.array(nodes[s_node], np.float32)
        rdist, rprev = C.bfs_rooms(radj, s_room)

        def room_path(g_room):
            path = [g_room]
            while path[-1] != s_room:
                p = rprev.get(path[-1])
                if p is None:
                    return None
                path.append(p)
            return path[::-1]            # [s_room, mid1, ..., g_room]

        # 选目标桶(按缺口加权) -> 目标 geodesic 带 + 期望房间跳数 want
        if targeted and spm:
            nl = list(needed); wts = np.array([needed[l] for l in nl], float)
            tgt = nl[int(rng.choice(len(nl), p=wts / wts.sum()))]
            tlo, thi = LVL_STEP[tgt]
            gc = 0.5 * (tlo + thi) / spm
            band_lo = max(RL_MIN_GEO, tlo / spm * 0.80)
            band_hi = thi / spm * band_hi_mult
            if category == "r2r":
                want = max(1, int(round(gc / 6.0)))         # ~6m/房跳: 远桶自然多跳到更远的 room
            else:
                want = 2 if gc < 11 else (3 if gc < 20 else 4)
        else:
            band_lo, band_hi = RL_MIN_GEO, 1e9
            want = 1 if category == "r2r" else int(rng.choice([2, 2, 3, 3, 4]))

        # r2r = 任意两房间(不限跳数): 起点房 -> 任意可达目标房, 走最短路(单 goal, 仍是 room-to-room)
        dmin, dmax = (1, 99) if category == "r2r" else (2, 99)
        groomes = sorted([r for r, d in rdist.items()
                          if r in rooms and r != s_room and regs[r]["objs"] and dmin <= d <= dmax],
                         key=lambda r: abs(rdist[r] - want))[:8]
        got = False
        for g_room in groomes:
            if category == "r2r":
                chain_rooms = room_path(g_room)             # 最短(穿过中间房, 但任务=单起点->单goal)
            else:
                chain_rooms = C.random_room_chain(radj, s_room, g_room, rng,
                                                  extra=1, detour_prob=0.20, max_total=4)  # rxr 房间可重复但克制(链短,快)
            if chain_rooms is None:
                continue
            mids = []
            for r in chain_rooms[1:-1]:
                wp = room_waypoint(r)
                if wp is not None:
                    mids.append((r, wp))
            objs = [(cn, oc) for cn, oc in regs[g_room]["objs"] if cn not in C.WALL_OBJECTS]
            if not objs:
                continue
            for oi in list(rng.permutation(len(objs)))[:4]:
                cn, oc = objs[oi]
                gp = C.reachable_goal(pf, oc)         # snap+离墙校验
                if gp is None:
                    continue
                gp = np.asarray(gp, np.float32)
                if not (band_lo <= C.geo(pf, sp, gp) <= band_hi):   # 廉价预筛: 不在目标 geo 带就别走(贵)
                    continue
                waypoints = [wp for _, wp in mids] + [gp]
                rec = try_pair(s_node, g_room, cn, gp, waypoints, mids)
                if rec is None:
                    continue
                if targeted:
                    lvl = level_name(rec["n_steps"])
                    if lvl is None or lvl_quota.get(lvl, 0) <= 0:
                        continue                      # 落桶已满 -> 丢弃(轨迹已入 dedup, 不会重复产)
                    lvl_quota[lvl] -= 1
                out.append(rec); stale = 0; got = True
                break
            if got:
                break
    return out



# =========================================================================
# Matterport trajectories.
# =========================================================================
def cmd_rl_waypoint():
    ap = argparse.ArgumentParser(prog="rl-waypoint")
    ap.add_argument("--total", type=int, default=5000)
    ap.add_argument("--category", choices=["r2r", "rxr"], default="r2r",
                    help="r2r=one start room to one goal room; rxr=waypoints along a room chain")
    ap.add_argument("--scenes", default="", help="逗号分隔; 留空=全部 train 场景")
    ap.add_argument("--forbidden", default="", help="预算的禁止对文件; 留空则现算 R2R+RxR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-geo", type=float, default=3.0, help="geodesic 下限(m); 调高右移步数分布补中等桶")
    ap.add_argument("--stale", type=int, default=800, help="连续无新增放弃阈值; rxr 调低(150)加速")
    ap.add_argument("--target-dist", default="",
                    help="每场景目标分布(配额): 'lo:count' 逗号分隔(lo=桶下界 0,50,100..). 空=原始无配额")
    ap.add_argument("--step-ratios", default="", help="Optional weights for 0-49,50-99,...,350-400 steps")
    ap.add_argument("--seed-existing", default="",
                    help="已生成数据集目录; 把其轨迹指纹种进 Dedup 以跨 run 去重(接着生成不重复)")
    ap.add_argument("--seed-real", action="store_true",
                    help="把真实 R2R/RxR 轨迹(ref_path 加密后)种进 Dedup, 新轨迹与真实重合超阈值即拒")
    ap.add_argument("--path-overlap", type=float, default=0.75,
                    help="轨迹重合判重阈值(格点重叠率); 调高=更宽松产量多, 调低=更严轨迹更分散")
    ap.add_argument("--fast-follower", action="store_true",
                    help="快通道: 用 pathfinder 折线 + 几何控制驱动 sim.step 替代 greedy follower(~13x, sim 自洽)")
    ap.add_argument("--band-hi-mult", type=float, default=1.30,
                    help="配额模式 geo 预筛上界系数; 调低(如1.15)减少走完才因步数超桶被丢的浪费")
    ap.add_argument("--out", default=os.path.join(C.DATA, "datasets", "mp3d_rl"))
    a = ap.parse_args()
    global RL_MIN_GEO, RL_STALE_MAX; RL_MIN_GEO = a.min_geo; RL_STALE_MAX = a.stale
    walk_fn = C.walk_pathfollow if a.fast_follower else C.walk_follower
    train_scenes = sorted({e["scene_id"].split("/")[1] for e in
                           json.load(gzip.open(os.path.join(C.DATA, "datasets/r2r/train/train.json.gz"), "rt"))["episodes"]})
    scenes = a.scenes.split(",") if a.scenes else train_scenes
    if a.total <= 0 or not scenes:
        ap.error("Positive total and at least one scene required")
    if a.step_ratios and a.target_dist:
        ap.error("Use either --step-ratios or --target-dist")
    per = max(1, math.ceil(a.total / len(scenes)))
    target = parse_target_dist(a.target_dist)          # {lvl_name: 本进程每场景配额} 或 None
    if a.step_ratios:
        caps = ratio_caps(a.step_ratios, per)
        target = {level_name(i * 50): n for i, n in enumerate(caps) if n}
    spm = SPM.get(a.category) if target else None
    existing = load_existing_trajs(a.seed_existing, set(scenes)) if a.seed_existing else {}
    real_eps = load_real_refpaths(set(scenes)) if a.seed_real else {}
    if a.seed_existing:
        print(f"[去重①合成] 种入现有轨迹: {sum(len(v) for v in existing.values())} 条 / {len(existing)} 场景", flush=True)
    if a.seed_real:
        print(f"[去重②真实] 种入 R2R/RxR: {sum(len(v) for v in real_eps.values())} 条 / {len(real_eps)} 场景", flush=True)
    if target:
        print(f"[{a.category}/waypoint] 配额模式 每场景目标 {target} (总~{sum(target.values())*len(scenes)})", flush=True)
    else:
        print(f"[{a.category}/waypoint] 目标 {a.total} 条, {len(scenes)} 场景, 每场景 ~{per}", flush=True)

    eps_lvl, gt_lvl = defaultdict(list), defaultdict(dict)
    stats = {"geodesic": [], "n_steps": [], "scene": [], "rooms_per_traj": []}
    tid, t0 = 0, time.time()
    for si, scene in enumerate(scenes):
        if not target and len(stats["scene"]) >= a.total:
            break
        rng = np.random.default_rng(a.seed + si)
        sim = habitat_sim.Simulator(C.make_cfg(C.scene_path(scene)))
        try:
            sim.pathfinder.seed(a.seed + si)
            nodes, edges = C.load_or_build_graph(sim, scene, a.seed + si)
            regs = C.load_regions(sim, with_objects=True)
            node_room = C.assign_room(nodes, regs)
            # 不做端点去重(端点不管): 只靠④轨迹重合判重 -> 允许同起终点多路径, 但轨迹不能重
            dedup = C.Dedup([], endpoint_dedup=False, path_overlap=a.path_overlap)
            for s, g, locs in existing.get(scene, []):    # 去重①: 之前生成的合成轨迹(已密集)
                if locs:
                    dedup.add(s, g, traj=np.asarray(locs, np.float32))
            for s, g, rp in real_eps.get(scene, []):      # 去重②: 真实 R2R/RxR(稀疏 ref_path -> navmesh 加密)
                dp = C.densify(sim.pathfinder, rp, step=0.25)
                if len(dp) >= 2:
                    dedup.add(s, g, traj=dp)
            quota = min(per, a.total - len(stats["scene"]))
            lvl_quota = dict(target) if target else None  # 每场景独立配额(分片进程间不共享, 各填各的)
            res = _rl_scene_waypoint(scene, sim, regs, nodes, edges, node_room, dedup, quota, rng,
                                     a.category, lvl_quota=lvl_quota, spm=spm,
                                     walk_fn=walk_fn, band_hi_mult=a.band_hi_mult)
        finally:
            sim.close()
        for r in res:
            lvl = level_name(r["n_steps"])
            if lvl is None:
                continue
            eps_lvl[lvl].append({
                "episode_id": tid, "trajectory_id": tid, "scene_id": f"mp3d/{scene}/{scene}.glb",
                "start_position": r["start"], "start_rotation": C.yaw_quat(r["yaw"]),
                "info": {"geodesic_distance": r["geodesic"], "num_steps": r["n_steps"],
                         "rooms": r["rooms"], "goal_object": r["goal_obj"]},
                "goals": [{"position": r["goal"], "radius": GOAL_RADIUS_RL}],
                "instruction": {"instruction_text": r["instruction"]},
                "reference_path": r["reference_path"]})
            gt_lvl[lvl][str(tid)] = {"locations": r["locations"], "actions": r["actions"],
                                     "forward_steps": len([x for x in r["actions"] if x == 1])}
            stats["geodesic"].append(r["geodesic"]); stats["n_steps"].append(r["n_steps"])
            stats["scene"].append(scene); stats["rooms_per_traj"].append(len(r["rooms"]))
            tid += 1
        print(f"[{si+1}/{len(scenes)}] {scene}: +{len(res)} 累计 {len(stats['scene'])} "
              f"{time.time()-t0:.0f}s", flush=True)

    for lvl in sorted(eps_lvl):
        voc = C.fill_vocab(eps_lvl[lvl])
        d = os.path.join(a.out, lvl); os.makedirs(d, exist_ok=True)
        json.dump({"episodes": eps_lvl[lvl], "instruction_vocab": {"word_list": voc}},
                  gzip.open(os.path.join(d, "rl.json.gz"), "wt"))
        json.dump(gt_lvl[lvl], gzip.open(os.path.join(d, "rl_gt.json.gz"), "wt"))
        print(f"  {lvl}: {len(eps_lvl[lvl])} 条", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(stats, open(os.path.join(a.out, "_stats.json"), "w"))
    rng_s = f"[{min(stats['n_steps'])},{max(stats['n_steps'])}]" if stats["n_steps"] else "[空]"
    print(f"完成 {len(stats['scene'])} 条, 场景 {len(set(stats['scene']))}, "
          f"步数{rng_s} {time.time()-t0:.0f}s", flush=True)
# =========================================================================
# merge: 合并分片 -> 最终分级数据 (全局重编号)
# =========================================================================
def cmd_merge():
    ap = argparse.ArgumentParser(prog="merge")
    ap.add_argument("--shards", default=os.path.join(C.DATA, "datasets/r2r_rl/_shards"))
    ap.add_argument("--extra", default="",
                    help="额外已合并数据集目录(逗号分隔, 各含 lvl_*/); 把现有数据一起折进来做加法")
    ap.add_argument("--out", default=os.path.join(C.DATA, "datasets/r2r_rl"))
    a = ap.parse_args()
    eps_lvl, gt_lvl = defaultdict(list), defaultdict(dict)
    stats = {"geodesic": [], "n_steps": [], "scene": [], "rooms_per_traj": []}
    tid = 0
    sources = sorted(glob.glob(os.path.join(a.shards, "*")))      # 每场景一分片
    sources += [d for d in a.extra.split(",") if d.strip()]       # + 现有数据集
    for sh in sources:
        for ef in sorted(glob.glob(os.path.join(sh, "lvl_*", "rl.json.gz"))):
            lvl = os.path.basename(os.path.dirname(ef))
            d = json.load(gzip.open(ef, "rt"))
            gt = json.load(gzip.open(ef.replace("rl.json.gz", "rl_gt.json.gz"), "rt"))
            for ep in d["episodes"]:
                old = str(ep["trajectory_id"])
                ep["episode_id"] = ep["trajectory_id"] = tid
                eps_lvl[lvl].append(ep); gt_lvl[lvl][str(tid)] = gt[old]; tid += 1
                info = ep.get("info", {})                          # stats 从 episode 现算(旧 _stats 不可信)
                stats["geodesic"].append(info.get("geodesic_distance"))
                stats["n_steps"].append(info.get("num_steps"))
                stats["scene"].append(ep["scene_id"].split("/")[1])
                stats["rooms_per_traj"].append(len(info.get("rooms", [])))
    vocab = ["<pad>", "<unk>"]; vidx = {w: i for i, w in enumerate(vocab)}
    import re
    for lvl in eps_lvl:
        for ep in eps_lvl[lvl]:
            toks = []
            for w in re.findall(r"[a-z]+", ep["instruction"]["instruction_text"].lower()):
                if w not in vidx:
                    vidx[w] = len(vocab); vocab.append(w)
                toks.append(vidx[w])
            ep["instruction"]["instruction_tokens"] = toks
    for lvl in sorted(eps_lvl):
        dd = os.path.join(a.out, lvl); os.makedirs(dd, exist_ok=True)
        json.dump({"episodes": eps_lvl[lvl], "instruction_vocab": {"word_list": vocab}},
                  gzip.open(os.path.join(dd, "rl.json.gz"), "wt"))
        json.dump(gt_lvl[lvl], gzip.open(os.path.join(dd, "rl_gt.json.gz"), "wt"))
        print(f"{lvl}: {len(eps_lvl[lvl])} 条")
    json.dump(stats, open(os.path.join(a.out, "_stats.json"), "w"))
    print(f"合计 {tid} 条, 场景 {len(set(stats['scene']))}")

_DATA = os.environ.get("VLN_DATA", "/root/code/data")
HM3D_ROOT = os.path.join(_DATA, "scene_datasets", "hm3d")
SV_PATHS = os.path.join(_DATA, "datasets", "scalevln", "ScaleVLN", "scalevln_subset_150k.json.gz")
PY = sys.executable


def list_scenes(dataset, split):
    out = []
    if dataset == "hm3d":
        for d in sorted(glob.glob(f"{HM3D_ROOT}/{split}/*")):
            g = glob.glob(f"{d}/*.basis.glb")
            if not g:
                continue
            nav = glob.glob(f"{d}/*.basis.navmesh")
            out.append((os.path.basename(d), g[0], nav[0] if nav else None, bool(glob.glob(f"{d}/*.semantic.glb"))))
    else:
        for s in sorted(os.listdir(C.SCENES_DIR)):
            glb = C.scene_path(s)
            if os.path.exists(glb):
                out.append((s, glb, None, True))
    return out


def open_sim(dataset, glb, navmesh, gpu):
    cfg = C.make_cfg(glb, with_rgb=True)
    try:
        cfg.sim_cfg.gpu_device_id = gpu          # 分摊多卡
    except Exception:
        pass
    sim = habitat_sim.Simulator(cfg)
    if navmesh and not sim.pathfinder.is_loaded:
        sim.pathfinder.load_nav_mesh(navmesh)
    return sim


def write_mp4_h264(path, frames, fps=8):
    if not frames:
        return
    h, w = frames[0].shape[:2]; tmp = path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", path], check=True)
    os.remove(tmp)


def sample_episode(sim, params, rng, eid, scene_id, glb, capture=False):
    pf, agent = sim.pathfinder, sim.get_agent(0)
    follower = sim.make_greedy_follower(agent_id=0, goal_radius=params["goal_radius"], stop_key="stop",
                                        forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    for _ in range(params["tries"]):
        start = pf.get_random_navigable_point()
        if not pf.is_navigable(start):
            continue
        goal = pf.get_random_navigable_point_near(start, rng.uniform(params["min_m"], params["max_m"]), max_tries=50)
        if goal is None or np.any(np.isnan(goal)) or not pf.is_navigable(goal):
            continue
        geod, pts = C.geo_path(pf, start, goal)
        if pts is None or geod < params["min_m"] or geod > params["max_m"]:
            continue
        yaw = float(rng.uniform(0, 2 * math.pi))
        st = agent.get_state(); st.position = np.array(start, np.float32)
        st.rotation = hsc.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0])); agent.set_state(st)
        try:
            actions = follower.find_path(goal)
        except habitat_sim.errors.GreedyFollowerError:
            continue
        moves = [x for x in actions if x != "stop"]
        if len(moves) < 4:
            continue
        agent.set_state(st); locs = [list(map(float, agent.get_state().position))]; frames = []
        if capture:
            frames.append(cv2.cvtColor(np.asarray(sim.get_sensor_observations()["rgb"])[:, :, :3], cv2.COLOR_RGB2BGR))
        for x in moves:
            sim.step(x); locs.append(list(map(float, agent.get_state().position)))
            if capture:
                frames.append(cv2.cvtColor(np.asarray(sim.get_sensor_observations()["rgb"])[:, :, :3], cv2.COLOR_RGB2BGR))
        pts = [list(map(float, p)) for p in pts]
        ep = {"episode_id": eid, "trajectory_id": eid, "scene_id": f"{scene_id}/{os.path.basename(glb)}",
              "start_position": list(map(float, start)), "start_rotation": C.yaw_quat(yaw),
              "info": {"geodesic_distance": geod, "num_steps": len(moves), "num_turns": C.count_turns(pts)},
              "goals": [{"position": list(map(float, goal)), "radius": params["goal_radius"]}],
              "instruction": {"instruction_text": ""}, "reference_path": pts}
        gt = {"locations": locs, "actions": [C.ACT[x] for x in moves] + [C.ACT["stop"]],
              "forward_steps": sum(1 for x in moves if x == "move_forward")}
        return ep, gt, frames
    return None, None, None


def topdown(pts, start, goal, W=480, H=380, pad=36):
    P = np.array([[p[0], p[2]] for p in pts], np.float32)
    allp = np.vstack([P, [[start[0], start[2]], [goal[0], goal[2]]]])
    lo, hi = allp.min(0), allp.max(0); span = hi - lo; span[span < 1e-3] = 1e-3
    to = lambda xz: (int(pad + (xz[0]-lo[0])/span[0]*(W-2*pad)), int(H-pad-(xz[1]-lo[1])/span[1]*(H-2*pad)))
    img = np.full((H, W, 3), 255, np.uint8); px = [to(p) for p in P]
    for a, b in zip(px[:-1], px[1:]):
        cv2.line(img, a, b, (200, 120, 0), 2)
    cv2.circle(img, to([start[0], start[2]]), 7, (0, 170, 0), -1)
    cv2.drawMarker(img, to([goal[0], goal[2]]), (0, 0, 220), cv2.MARKER_STAR, 16, 2)
    return img


# ---------------- 去重 / nDTW / 重合 ----------------
def _dense_path(rp, step=0.25):
    out = []
    for a, b in zip(rp[:-1], rp[1:]):
        a = np.array([a[0], a[2]]); b = np.array([b[0], b[2]]); d = np.linalg.norm(b - a); n = max(1, int(d / step))
        for t in range(n):
            out.append(a + (b - a) * t / n)
    out.append(np.array([rp[-1][0], rp[-1][2]])); return np.array(out)


def _xy(rp): return np.array([[p[0], p[2]] for p in rp])


def _resample_xy(rp, step=1.5):
    P = np.array([[p[0], p[2]] for p in rp], float)
    if len(P) < 2:
        return P
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); L = seg.sum()
    if L < 1e-6:
        return P[:1]
    cum = np.concatenate([[0], np.cumsum(seg)]); out = []
    for t in np.arange(0, L, step):
        i = min(max(np.searchsorted(cum, t) - 1, 0), len(P) - 2)
        r = (t - cum[i]) / max(seg[i], 1e-9); out.append(P[i] + (P[i + 1] - P[i]) * r)
    out.append(P[-1]); return np.array(out)


def _dtwc(P, Q):
    n, m = len(P), len(Q); D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf); acc[0, 0] = 0
    for i in range(1, n + 1):
        ai, ai1, Di = acc[i], acc[i - 1], D[i - 1]
        for j in range(1, m + 1):
            ai[j] = Di[j - 1] + min(ai1[j], ai[j - 1], ai1[j - 1])
    return acc[n, m]


def _fwd_ndtw(P, Q):   # 方向敏感(正向), 对称取较大
    return max(math.exp(-_dtwc(P, Q) / (len(Q) * 3.0)), math.exp(-_dtwc(Q, P) / (len(P) * 3.0)))


def _ndtw_dup(pa, kept, thr):
    return any(_fwd_ndtw(pa, k) >= thr for k in kept)


def _ref_dup(pa, cs, ce, refs, thr, pre=4.0):
    for path, rs, re in refs:
        if abs(cs[0]-rs[0]) < pre and abs(cs[1]-rs[1]) < pre and abs(ce[0]-re[0]) < pre and abs(ce[1]-re[1]) < pre:
            if _fwd_ndtw(pa, path) >= thr:
                return True
    return False


def _max_cover(pa, sv_list, thr=0.5, reject_at=1.1):
    best = 0.0
    for s in sv_list:
        c = float(np.mean(np.min(np.linalg.norm(pa[:, None, :] - s[None, :, :], axis=2), axis=1) < thr))
        if c > best:
            best = c
            if best > reject_at:
                return best
    return best


# ---------------- 采样 worker (单分片) ----------------
def _write_checkpoint(path, state):
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _resume_shard(jsonl, cfg, scenes, shard_id, shard_count):
    config = {k: v for k, v in vars(cfg).items() if k not in ("out", "gpu")}
    signature = hashlib.sha256(json.dumps(
        [config, scenes, shard_id, shard_count], sort_keys=True).encode()).hexdigest()
    path = jsonl + ".checkpoint.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if state["signature"] != signature:
            raise ValueError("Sampling settings changed; use a new output directory")
        size = os.path.getsize(jsonl) if os.path.exists(jsonl) else 0
        if size < state["offset"]:
            raise ValueError(f"Shard is shorter than its checkpoint: {jsonl}")
        if os.path.exists(jsonl):
            with open(jsonl, "rb+") as f:
                f.truncate(state["offset"])
    else:
        if os.path.exists(jsonl) and os.path.getsize(jsonl):
            raise ValueError("Existing shard has no checkpoint; use a new output directory")
        state = dict(signature=signature, offset=0, count=0, done=[])
        _write_checkpoint(path, state)
    return path, state


def _worker_sample(cfg):
    gpu = int(os.environ.get("GPU_ID", 0)); sid = int(os.environ["SHARD_ID"]); sn = int(os.environ["SHARD_N"])
    scenes = sorted([s for s in list_scenes(cfg.dataset, cfg.split) if s[2] or cfg.dataset == "mp3d"])
    if cfg.scenes > 0:
        scenes = scenes[:cfg.scenes]
    scenes = [s for i, s in enumerate(scenes) if i % sn == sid]
    os.makedirs(cfg.out, exist_ok=True)
    jsonl = f"{cfg.out}/_traj_shard{sid}of{sn}.jsonl"
    checkpoint_path, checkpoint = _resume_shard(jsonl, cfg, scenes, sid, sn)
    done = set(checkpoint["done"])
    scenes = [s for s in scenes if s[0] not in done]
    svp = {}
    if (cfg.sv_overlap_max > 0 or cfg.dedup_sv) and os.path.exists(SV_PATHS):
        for e in json.load(gzip.open(SV_PATHS, "rt"))["episodes"]:
            if e.get("reference_path"):
                svp.setdefault(e["scene_id"].split("/")[-2], []).append(e["reference_path"])
    if cfg.bin_caps and cfg.bin_ratios:
        raise ValueError("Use either --bin-caps or --bin-ratios")
    bincap = [int(x) for x in cfg.bin_caps.split(",")] if cfg.bin_caps else ratio_caps(cfg.bin_ratios, cfg.per_scene)
    if bincap is not None and (not bincap or any(x < 0 for x in bincap) or sum(bincap) != cfg.per_scene):
        raise ValueError("Bin caps must be nonnegative and sum to per-scene count")
    refset = {}
    if cfg.dedup_sv:
        for scn, ps in svp.items():
            for rp in ps:
                pa = _resample_xy(rp); refset.setdefault(scn, []).append((pa, pa[0], pa[-1]))
    if cfg.dedup_keep and os.path.exists(cfg.dedup_keep):
        keep = set(json.load(open(cfg.dedup_keep)))
        for e in json.load(gzip.open(f"{cfg.keep_src}/rl.json.gz", "rt"))["episodes"]:
            if e["episode_id"] in keep:
                pa = _resample_xy(e["reference_path"]); refset.setdefault(e["scene_id"].split("/")[-2], []).append((pa, pa[0], pa[-1]))
    use_ref = bool(cfg.dedup_sv or cfg.dedup_keep)
    max_att = cfg.max_attempts if cfg.max_attempts > 0 else cfg.per_scene * 3
    params = dict(goal_radius=cfg.goal_radius, min_m=cfg.min_m, max_m=cfg.max_m, tries=cfg.tries)
    existing = checkpoint["count"]
    eid = sid * 1000000 + existing
    fout = open(jsonl, "ab")
    print(f"[shard{sid}/{sn}] {len(scenes)}场景 gpu{gpu} bincap={bincap} ndtw={cfg.ndtw_dedup}/{cfg.ndtw_hi} ref={use_ref}", flush=True)
    for sdir, glb, nav, sem in scenes:
        sv_dense = [_dense_path(p) for p in svp.get(sdir, [])] if cfg.sv_overlap_max > 0 else []
        rng = np.random.default_rng(cfg.seed + int.from_bytes(hashlib.sha256(sdir.encode("utf-8")).digest()[:4], "big") % 100000)
        try:
            sim = open_sim(cfg.dataset, glb, nav, gpu)
        except Exception as ex:
            raise RuntimeError(f"Cannot open scene {sdir}") from ex
        sim.pathfinder.seed(cfg.seed + int.from_bytes(hashlib.sha256(sdir.encode("utf-8")).digest()[:4], "big") % 100000)
        got = att = 0; bincnt = [0] * (len(bincap) if bincap else 0); kept = []
        while got < cfg.per_scene and att < max_att:
            att += 1
            ep, gt, frames = sample_episode(sim, params, rng, eid, f"{cfg.dataset}/{sdir}", glb, capture=cfg.viz)
            if ep is None:
                continue
            nstep = sum(1 for a in gt["actions"] if a != 0)
            if bincap is not None:
                b = nstep // 50
                if b >= len(bincap) or bincnt[b] >= bincap[b]:
                    continue
            if sv_dense and _max_cover(_xy(ep["reference_path"]), sv_dense, reject_at=cfg.sv_overlap_max) > cfg.sv_overlap_max:
                continue
            if cfg.ndtw_dedup > 0:
                pa = _resample_xy(ep["reference_path"]); thr = cfg.ndtw_dedup if nstep < 100 else cfg.ndtw_hi
                if use_ref and _ref_dup(pa, pa[0], pa[-1], refset.get(sdir, []), thr):
                    continue
                if _ndtw_dup(pa, kept, thr):
                    continue
                kept.append(pa)
            fout.write((json.dumps({"ep": ep, "gt": gt}, ensure_ascii=False) + "\n").encode("utf-8")); fout.flush()
            if bincap is not None:
                bincnt[nstep // 50] += 1
            if cfg.viz:
                cv2.imwrite(f"{cfg.out}/viz_{sdir}_ep{eid}_top.png", topdown(ep["reference_path"], ep["start_position"], ep["goals"][0]["position"]))
                write_mp4_h264(f"{cfg.out}/viz_{sdir}_ep{eid}_fpv.mp4", frames)
            eid += 1; got += 1
        sim.close()
        fout.flush()
        os.fsync(fout.fileno())
        done.add(sdir)
        checkpoint.update(offset=fout.tell(), count=eid - sid * 1000000, done=sorted(done))
        _write_checkpoint(checkpoint_path, checkpoint)
        print(f"  [{sdir}] {got}/{att}" + (f" 桶{bincnt}" if bincap else ""), flush=True)
    fout.close(); print(f"[shard{sid}] ___SHARD_DONE___", flush=True)


# ---------------- 采样 launcher (多卡) ----------------
def _ngpu():
    try:
        return max(1, len(subprocess.check_output(["nvidia-smi", "-L"]).decode().strip().splitlines()))
    except Exception:
        return 1


def launch_sample(cfg):
    for path in glob.glob(os.path.join(cfg.out, "_traj_shard*of*.jsonl")):
        if not path.endswith(f"of{cfg.workers}.jsonl"):
            raise ValueError("Worker count changed; use a new output directory")
    ng = _ngpu(); procs = []
    passthrough = ["--dataset", cfg.dataset, "--split", cfg.split, "--scenes", str(cfg.scenes),
                   "--per-scene", str(cfg.per_scene), "--min-m", str(cfg.min_m), "--max-m", str(cfg.max_m),
                   "--goal-radius", str(cfg.goal_radius), "--tries", str(cfg.tries), "--max-attempts", str(cfg.max_attempts), "--out", cfg.out,
                   "--sv-overlap-max", str(cfg.sv_overlap_max), "--bin-ratios", cfg.bin_ratios,
                   "--bin-caps", cfg.bin_caps, "--ndtw-dedup", str(cfg.ndtw_dedup), "--ndtw-hi", str(cfg.ndtw_hi),
                   "--keep-src", cfg.keep_src, "--dedup-keep", cfg.dedup_keep, "--seed", str(cfg.seed)]
    if cfg.dedup_sv:
        passthrough.append("--dedup-sv")
    if cfg.viz:
        passthrough.append("--viz")
    os.makedirs(cfg.out, exist_ok=True)
    for i in range(cfg.workers):
        env = dict(os.environ, SHARD_ID=str(i), SHARD_N=str(cfg.workers), GPU_ID=str(i % ng))
        logf = open(f"{cfg.out}/_w{i}of{cfg.workers}.log", "w")
        procs.append(subprocess.Popen([PY, "-u", __file__, "sample", *passthrough], env=env, stdout=logf, stderr=subprocess.STDOUT))
    print(f"[sample] {cfg.workers} worker × {ng} GPU -> {cfg.out}", flush=True)
    t0 = time.time()
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"sampling workers failed: {codes}")
    # 合并分片 -> gen.json.gz
    eps, gts, seen = [], {}, set()
    for f in sorted(glob.glob(f"{cfg.out}/_traj_shard*of{cfg.workers}.jsonl")):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l); e = r["ep"]
            if e["episode_id"] in seen:
                continue
            seen.add(e["episode_id"]); eps.append(e); gts[str(e["episode_id"])] = r["gt"]
    json.dump({"episodes": eps}, gzip.open(f"{cfg.out}/gen.json.gz", "wt"))
    json.dump(gts, gzip.open(f"{cfg.out}/gen_gt.json.gz", "wt"))
    print(f"[sample] 合并 -> {cfg.out}/gen.json.gz  {len(eps)} 条, 用时 {time.time()-t0:.0f}s\n___SAMPLE_DONE___", flush=True)


# ---------------- dedup 子命令 ----------------
def cmd_dedup(a):
    r = json.load(gzip.open(a.rl, "rt"))["episodes"]
    gtf = a.rl.replace("rl.json.gz", "rl_gt.json.gz").replace("gen.json.gz", "gen_gt.json.gz")
    gt = json.load(gzip.open(gtf, "rt"))
    byscene = {}
    for e in r:
        byscene.setdefault(e["scene_id"].split("/")[-2], []).append(e)
    keep_ids, removed = [], 0
    for sc, E in byscene.items():
        E = sorted(E, key=lambda e: -sum(1 for x in gt[str(e["episode_id"])]["actions"] if x != 0))
        P = [_resample_xy(e["reference_path"]) for e in E]; kept = []
        for i, e in enumerate(E):
            if all(_fwd_ndtw(P[i], P[j]) < a.thr for j in kept):
                kept.append(i); keep_ids.append(e["episode_id"])
            else:
                removed += 1
    json.dump(keep_ids, open(a.out, "w"))
    print(f"dedup(thr={a.thr}): {len(r)} -> 保留 {len(keep_ids)} (删 {removed}) -> {a.out}")


# ---------------- assemble 子命令 ----------------
def _step(gt, eid): return sum(1 for x in gt[str(eid)]["actions"] if x != 0)


def cmd_assemble(a):
    def norm(sid):
        p = sid.split("/"); return sid if len(p) > 1 and p[1] == "train" else f"{p[0]}/train/{p[1]}/{p[2]}"
    items = []   # (ep, gt, instr, is_new)
    if a.labeled:
        lr = json.load(gzip.open(f"{a.labeled}/rl.json.gz", "rt"))["episodes"]
        lg = json.load(gzip.open(f"{a.labeled}/rl_gt.json.gz", "rt"))
        keep = set(json.load(open(a.keep))) if a.keep else set(e["episode_id"] for e in lr)
        for e in lr:
            if e["episode_id"] in keep:
                items.append((e, lg[str(e["episode_id"])], e["instruction"]["instruction_text"], False))
    for nd in a.new:
        r = json.load(gzip.open(f"{nd}/gen.json.gz", "rt"))["episodes"]; g = json.load(gzip.open(f"{nd}/gen_gt.json.gz", "rt"))
        for e in r:
            items.append((e, g[str(e["episode_id"])], "", True))
    if a.trim_ratio <= 0:
        kept = items
    else:
        # 只裁指定桶到目标比例, 优先丢新的
        tb = a.trim_bin // 50
        inbin = [it for it in items if min(_step_it(it) // 50, 4) == tb]
        other = [it for it in items if min(_step_it(it) // 50, 4) != tb]
        T = round(len(other) / (1 - a.trim_ratio)); keepn = round(a.trim_ratio * T); drop = len(inbin) - keepn
        inbin.sort(key=lambda it: (0 if it[3] else 1))
        kept = (inbin[drop:] if drop > 0 else inbin) + other
    if not kept:
        raise SystemExit("No episodes to pack")
    os.makedirs(a.out, exist_ok=True); tol = a.out + "/_tolabel"; os.makedirs(tol, exist_ok=True)
    feps, fgt, teps, tgt = [], {}, [], {}
    for nid, (e, gt, ins, isnew) in enumerate(kept):
        ep = dict(e); ep["episode_id"] = nid; ep["trajectory_id"] = nid; ep["scene_id"] = norm(e["scene_id"])
        ep["instruction"] = {"instruction_text": ins}
        feps.append(ep); fgt[str(nid)] = gt
        if isnew:
            te = dict(ep); te["scene_id"] = ep["scene_id"].replace("hm3d/train/", "hm3d/")
            teps.append(te); tgt[str(nid)] = gt
    json.dump({"episodes": feps}, gzip.open(f"{a.out}/rl.json.gz", "wt")); json.dump(fgt, gzip.open(f"{a.out}/rl_gt.json.gz", "wt"))
    json.dump({"episodes": teps}, gzip.open(f"{tol}/gen.json.gz", "wt")); json.dump(tgt, gzip.open(f"{tol}/gen_gt.json.gz", "wt"))
    b = Counter(min(_step(fgt, e["episode_id"]) // 50 * 50, 200) for e in feps); n = len(feps)
    print(f"assemble: {n} 条 (待标 {len(teps)}) -> {a.out}; 待标集 {tol}")
    print("桶:", {f"{k}-{k+50}": f"{100*b[k]/n:.0f}%" for k in (0, 50, 100, 150)})


def _step_it(it): return sum(1 for x in it[1]["actions"] if x != 0)


def _hm3d_main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--workers", type=int, default=1, help=">1=多卡launcher; 内部worker由SHARD_N触发")
    for name, kw in [("--dataset", dict(default="hm3d", choices=["hm3d", "mp3d"])), ("--split", dict(default="train")),
                     ("--scenes", dict(type=int, default=0)), ("--per-scene", dict(type=int, default=30)),
                     ("--min-m", dict(type=float, default=5.0)), ("--max-m", dict(type=float, default=13.0)),
                     ("--goal-radius", dict(type=float, default=0.5)), ("--tries", dict(type=int, default=200)),
                     ("--max-attempts", dict(type=int, default=0)), ("--sv-overlap-max", dict(type=float, default=0.0)),
                     ("--bin-ratios", dict(default="")), ("--bin-caps", dict(default="")),
                     ("--ndtw-dedup", dict(type=float, default=0.0)), ("--ndtw-hi", dict(type=float, default=0.55)),
                     ("--dedup-keep", dict(default="")), ("--keep-src", dict(default="")),
                     ("--seed", dict(type=int, default=0)), ("--gpu", dict(type=int, default=0)),
                     ("--out", dict(default=f"{HM3D_ROOT}/_gen"))]:
        s.add_argument(name, **kw)
    s.add_argument("--dedup-sv", action="store_true"); s.add_argument("--viz", action="store_true")
    d = sub.add_parser("dedup"); d.add_argument("--rl", required=True); d.add_argument("--thr", type=float, default=0.52); d.add_argument("--out", required=True)
    b = sub.add_parser("assemble"); b.add_argument("--labeled", default=""); b.add_argument("--keep", default="")
    b.add_argument("--new", nargs="+", default=[]); b.add_argument("--trim-bin", type=int, default=0)
    b.add_argument("--trim-ratio", type=float, default=0.25); b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "sample":
        if a.per_scene <= 0 or a.workers <= 0:
            ap.error("per-scene and workers must be positive")
        if a.bin_caps and a.bin_ratios:
            ap.error("Use either --bin-caps or --bin-ratios")
        ratio_caps(a.bin_ratios, a.per_scene)
        (_worker_sample if os.environ.get("SHARD_N") else launch_sample)(a)
    elif a.cmd == "dedup":
        cmd_dedup(a)
    elif a.cmd == "assemble":
        cmd_assemble(a)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: sample.py {mp3d|merge|hm3d|dedup|pack} ...")
    cmd = sys.argv[1]
    if cmd == "mp3d":
        sys.argv = ["sample mp3d"] + sys.argv[2:]
        cmd_rl_waypoint()
        return
    if cmd == "merge":
        sys.argv = ["sample merge"] + sys.argv[2:]
        cmd_merge()
        return
    if cmd == "hm3d":
        sys.argv[1] = "sample"
    elif cmd == "pack":
        sys.argv[1] = "assemble"
    elif cmd not in ("sample", "dedup", "assemble"):
        raise SystemExit("用法: sample.py {mp3d|merge|hm3d|dedup|pack} ...")
    _hm3d_main()


if __name__ == "__main__":
    main()
