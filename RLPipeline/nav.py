#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulator, navigation graph, and the waypoint views used when writing instructions."""
import heapq
import math
import os
import re
from collections import Counter, defaultdict, deque

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

import habitat_sim
import habitat_sim.utils.common as hsc

# ----------------------------- 常量 -----------------------------
DATA = os.environ.get("VLN_DATA", "/root/code/data")
SCENES_DIR = os.path.join(DATA, "scene_datasets", "mp3d")
GRAPH_DIR = "./nav_graphs"
FORWARD_STEP = 0.25          # 米/步
TURN_ANGLE = 15              # 度/步
# HabitatSimActions 标准编码 (与 R2R gt 对齐)
ACT = {"stop": 0, "move_forward": 1, "turn_left": 2, "turn_right": 3}
ACTION_ID = ACT             # 别名
STRUCT = {"wall", "floor", "ceiling", "misc", "void", "objects", "column",
          "beam", "board_panel", "unlabeled", "railing", "door", "window"}


def scene_path(scene):
    return os.path.join(SCENES_DIR, scene, f"{scene}.glb")


# ----------------------------- Simulator 配置 -----------------------------
def make_cfg(scene_glb, with_rgb=False):
    """普通 agent: 前进0.25/转15 动作空间; with_rgb 时挂一个第一视角 RGB。"""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.enable_physics = False
    ag = habitat_sim.agent.AgentConfiguration()
    ag.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)),
    }
    if with_rgb:
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [480, 640]; rgb.position = [0.0, 1.25, 0.0]
        ag.sensor_specifications = [rgb]
    else:
        ag.sensor_specifications = []
    return habitat_sim.Configuration(sim_cfg, [ag])


def make_ortho_cfg(scene_glb, w_px, h_px, mpp):
    """正交俯视相机, ortho_scale = 米/像素。"""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.enable_physics = False
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "ortho"; cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    cam.resolution = [h_px, w_px]; cam.ortho_scale = mpp
    cam.position = [0.0, 0.0, 0.0]; cam.orientation = [-math.pi / 2, 0.0, 0.0]
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [cam]
    return habitat_sim.Configuration(sim_cfg, [ag])


# ----------------------------- 几何 / 寻路 -----------------------------
def euclid(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
    return sp.geodesic_distance if pf.find_path(sp) else float("inf")


def geo_path(pf, a, b):
    """返回 (geodesic, points[list np]); 不可达返回 (inf, None)。"""
    sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
    if pf.find_path(sp):
        return sp.geodesic_distance, [np.array(p, np.float32) for p in sp.points]
    return float("inf"), None


def dijkstra(adj, src, dst, blocked=()):
    """加权最短路; blocked 为屏蔽节点集 (端点除外)。adj[u]=[(v,w),...]。"""
    bl = set(blocked) - {src, dst}
    dist = {src: 0.0}; prev = {}; pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, 1e18):
            continue
        for v, w in adj[u]:
            if v in bl:
                continue
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return p[::-1]


def bfs_rooms(radj, src):
    dist = {src: 0}; prev = {src: None}; q = deque([src])
    while q:
        u = q.popleft()
        for v in radj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; prev[v] = u; q.append(v)
    return dist, prev


def random_room_chain(radj, s_room, g_room, rng, extra=3, detour_prob=0.5, max_total=8):
    """房间图上从 s_room 有偏随机游走到 g_room, 允许绕路/重复经过房间。
    替代 BFS 最短路: 一对(start,goal)能生成多条结构不同的房间链 -> 组合数与多样性大增,
    单场景容量翻几倍。extra=最短跳数外额外绕路预算; detour_prob=每步走非最短邻居的概率;
    max_total=链总房间数上限(防步数爆). detour_prob=0 退化为最短路。
    返回房间序列 [s_room,...,g_room](可含重复); 不连通/到不了返回 None。"""
    dg = {g_room: 0}; q = deque([g_room])          # 到 g_room 的最短跳数, 用于导航保证可达
    while q:
        u = q.popleft()
        for v in radj[u]:
            if v not in dg:
                dg[v] = dg[u] + 1; q.append(v)
    if s_room not in dg:
        return None
    budget = dg[s_room] + int(rng.integers(0, extra + 1))   # 允许走的总步数(含绕路)
    chain = [s_room]; cur = s_room
    while cur != g_room and len(chain) < max_total:
        nbrs = [v for v in radj[cur] if v in dg]
        if not nbrs:
            return None
        remain = max_total - len(chain)                    # 还剩几步能走
        feas = [v for v in nbrs if dg[v] < remain] or nbrs # 留得下回 g 的步数
        if len(chain) - 1 >= budget or rng.random() >= detour_prob:
            mn = min(dg[v] for v in feas)                  # 超预算/不绕 -> 朝 goal
            cur = int(rng.choice([v for v in feas if dg[v] == mn]))
        else:
            cur = int(rng.choice(feas))                    # 绕路 -> 随机可行邻居
        chain.append(cur)
    return chain if cur == g_room else None


def heading(a, b):
    d = np.asarray(b) - np.asarray(a)
    return math.atan2(d[0], d[2])


def norm_deg(rad):
    deg = math.degrees(rad)
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def turn_word(prev_h, cur_h):
    t = norm_deg(cur_h - prev_h)
    if abs(t) < 25:
        return None
    if abs(t) > 150:
        return "turn around"
    return "turn left" if t > 0 else "turn right"


def path_to_actions(nodes, seq, detour_edge=None, start_yaw=None):
    """节点序列 -> [(type, value, is_detour)] (转向角 + 前进米)。"""
    acts = []
    cur = heading(nodes[seq[0]], nodes[seq[1]]) if start_yaw is None else start_yaw
    for i in range(len(seq) - 1):
        a, b = nodes[seq[i]], nodes[seq[i + 1]]
        is_d = bool(detour_edge[i]) if detour_edge else False
        h = heading(a, b)
        turn = norm_deg(h - cur)
        if abs(turn) >= 1.0:
            acts.append(("turn_left" if turn > 0 else "turn_right", round(abs(turn), 1), is_d))
        dist = float(np.linalg.norm(b - a))
        if dist > 1e-3:
            acts.append(("move_forward", round(dist, 2), is_d))
        cur = h
    acts.append(("stop", 0.0, False))
    return acts


class Dedup:
    """去重: ①vs 数据集(start,goal 6D KDTree) ②vs 已生成(start/goal 格点)
    ③goal 空间分散(已用 goal 1.5m 内限选 max_per_goal 次) ④轨迹重合(RDP指纹格点重叠率)。
    一场景一实例 -> 零跨进程共享, 天然并行。"""
    def __init__(self, pairs, ts=1.5, tg=1.5, cell=1.0,
                 goal_spread=1.5, max_per_goal=8, start_diverse=2.0,
                 path_overlap=0.8, path_cell=1.0, endpoint_dedup=True, dataset_dedup=None):
        self.ts, self.tg, self.cell = ts, tg, cell
        self.endpoint_dedup = endpoint_dedup   # ②③: 关=允许自己同起终点多路径(只靠④轨迹重合判重)
        # ①: 是否避开传入的 pairs(真实 R2R/RxR 端点); 默认跟随 endpoint_dedup, 可单独打开
        self.dataset_dedup = endpoint_dedup if dataset_dedup is None else dataset_dedup
        # goal 可复用, 但: 每个 goal 区域最多 max_per_goal 条, 且复用时 start 要离已有 >start_diverse
        self.goal_spread, self.max_per_goal, self.start_diverse = goal_spread, max_per_goal, start_diverse
        self.path_overlap, self.path_cell = path_overlap, path_cell
        self.pts = (np.array([np.concatenate([np.asarray(s, np.float32), np.asarray(g, np.float32)])
                              for s, g in pairs], np.float32) if pairs else np.empty((0, 6), np.float32))
        self.tree = cKDTree(self.pts) if len(self.pts) else None
        self.seen = set()                 # (start_cell, goal_cell)
        self.by_goal = []                 # [(goal_pos, [start_pos,...])] 每个 goal 用过的 start
        self.path_fps = []

    def _key(self, start, goal):
        return (tuple((np.asarray(start) // self.cell).astype(int)),
                tuple((np.asarray(goal) // self.cell).astype(int)))

    def _fp(self, traj):
        return {tuple((np.asarray(p) // self.path_cell).astype(int)) for p in traj}

    def is_dup(self, start, goal, traj=None):
        start = np.asarray(start, np.float32); goal = np.asarray(goal, np.float32)
        # ① vs 数据集(真实 R2R/RxR forbidden 端点); 独立开关 -> waypoint 模式也能避开真实数据
        if self.dataset_dedup and self.tree is not None:
            p6 = np.concatenate([start, goal])
            for idx in self.tree.query_ball_point(p6, r=(self.ts ** 2 + self.tg ** 2) ** 0.5):
                q = self.pts[idx]
                if np.linalg.norm(q[:3] - start) < self.ts and np.linalg.norm(q[3:] - goal) < self.tg:
                    return True
        if self.endpoint_dedup:
            # ② vs 已生成端点
            if self._key(start, goal) in self.seen:
                return True
            # ③ goal 可复用, 但同一 goal 区域: 不超 max_per_goal 条, 且新 start 要离已有 start 够远(换方向)
            for g, starts in self.by_goal:
                if np.linalg.norm(g - goal) < self.goal_spread:
                    if len(starts) >= self.max_per_goal:
                        return True
                    if any(np.linalg.norm(np.asarray(s) - start) < self.start_diverse for s in starts):
                        return True            # start 太近 = 同方向逼近, 弃
        # ④ 轨迹与已有重合过高
        if traj is not None and self.path_fps:
            fp = self._fp(traj)
            for ofp in self.path_fps:
                if len(fp & ofp) / max(min(len(fp), len(ofp)), 1) > self.path_overlap:
                    return True
        return False

    def add(self, start, goal, traj=None):
        start = np.asarray(start, np.float32); goal = np.asarray(goal, np.float32)
        self.seen.add(self._key(start, goal))
        for g, starts in self.by_goal:
            if np.linalg.norm(g - goal) < self.goal_spread:
                starts.append(start); break
        else:
            self.by_goal.append((goal, [start]))
        if traj is not None:
            self.path_fps.append(self._fp(traj))


def densify(pf, pts, step=0.25):
    """相邻点之间用 navmesh 测地折线补点 (贴墙绕角); step>0 时再按固定弧长重采样 -> 平滑稠密。"""
    pts = [np.asarray(p, np.float32) for p in pts]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
        seg = [np.asarray(p, np.float32) for p in sp.points] if pf.find_path(sp) else [a, b]
        out += seg[:-1]
    out.append(pts[-1])
    poly = np.array(out, np.float32)
    if step <= 0 or len(poly) < 2:
        return poly
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return poly
    res = []
    for si in np.arange(0, total, step):
        k = min(max(np.searchsorted(cum, si) - 1, 0), len(seg) - 1)
        t = (si - cum[k]) / max(seg[k], 1e-9)
        res.append(poly[k] * (1 - t) + poly[k + 1] * t)
    res.append(poly[-1])
    return np.array(res, np.float32)


# 挂墙/天花板/嵌入类物体: 选作 goal 必不可达, 排除
WALL_OBJECTS = {"picture", "mirror", "tv_monitor", "window", "curtain", "blinds",
                "lighting", "clock", "painting", "shelving", "panel", "fireplace",
                "air_vent", "switch", "board_panel", "beam", "column"}


def reachable_goal(pf, obj_center, ring=(0.6, 1.0, 1.5), n_dir=8, clearance=0.2):
    """在物体中心周围一圈找可走点作 goal (站在物体旁看它)。
    物体中心本身常在几何内部不可走 -> 不直接 snap 中心, 而是外扩取环上点。
    返回可走点或 None。"""
    c = np.asarray(obj_center, np.float32)
    # 先试中心 snap (小物体/开阔处直接可用)
    s = pf.snap_point(c)
    if not np.any(np.isnan(s)) and pf.is_navigable(s) and np.linalg.norm(s - c) < 1.0:
        return np.asarray(s, np.float32)
    # 否则在水平环上找最近可走点
    best, bd = None, 1e9
    for rad in ring:
        for k in range(n_dir):
            a = 2 * math.pi * k / n_dir
            q = c + np.array([rad * math.cos(a), 0.0, rad * math.sin(a)], np.float32)
            p = pf.snap_point(q)
            if np.any(np.isnan(p)) or not pf.is_navigable(p):
                continue
            if pf.distance_to_closest_obstacle(p) < clearance:
                continue
            d = abs(np.linalg.norm(np.asarray(p) - c) - rad)   # 离目标环最贴的
            if d < bd:
                bd, best = d, np.asarray(p, np.float32)
        if best is not None:
            return best                       # 最内环找到就用 (离物体最近)
    return best


def rdp(pts, eps=0.5):
    """Ramer-Douglas-Peucker 折线简化 (xz 平面), 返回保留点索引。去碎转。"""
    pts = np.asarray(pts)
    if len(pts) < 3:
        return list(range(len(pts)))

    def rec(a, b):
        if b <= a + 1:
            return [a]
        p0, p1 = pts[a, [0, 2]], pts[b, [0, 2]]
        d = p1 - p0; L = np.linalg.norm(d)
        if L < 1e-6:
            ds = np.linalg.norm(pts[a + 1:b, [0, 2]] - p0, axis=1)
        else:
            n = np.array([-d[1], d[0]]) / L
            ds = np.abs((pts[a + 1:b, [0, 2]] - p0) @ n)
        k = int(np.argmax(ds)) + a + 1
        if ds[k - a - 1] > eps:
            return rec(a, k) + rec(k, b)
        return [a]
    return rec(0, len(pts) - 1) + [len(pts) - 1]


def resample_waypoints(locs, anchors, step=1.8, min_pts=5):
    """从密集轨迹 locs(走过的每帧位置) 生成 reference_path waypoint。
    以 RDP 角点 anchors 为分段锚(保留转弯结构), 每段再按弧长 ~step 均匀插点,
    使相邻 waypoint 间距 <= step(贴近原始 R2R/RxR ~1.8m 采样密度); 角点附近自然更密。
    若点数仍 < min_pts(短轨迹) 则全程按更细步长兜底重采样, 禁止退化成 2 点。
    返回 list[[x,y,z]]。"""
    locs = np.asarray(locs, np.float32)
    if len(locs) < 2:
        return [p.tolist() for p in locs]

    def _seg_resample(seg, st):
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        L = cum[-1]
        n = max(1, int(round(L / st)))            # 该段切成 n 份
        res = []
        for k in range(1, n + 1):
            s = L * k / n
            j = min(max(int(np.searchsorted(cum, s)) - 1, 0), len(d) - 1)
            t = (s - cum[j]) / max(d[j], 1e-9)
            res.append(seg[j] * (1 - t) + seg[j + 1] * t)
        return res

    anchors = sorted(set(int(i) for i in anchors) | {0, len(locs) - 1})
    pts = [locs[anchors[0]]]
    for a, b in zip(anchors[:-1], anchors[1:]):
        pts.extend(_seg_resample(locs[a:b + 1], step))
    # 去掉过近重复点
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > 1e-2:
            out.append(p)
    # 兜底: 不足 min_pts 时全程均匀重采样到 min_pts 个点
    if len(out) < min_pts:
        d = np.linalg.norm(np.diff(locs, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        L = cum[-1]
        out = []
        for k in range(min_pts):
            s = L * k / (min_pts - 1)
            j = min(max(int(np.searchsorted(cum, s)) - 1, 0), len(d) - 1)
            t = (s - cum[j]) / max(d[j], 1e-9)
            out.append(locs[j] * (1 - t) + locs[j + 1] * t)
    return [p.tolist() for p in out]


def count_turns(points, deg_thresh=30.0):
    pts = np.asarray(points)
    if len(pts) < 3:
        return 0
    segs = pts[1:, [0, 2]] - pts[:-1, [0, 2]]
    turns = 0
    for i in range(len(segs) - 1):
        v1, v2 = segs[i], segs[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        if math.degrees(math.acos(cos)) > deg_thresh:
            turns += 1
    return turns


def classify_difficulty(geod, eu, turns, floor_span):
    detour = geod / max(eu, 1e-3)
    if floor_span > 1.5 or detour >= 1.6 or turns >= 4:
        return "hard"
    if detour >= 1.25 or turns >= 2:
        return "medium"
    return "easy"


def yaw_quat(yaw):
    q = hsc.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


# ----------------------------- 导航图 (ScaleVLN 式) -----------------------------
def dense_sample(pf, n_target, min_sep, seed):
    pf.seed(seed)
    pts, tree_pts, attempts = [], np.empty((0, 3)), 0
    while len(pts) < n_target and attempts < n_target * 30:
        attempts += 1
        p = pf.get_random_navigable_point()
        if np.any(np.isnan(p)):
            continue
        if tree_pts.shape[0] and cKDTree(tree_pts).query(p)[0] < min_sep:
            continue
        pts.append(np.asarray(p, np.float32)); tree_pts = np.vstack([tree_pts, p])
    return np.array(pts)


def cluster_nodes(pts, thresh, pf):
    if len(pts) < 2:
        return pts
    labels = fcluster(linkage(pdist(pts), method="average"), t=thresh, criterion="distance")
    nodes = []
    for lab in np.unique(labels):
        snapped = pf.snap_point(pts[labels == lab].mean(axis=0))
        if not np.any(np.isnan(snapped)):
            nodes.append(np.asarray(snapped, np.float32))
    return np.array(nodes)


def build_edges(nodes, pf, radius, max_deg, straight):
    tree = cKDTree(nodes); deg = defaultdict(int); edges = set()
    for i in range(len(nodes)):
        cand = [j for j in tree.query_ball_point(nodes[i], radius) if j != i]
        cand.sort(key=lambda j: np.linalg.norm(nodes[i] - nodes[j]))
        for j in cand:
            if deg[i] >= max_deg:
                break
            if deg[j] >= max_deg or (min(i, j), max(i, j)) in edges:
                continue
            eu = np.linalg.norm(nodes[i] - nodes[j])
            if eu < 1e-3:
                continue
            if geo(pf, nodes[i], nodes[j]) / eu < straight:
                edges.add((min(i, j), max(i, j))); deg[i] += 1; deg[j] += 1
    return edges


def largest_component(n, edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    seen, best = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, q = [], deque([s]); seen.add(s)
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        if len(comp) > len(best):
            best = comp
    keep = set(best)
    return keep, {(a, b) for a, b in edges if a in keep and b in keep}


def build_nav_graph(pf, seed, min_sep=0.4, cluster_thresh=1.0, connect_radius=5.0,
                    max_degree=5, edge_straight=1.1, oversample=2.0):
    """从 pathfinder 直接构图, 返回 (nodes[Nx3], edges[[i,j],...])。"""
    n_target = int(pf.navigable_area / (min_sep ** 2) * oversample)
    pts = dense_sample(pf, n_target, min_sep, seed)
    nodes = cluster_nodes(pts, cluster_thresh, pf)
    np.random.seed(seed)
    edges = build_edges(nodes, pf, connect_radius, max_degree, edge_straight)
    keep, edges = largest_component(len(nodes), edges)
    remap = {o: n for n, o in enumerate(sorted(keep))}
    nodes = np.array([nodes[i] for i in sorted(keep)], np.float32)
    edges = [[remap[a], remap[b]] for a, b in edges]
    return nodes, edges


def load_or_build_graph(sim, scene, seed, graph_dir=GRAPH_DIR):
    import json
    path = os.path.join(graph_dir, f"{scene}.seed_{seed}.nav_graph.json")
    if os.path.exists(path):
        g = json.load(open(path))
        return np.array(g["nodes"], np.float32), g["edges"]
    nodes, edges = build_nav_graph(sim.pathfinder, seed)
    os.makedirs(graph_dir, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"scene": scene, "seed": seed, "nodes": nodes.tolist(), "edges": edges}, f)
    os.replace(tmp, path)
    return nodes, edges


# ----------------------------- 语义 (房间 + 物体) -----------------------------
def load_regions(sim, with_objects=False):
    """返回房间 dict 列表: {name, c, lo, hi, objs}。objs=[(类别,中心xyz)] (with_objects 时)。"""
    regs = []
    for r in sim.semantic_scene.regions:
        name = r.category.name() if r.category else "?"
        if not name:
            continue
        aabb = r.aabb
        c = aabb.center() if callable(getattr(aabb, "center", None)) else aabb.center
        s = aabb.size() if callable(getattr(aabb, "size", None)) else getattr(aabb, "sizes", None)
        c = np.array(c, np.float32); half = np.array(s, np.float32) / 2
        objs = []
        if with_objects:
            for o in r.objects:
                if not o.category:
                    continue
                cn = o.category.name()
                if not cn or cn in STRUCT:
                    continue
                oa = o.aabb
                oc = oa.center() if callable(getattr(oa, "center", None)) else oa.center
                objs.append((cn, np.array(oc, np.float32)))
        regs.append({"name": name, "c": c, "lo": c - half, "hi": c + half, "objs": objs})
    return regs


def region_at(point, regs):
    p = np.asarray(point, np.float32)
    for r in regs:
        if np.all(p >= r["lo"]) and np.all(p <= r["hi"]):
            return r["name"]
    return None


def assign_room(nodes, regs):
    centers = np.array([r["c"] for r in regs]) if regs else np.zeros((1, 3))
    out = []
    for p in nodes:
        hit = -1
        for i, r in enumerate(regs):
            if np.all(p >= r["lo"]) and np.all(p <= r["hi"]):
                hit = i; break
        if hit < 0 and regs:
            hit = int(np.argmin(np.linalg.norm(centers - p, axis=1)))
        out.append(hit)
    return np.array(out)


def nearest_landmark(p, regs, room_idx, exclude=()):
    best, bd = None, 1e9
    pool = regs[room_idx]["objs"] if 0 <= room_idx < len(regs) else []
    for cn, oc in pool:
        if cn in exclude:
            continue
        d = np.linalg.norm(oc - p)
        if d < bd:
            bd, best = d, cn
    rname = regs[room_idx]["name"] if 0 <= room_idx < len(regs) else "area"
    return best, rname


def split_floors(nodes, gap=1.0, min_nodes=5):
    """按 y 把节点分楼层; 碎层(<min_nodes)并入最近主层。返回 (floor_of[N], {主层->中位y})。"""
    ys = np.sort(nodes[:, 1])
    centers = [ys[0]]
    for y in ys[1:]:
        if y - centers[-1] > gap:
            centers.append(y)
    centers = np.array(centers)
    floor_of = np.array([int(np.argmin(np.abs(centers - y))) for y in nodes[:, 1]])
    main = [fi for fi in range(len(centers)) if (floor_of == fi).sum() >= min_nodes]
    if not main:
        main = [int(Counter(floor_of).most_common(1)[0][0])]
    mc = centers[main]
    remap = {fi: main[int(np.argmin(np.abs(mc - centers[fi])))] for fi in range(len(centers))}
    floor_of = np.array([remap[f] for f in floor_of])
    return floor_of, {m: float(np.median(nodes[floor_of == m, 1])) for m in main}


# ----------------------------- 正交俯视渲染 -----------------------------
def render_topdown(scene_glb, floor_y, bounds, mpp=0.04, cam_up=1.8, max_px=1600):
    """渲染某楼层纹理俯视图, 返回 (rgb, to_px, w, h)。to_px(p)->(col,row)。"""
    lo, hi = np.array(bounds[0]), np.array(bounds[1])
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    w = min(max_px, int(math.ceil((hi[0] - lo[0]) / mpp)) + 20)
    h = min(max_px, int(math.ceil((hi[2] - lo[2]) / mpp)) + 20)
    sim = habitat_sim.Simulator(make_ortho_cfg(scene_glb, w, h, mpp))
    ag = sim.get_agent(0); st = ag.get_state()
    st.position = np.array([cx, floor_y + cam_up, cz], np.float32)
    st.rotation = np.quaternion(1, 0, 0, 0); ag.set_state(st)
    rgb = np.asarray(sim.get_sensor_observations()["ortho"])[:, :, :3].copy()
    sim.close()

    def to_px(p):
        return (p[0] - cx) / mpp + w / 2, (p[2] - cz) / mpp + h / 2
    return rgb, to_px, w, h


# ----------------------------- follower 走轨迹 -----------------------------
def walk_follower(sim, follower, agent, start, yaw, waypoints):
    """沿 waypoints 用 greedy follower 走, 返回 (actions[int], locations[xyz])。"""
    st = agent.get_state()
    st.position = np.array(start, np.float32)
    st.rotation = hsc.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(st)
    actions = []; locations = [list(map(float, agent.get_state().position))]
    for wp in waypoints:
        for _ in range(2000):
            try:
                a = follower.next_action_along(np.array(wp, np.float32))
            except habitat_sim.errors.GreedyFollowerError:
                a = "stop"
            if a == "stop" or a is None:
                break
            sim.step(a); actions.append(ACT[a])
            locations.append(list(map(float, agent.get_state().position)))
    actions.append(ACT["stop"])
    return actions, locations


def walk_pathfollow(sim, follower, agent, start, yaw, waypoints, pf=None):
    """快通道: 与 walk_follower 同签名/同返回, 但不调用昂贵的 greedy follower 规划。
    先用 pathfinder.find_path 拿每段 navmesh 测地折线(C++, 一次调用), 再用便宜的几何控制
    (朝目标顶点转 15°/前进 0.25)驱动 sim.step。**动作与位置仍来自 sim.step -> 自洽**(replay
    可复现), 只把 O(贵) 的 next_action_along 换成 O(便宜) 的角度判断。约 13x 墙钟。
    转向符号用 sim 真实朝向向量(quat_rotate_vector)算, 不假设 yaw 约定; 符号错也只多转不出错。
    pf 缺省用 sim.pathfinder。follower 参数仅为签名对齐, 不使用。"""
    if pf is None:
        pf = sim.pathfinder
    turn = math.radians(TURN_ANGLE)
    st = agent.get_state()
    st.position = np.array(start, np.float32)
    st.rotation = hsc.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(st)
    actions = []; locations = [list(map(float, agent.get_state().position))]
    cur_pos = np.array(start, np.float32)
    wl = list(waypoints)
    for wi, wp in enumerate(wl):
        wp = np.array(wp, np.float32)
        # 每段起点用当前实际位置, 拿 navmesh 折线顶点作为途经小目标
        sp = habitat_sim.ShortestPath(); sp.requested_start = cur_pos; sp.requested_end = wp
        verts = [np.asarray(p, np.float32) for p in sp.points] if pf.find_path(sp) else [wp]
        verts = verts[1:] if len(verts) > 1 else [wp]     # 去掉起点顶点
        last_wp = (wi == len(wl) - 1)
        for vi, tg in enumerate(verts):
            last_vert = last_wp and vi == len(verts) - 1
            rad = 0.20 if last_vert else 0.30            # 终点收紧, 保证过 0.4m 到达校验
            for _ in range(2000):
                sstate = agent.get_state()
                pos = np.asarray(sstate.position, np.float32)
                if math.hypot(float(tg[0] - pos[0]), float(tg[2] - pos[2])) <= rad:
                    break
                fwd = hsc.quat_rotate_vector(sstate.rotation, np.array([0.0, 0.0, -1.0]))
                cur = math.atan2(float(fwd[0]), float(fwd[2]))
                des = math.atan2(float(tg[0] - pos[0]), float(tg[2] - pos[2]))
                dth = (des - cur + math.pi) % (2 * math.pi) - math.pi
                a = ("turn_left" if dth > 0 else "turn_right") if abs(dth) > turn / 2 else "move_forward"
                sim.step(a); actions.append(ACT[a])
                locations.append(list(map(float, agent.get_state().position)))
        cur_pos = np.asarray(agent.get_state().position, np.float32)
    actions.append(ACT["stop"])
    return actions, locations


# ----------------------------- 词表 -----------------------------
_WORD = re.compile(r"[a-z]+")


def fill_vocab(episodes):
    """给 episodes 回填 instruction_tokens, 返回 word_list。"""
    vocab = ["<pad>", "<unk>"]; vidx = {w: i for i, w in enumerate(vocab)}
    for ep in episodes:
        toks = []
        for w in _WORD.findall(ep["instruction"]["instruction_text"].lower()):
            if w not in vidx:
                vidx[w] = len(vocab); vocab.append(w)
            toks.append(vidx[w])
        ep["instruction"]["instruction_tokens"] = toks
    return vocab

AMAP = {1: "move_forward", 2: "turn_left", 3: "turn_right", 0: "stop"}

SKIP_CATS = (set(STRUCT) | {"objects", "misc", "unlabeled", "void", "ceiling", "floor"}) \
            - {"door", "window", "column", "railing"}

ZH = {
    "bedroom": "卧室", "bathroom": "卫生间", "toilet": "厕所", "kitchen": "厨房",
    "living room": "客厅", "dining room": "餐厅", "family room": "家庭厅",
    "familyroom/lounge": "家庭厅/休息室", "hallway": "走廊", "stairs": "楼梯",
    "office": "办公室", "closet": "壁橱/衣帽间", "porch/terrace/deck": "阳台/露台",
    "spa/sauna": "spa/桑拿房", "workout/gym/exercise": "健身房", "lounge": "休息室",
    "garage": "车库", "lobby": "门厅", "laundryroom/mudroom": "洗衣房",
    "utilityroom/toolroom": "杂物间", "meetingroom/conferenceroom": "会议室",
    "tv": "影音室", "bar": "吧台区", "balcony": "阳台",
    "entryway/foyer/lobby": "玄关/门厅", "library": "图书室/书房", "classroom": "教室",
    "dining booth": "卡座", "rec/game": "娱乐室", "outdoor": "室外", "junk": "杂物区",
    "no label": "未标注", "other room": "其他房间", "unknown": "未知",
    "chair": "椅子", "table": "桌子", "picture": "画框", "cabinet": "橱柜",
    "cushion": "靠垫", "sofa": "沙发", "bed": "床", "chest_of_drawers": "抽屉柜",
    "plant": "植物", "sink": "水槽", "stool": "凳子", "towel": "毛巾",
    "mirror": "镜子", "tv_monitor": "电视", "shower": "淋浴", "counter": "台面",
    "fireplace": "壁炉", "lighting": "灯具", "shelving": "置物架", "blinds": "百叶窗",
    "seating": "座椅", "furniture": "家具", "appliances": "电器", "clothes": "衣物",
    "curtain": "窗帘", "bathtub": "浴缸", "gym_equipment": "健身器材",
    "stairs": "楼梯", "railing": "栏杆", "column": "柱子", "toilet": "马桶",
    "dresser": "梳妆台", "wardrobe": "衣柜", "desk": "书桌", "refrigerator": "冰箱",
    "fridge": "冰箱", "oven": "烤箱", "microwave": "微波炉", "fan": "风扇",
    "lamp": "台灯", "vase": "花瓶", "rug": "地毯", "carpet": "地毯",
    "board_panel": "板/隔断", "bench": "长凳", "shoes": "鞋",
}
def base_cat(name):
    """去掉房间/物体标签的 '#序号', 取基础类别名 (用于查表/交叉校验)。"""
    return name.split("#")[0] if name else name


def bilingual(name):
    base, _, num = name.partition("#")
    z = ZH.get(base)
    en = f"{base}#{num}" if num else base       # 英文侧保留序号: 'bedroom#2 / 卧室'
    return f"{en} / {z}" if z else en


MP3D_SCENE_CFG = os.path.join(SCENES_DIR, "mp3d.scene_dataset_config.json")


def make_sem_cfg(glb, with_rgb=False, gpu=0):
    sc = habitat_sim.SimulatorConfiguration()
    # 关键: 走 scene_dataset_config, 否则语义 ply 的 semantic_up/front 变换不生效,
    # 语义传感器会与 RGB 错位(habitat v0.3.x + MP3D 已知 bug)。
    if os.path.exists(MP3D_SCENE_CFG):
        sc.scene_dataset_config_file = MP3D_SCENE_CFG
        sc.scene_id = os.path.basename(glb).replace(".glb", "")
    else:
        sc.scene_id = glb
    sc.enable_physics = False
    sc.gpu_device_id = gpu
    ag = habitat_sim.agent.AgentConfiguration()
    ag.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=float(TURN_ANGLE))),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=float(TURN_ANGLE))),
    }
    sem = habitat_sim.CameraSensorSpec()
    sem.uuid = "sem"
    sem.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem.resolution = [480, 640]
    sem.position = [0.0, 1.25, 0.0]
    # depth 传感器渲染的是 .glb 渲染网格(与 RGB 同一套墙), 用于对语义 .ply
    # 的"穿墙"实例做遮挡门控: MP3D 语义 ply 墙薄/有洞, sem 会看到隔壁物体。
    dep = habitat_sim.CameraSensorSpec()
    dep.uuid = "depth"
    dep.sensor_type = habitat_sim.SensorType.DEPTH
    dep.resolution = [480, 640]
    dep.position = [0.0, 1.25, 0.0]
    specs = [sem, dep]
    if with_rgb:
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [480, 640]
        rgb.position = [0.0, 1.25, 0.0]
        specs = [rgb, sem, dep]
    ag.sensor_specifications = specs
    return habitat_sim.Configuration(sc, [ag])


def _cs(a):
    c = a.center() if callable(getattr(a, "center", None)) else a.center
    s = a.size() if callable(getattr(a, "size", None)) else a.sizes
    return np.array(c, np.float32), np.array(s, np.float32) / 2.0


def build_maps(sim):
    """semantic_id -> (类别, 中心xyz, AABB半长, 所属region类别); 以及 floor 物体 (几何兜底判房间)。"""
    ss = sim.semantic_scene
    id2cat, id2ctr, id2reg, id2half = {}, {}, {}, {}
    for o in ss.objects:
        if o is None:
            continue
        c, h = _cs(o.aabb)
        id2cat[o.semantic_id] = o.category.name() if o.category else "?"
        id2ctr[o.semantic_id] = c
        id2half[o.semantic_id] = h
    floors = []
    per_reg = {}   # 每个 region 一个实例号(同类多个房间可区分, 重复进入=同号)
    for r in ss.regions:
        if not r.category:
            continue
        rc = r.category.name()
        per_reg[rc] = per_reg.get(rc, 0) + 1
        label = f"{rc}#{per_reg[rc]}"
        for o in r.objects:
            if o is None:
                continue
            id2reg[o.semantic_id] = label
            if o.category and o.category.name() == "floor":
                c, h = _cs(o.aabb)
                floors.append((label, c, h))
    return id2cat, id2ctr, id2reg, id2half, floors


def aabb_dist(cam, ctr, half):
    """相机到物体 AABB 表面的 3D 距离(在 box 内则为 0); ctr/half 缺失返回大值。
    比中心距更准: 大件物体(长吧台/大沙发/楼梯)贴身时中心距虚高, 会被 maxdist 误杀。"""
    if ctr is None or half is None:
        return 1e9
    d = np.maximum(np.abs(np.asarray(cam, np.float32) - ctr) - half, 0.0)
    return float(np.linalg.norm(d))


def occluded_ids(sem, depth, cam, id2ctr, id2half, margin=0.6):
    """穿墙实例门控: 语义 ply 墙薄/有洞, sem 会看到隔壁房间的物体。若某实例像素处的
    depth(渲染 glb 网格, 与 RGB 同一套墙)明显近于到该实例 AABB 的距离, 说明中间隔着
    一堵 glb 的墙 -> 判为被遮挡。返回被遮挡的 semantic_id 集合。"""
    if depth is None:
        return set()
    d = np.asarray(depth)
    bad = set()
    for sid in np.unique(sem):
        sid = int(sid)
        if sid not in id2half:
            continue
        m = (sem == sid) & (d > 1e-3)          # 排除无效/背景深度
        if int(m.sum()) < 16:
            continue
        if float(np.median(d[m])) + margin < aabb_dist(cam, id2ctr.get(sid), id2half.get(sid)):
            bad.add(sid)
    return bad


def room_from_sem(sem_img, id2cat, id2reg):
    """看画面最底部窄条(脚下最近地板)的 floor 像素属于哪个 region -> 当前房间。
    只取底部窄条而非整个下半屏, 避免看到前方别的房间地板导致房间抖动。"""
    patch = sem_img[430:, 224:416].reshape(-1)
    vals, cnts = np.unique(patch, return_counts=True)
    votes = Counter()
    for sid, c in zip(vals.tolist(), cnts.tolist()):
        if id2cat.get(sid) == "floor" and sid in id2reg:
            votes[id2reg[sid]] += c
    return votes.most_common(1)[0][0] if votes else None


def room_at(p, floors):
    """几何兜底: agent 脚下属于哪块 floor patch -> region 类别 (语义投票失败时用)。"""
    p = np.asarray(p, np.float32)
    best, bd = "unknown", 1e9
    for rc, c, h in floors:
        if abs(p[0] - c[0]) <= h[0] + 0.1 and abs(p[2] - c[2]) <= h[2] + 0.1:
            dy = p[1] - c[1]
            if -0.4 <= dy <= 2.6 and abs(dy) < bd:
                bd, best = abs(dy), rc
    return best


NEAR_DIST = 3.5  # 这么近的物体即使 region 不匹配也算地标(如脚下的楼梯)
MIN_ROOM_RUN = 6  # 房间连续驻留不足此步数视为"路过", 不记录为 Location

# 显著地标类别: 导航辨识度高, 即使占屏小也保留(放宽入选阈值), 排序时加权上浮。
STRONG_LM = {"picture", "tv_monitor", "fireplace", "mirror", "stairs", "toilet",
             "bed", "sofa", "refrigerator", "fridge", "sink", "plant", "bathtub",
             "oven", "microwave", "vase", "piano"}
# 大而泛用、辨识度低: 易靠体量霸榜、把小而醒目的地标挤掉, 排序时降权。
BULKY_GENERIC = {"cabinet", "shelving", "counter", "furniture", "curtain", "blinds",
                 "board_panel", "seating", "railing", "column", "wardrobe",
                 "chest_of_drawers", "dresser", "rug", "carpet"}
LM_WEIGHT = {**{c: 2.0 for c in STRONG_LM}, **{c: 0.5 for c in BULKY_GENERIC}}

# 房间特征物 (mpcat40): 用于交叉校验 floor 房间标签
ROOM_SIG = {
    "bathroom": {"toilet", "shower", "bathtub", "towel"},
    "bedroom": {"bed"},
    "kitchen": {"oven", "refrigerator", "fridge", "microwave", "stove", "appliances"},
    "living room": {"sofa", "cushion", "tv_monitor", "fireplace"},
}
# floor 房间名 -> ROOM_SIG 键 (判该 floor 标签自身有无特征物支持)
_ROOM_CANON = {
    "toilet": "bathroom", "bathroom": "bathroom", "spa/sauna": "bathroom",
    "bedroom": "bedroom", "kitchen": "kitchen",
    "living room": "living room", "familyroom/lounge": "living room",
    "lounge": "living room", "tv": "living room", "rec/game": "living room",
    "family room": "living room",
}


def peak_objects(hists, cams, id2cat, id2ctr, id2half, maxdist):
    """节点跨段内, 每个非结构物体类别的单帧最大占屏比(视线内 ∩ ≤maxdist), 不做 region 过滤。"""
    tot = 480 * 640
    peak = {}
    for h, cam in zip(hists, cams):
        for sid, n in h.items():
            sid = int(sid)
            cat = id2cat.get(sid, "?")
            if cat in SKIP_CATS or cat == "?":
                continue
            ctr = id2ctr.get(sid)
            if ctr is None or aabb_dist(cam, ctr, id2half.get(sid)) > maxdist:
                continue
            f = n / tot
            if f > peak.get(cat, 0.0):
                peak[cat] = f
    return peak


def reconcile_room(floor_room, peak, strong=0.04, floor_min=0.015):
    """强物体证据优先: 若 floor 标签自身无特征物支持(<floor_min), 而某房间特征物占屏强(≥strong),
    则按物体覆盖 floor 标签; 否则保留 floor。返回 (房间, 是否被物体覆盖)。"""
    scores = {r: sum(peak.get(o, 0.0) for o in objs) for r, objs in ROOM_SIG.items()}
    best = max(scores, key=lambda r: scores[r])
    best_s = scores[best]
    floor_key = _ROOM_CANON.get(base_cat(floor_room))   # 房间标签带 #序号, 取基础类别校验
    floor_s = scores.get(floor_key, 0.0) if floor_key else 0.0
    if best_s >= strong and floor_s < floor_min and best != floor_key:
        return best, True                               # 物体证据覆盖: 回落到类别(无区域号)
    return floor_room, False


def landmarks_from(hists, cams, id2cat, id2ctr, id2half, id2reg, allowed, topn, maxdist, min_frac):
    """该段每一帧里, 物体类别的单帧最大占屏比 (峰值)。保留条件:
    视线内 ∩ 峰值≥阈值 ∩ ( 物体所属 region∈本节点房间 且 ≤maxdist
                          或 距相机(AABB表面)≤NEAR_DIST(很近, 不管 region) )。
    -> 剔除"隔开放空间/门口瞥到的别房间远物"(如客厅里的健身器材),
       但保住脚下很近的相关物(如正在下的楼梯)。
    排序按【显著性 = 峰值占屏 × 类别权重】: 显著地标(picture/楼梯/镜子等)放宽入选阈值
    (min_frac×0.4)并上浮, 大而泛用物(橱柜/置物架)降权, 避免它们靠体量霸榜挤掉小地标。"""
    tot = 480 * 640
    peak = {}
    for h, cam in zip(hists, cams):
        for sid, n in h.items():
            sid = int(sid)
            cat = id2cat.get(sid, "?")
            if cat in SKIP_CATS or cat == "?":
                continue
            ctr = id2ctr.get(sid)
            if ctr is None:
                continue
            d = aabb_dist(cam, ctr, id2half.get(sid))      # 到 AABB 表面, 不用中心距
            if d > maxdist:
                continue
            if allowed is not None:                        # 精度模式: region 约束
                in_region = id2reg.get(sid) in allowed
                if not in_region and d > NEAR_DIST:
                    continue
            f = n / tot
            if f > peak.get(cat, 0.0):
                peak[cat] = f
    cand = []
    for cat, f in peak.items():
        thr = min_frac * 0.4 if cat in STRONG_LM else min_frac
        if f >= thr:
            cand.append((f * LM_WEIGHT.get(cat, 1.0), cat))
    cand.sort(reverse=True)
    return [c for _, c in cand[:topn]]


def turn_label(net_deg, turn_deg=45):
    a = abs(net_deg)
    if a < turn_deg:      # 净转角 <turn_deg° 视为直行, 不算转弯 (默认45, 可调30)
        return None
    if a >= 150:
        return "Turn around"
    side = "left" if net_deg > 0 else "right"
    return f"Turn {side}" if a >= 75 else f"Turn slightly {side}"


def merge_sig(actions, dy, is_last, turn_deg=45):
    """合并签名: 同 (转向方向, 上下楼, 是否停止) 的连续航点并为一个节点 (跨房间也合并)。
    转向只看方向(左/右/掉头/直行), 不区分 slightly; 房间作为节点内的路径单独记录。"""
    net = sum(15 if a == 2 else (-15 if a == 3 else 0) for a in actions)
    if abs(net) >= 150:
        d = "around"
    elif net >= turn_deg:
        d = "left"
    elif net <= -turn_deg:
        d = "right"
    else:
        d = "straight"
    stair = "up" if dy > 0.7 else ("down" if dy < -0.7 else "")
    return (d, stair, bool(is_last))


def fwd_phrase(nfwd):
    """按前进距离(步数×0.25m)给 walk forward 加程度词: 长/短/中(不加)。"""
    m = nfwd * FORWARD_STEP
    if m >= 8.0:
        return "walk forward a long way"
    if m <= 1.5:
        return "walk forward a short way"
    return "walk forward"


FWD_MIN_DISP = 0.5  # 前进段净水平位移≤此值视为"原地"(撞墙/被挡/仅调整), 不算 walk forward


def seg_action(actions, dy, is_first, is_last, disp=None, turn_deg=45, max_parts=3):
    """按【时序】把动作流切成 转向段/前进段, 保序生成短语(每个转向段独立判净角);
    不再把整段转向塌缩成单一净角 -> 能区分 '先走到门口再左转' 这类。最多 max_parts 段。"""
    # 时序切段: 连续 forward 归一段(记步数), 连续 turn 归一段(记净角)
    seq, i = [], 0
    while i < len(actions):
        a = actions[i]
        if a == 1:
            j = i
            while j < len(actions) and actions[j] == 1:
                j += 1
            seq.append(("fwd", j - i)); i = j
        elif a in (2, 3):
            j, net = i, 0
            while j < len(actions) and actions[j] in (2, 3):
                net += 15 if actions[j] == 2 else -15; j += 1
            seq.append(("turn", net)); i = j
        else:
            i += 1
    stair = "walk downstairs" if dy < -0.7 else ("walk upstairs" if dy > 0.7 else "walk forward")

    def fold(runs):
        """相邻同类段合并(转向累加净角, 前进累加步数), 保证严格交替。"""
        out = []
        for k, v in runs:
            if out and out[-1][0] == k:
                out[-1][1] += v
            else:
                out.append([k, v])
        return out

    # 1. 先丢子阈值微转 + 折叠: 把被微转打散的前进拼回来(否则碎步会被逐段当原地丢光, 误判 Adjust heading)
    clean = fold([[k, v] for k, v in seq
                  if not (k == "turn" and turn_label(v, turn_deg) is None)])
    # 2. 再丢合并后仍≤0.5m 的原地前进 + 折叠(此时相邻两转才会并, 因为中间只是原地小挪)
    clean = fold([[k, v] for k, v in clean
                  if not (k == "fwd" and v * FORWARD_STEP <= FWD_MIN_DISP)])

    def sig(run):
        # 段的显著度: 转向按|净角|, 前进按 米数×90(1m≈90°), 便于统一比较取舍
        return abs(run[1]) if run[0] == "turn" else run[1] * FORWARD_STEP * 90.0

    # 超过 max_parts: 优先丢"走"、保留转向(转向=决策点, 更该留); 但不丢夹在两转向之间的走
    # (丢了会让两转向相邻被并成假大转/连续turn); 实在没可丢的走时才丢最不显著的转向。
    while len(clean) > max_parts:
        droppable = [j for j, (k, _) in enumerate(clean)
                     if k == "fwd" and not (0 < j < len(clean) - 1
                                            and clean[j - 1][0] == "turn" and clean[j + 1][0] == "turn")]
        if droppable:
            i = min(droppable, key=lambda j: sig(clean[j]))
        else:
            i = min(range(len(clean)), key=lambda j: sig(clean[j]))
        del clean[i]
        clean = fold(clean)      # 丢"走"后两侧若同类会合并(只并走, 不会并转向)
    parts = []
    for k, v in clean:
        if k == "turn":
            t = turn_label(v, turn_deg)
            if t:
                parts.append(t)
        else:
            parts.append(stair)
    if not parts:
        phrase = "Adjust heading"
    else:
        phrase = parts[0]
        if len(parts) > 1:
            phrase = parts[0] + " and " + " and ".join(p.lower() for p in parts[1:])
    phrase = phrase[0].upper() + phrase[1:]
    if is_first and not parts:
        phrase = "Start"
    elif is_first:
        phrase = "Start, " + phrase[0].lower() + phrase[1:]
    if is_last:
        phrase = phrase + " and Stop"
    return phrase


def nearest_step(loc_arr, wp):
    d = np.linalg.norm(loc_arr - np.asarray(wp, np.float32)[None, :], axis=1)
    return int(np.argmin(d))


def process_episode(sim, ep, gt, maps, topn, maxdist, min_frac, region_filter,
                    capture_rgb=False, obj_room_check=True, cap_off=0, mid_off_thresh=None,
                    turn_deg=45):
    id2cat, id2ctr, id2reg, id2half, floors = maps
    locs = np.array(gt["locations"], np.float32)
    actions = gt["actions"]
    ag = sim.get_agent(0)
    sr = ep["start_rotation"]
    st = ag.get_state()
    st.position = np.array(ep["start_position"], np.float32)
    st.rotation = np.quaternion(sr[3], sr[0], sr[1], sr[2])
    ag.set_state(st)

    # 需要存帧的候选步(航点最近步), 只在这些步抓 RGB, 省内存
    cand = set()
    if capture_rgb:
        nloc = len(locs)
        for wp in ep["reference_path"]:
            s = min(nearest_step(locs, wp), nloc - 1)
            cand.add(s); cand.add(max(0, s - cap_off))   # 本帧和偏移帧都抓, 供条件挑选

    hist, votes, rgb_at = [], [], {}

    def grab(k):
        obs = sim.get_sensor_observations()
        sem = np.asarray(obs["sem"])
        # 穿墙门控: 用 depth(glb 墙)把被墙挡住的语义 ply 实例抹回背景, 不计入地标/房间
        cam = locs[k] if k < len(locs) else np.asarray(ag.get_state().position, np.float32)
        occ = occluded_ids(sem, obs.get("depth"), cam, id2ctr, id2half)
        if occ:
            sem = sem.copy()
            for sid in occ:
                sem[sem == sid] = 0
        hist.append(Counter(sem.reshape(-1).tolist()))
        votes.append(room_from_sem(sem, id2cat, id2reg))
        if capture_rgb and k in cand:
            rgb_at[k] = np.asarray(obs["rgb"])[:, :, :3].copy()

    grab(0)
    k = 0
    for aid in actions:
        if aid == 0:
            break
        sim.step(AMAP[aid])
        k += 1
        grab(k)
    n_steps = k

    # 每步房间: 语义 floor 投票优先, 失败用几何兜底
    raw_room = [votes[s] if votes[s] else room_at(locs[s], floors)
                for s in range(len(hist))]
    # 时间平滑: 窗口众数, 去掉逐帧抖动 (走开放空间时会瞥到前方别的房间地板)
    W = 5
    step_room = []
    for s in range(len(raw_room)):
        win = [r for r in raw_room[max(0, s - W):s + W + 1] if r and r != "unknown"]
        step_room.append(Counter(win).most_common(1)[0][0] if win else "unknown")
    # 房间"存活"过滤: 连续同房间不足 MIN_ROOM_RUN 步的瞬时段记为 unknown(只是路过, 不作为 Location)
    i = 0
    while i < len(step_room):
        j = i
        while j < len(step_room) and step_room[j] == step_room[i]:
            j += 1
        if step_room[i] != "unknown" and (j - i) < MIN_ROOM_RUN:
            for t in range(i, j):
                step_room[t] = "unknown"
        i = j

    def span_room(lo, hi):
        cnt = Counter(r for r in step_room[lo:hi + 1] if r and r != "unknown")
        return cnt.most_common(1)[0][0] if cnt else "unknown"

    def room_path(lo, hi, min_run=4):
        """节点跨越的房间序列: 合并连续同房间, 并丢掉驻留<min_run步的瞬时抖动。"""
        runs = []
        for s in range(lo, hi + 1):
            r = step_room[s]
            if runs and runs[-1][0] == r:
                runs[-1][1] += 1
            else:
                runs.append([r, 1])
        seq = [r for r, n in runs if r and r != "unknown" and n >= min_run]
        # 去掉相邻重复(过滤后可能再次相邻)
        out = []
        for r in seq:
            if not out or out[-1] != r:
                out.append(r)
        return out or [span_room(lo, hi)]

    ref = ep["reference_path"]
    wp_steps = []
    for wp in ref:
        s = min(nearest_step(locs[:len(hist)], wp), len(hist) - 1)
        wp_steps.append(s)
    for i in range(1, len(wp_steps)):
        wp_steps[i] = max(wp_steps[i], wp_steps[i - 1])

    # 每个原始航点的属性
    raw = []
    for i, s in enumerate(wp_steps):
        is_first, is_last = i == 0, i == len(ref) - 1
        a0 = wp_steps[i - 1] if not is_first else 0
        seg_acts = list(actions[a0:s])
        dy = float(ref[i][1] - ref[i - 1][1]) if not is_first else 0.0
        # 相邻 reference_path 航点的水平间距; ≤FWD_MIN_DISP 视为"原地", 不计入前进
        wp_disp = 0.0 if is_first else float(np.hypot(ref[i][0] - ref[i - 1][0],
                                                       ref[i][2] - ref[i - 1][2]))
        pos = locs[s] if s < len(locs) else np.array(ref[i], np.float32)
        room = span_room(a0, s)
        raw.append({
            "i": i, "step": s, "a0": a0, "acts": seg_acts, "dy": dy,
            "is_first": is_first, "is_last": is_last, "room": room, "pos": pos,
            "wp_disp": wp_disp, "sig": merge_sig(seg_acts, dy, is_last, turn_deg),
        })

    # 聚类合并: 连续 merge_sig 相同的航点并为一个节点 (累计动作子序列与 dy)
    nodes = []
    for r in raw:
        # Start(第一个航点)强制单独成节点, 不与后续合并
        can_merge = (nodes and nodes[-1]["sig"] == r["sig"]
                     and not r["is_first"] and not nodes[-1]["is_first"])
        eff_fwd = r["wp_disp"] if r["wp_disp"] > FWD_MIN_DISP else 0.0   # 原地航点不计入前进
        if can_merge:
            nd = nodes[-1]
            nd["members"].append(r["i"]); nd["steps"].append(r["step"])
            nd["acts"] += r["acts"]; nd["dy"] += r["dy"]
            nd["cams"].append(r["pos"]); nd["end_pos"] = r["pos"]
            nd["is_last"] = r["is_last"]; nd["fwd_disp"] += eff_fwd
        else:
            nodes.append({
                "sig": r["sig"], "room": r["room"], "members": [r["i"]],
                "steps": [r["step"]], "a0": r["a0"], "acts": list(r["acts"]),
                "dy": r["dy"], "is_first": r["is_first"], "is_last": r["is_last"],
                "cams": [r["pos"]], "end_pos": r["pos"], "fwd_disp": eff_fwd,
            })

    out_nodes = []
    for ni, nd in enumerate(nodes):
        s_lo, s_hi = nd["a0"], nd["steps"][-1]
        lo, hi = min(s_lo, s_hi), s_hi
        hists = hist[lo:hi + 1] or [hist[s_hi]]
        cams = list(locs[lo:hi + 1]) or [locs[s_hi]]
        # 节点有效前进 = 成员航点里 >0.5m 的间距之和(原地航点已剔除), 判定是否算 walk forward
        action = seg_action(nd["acts"], nd["dy"], nd["is_first"], nd["is_last"], nd["fwd_disp"], turn_deg)
        froom = span_room(lo, hi)
        if obj_room_check:
            pk = peak_objects(hists, cams, id2cat, id2ctr, id2half, maxdist)
            froom, overridden = reconcile_room(froom, pk)
        out_nodes.append({
            "node_index": ni,
            "waypoints": nd["members"],
            "step": nd["steps"][-1],
            "step_range": [int(s_lo), int(s_hi)],
            "Action": action,
            "Location": bilingual(froom),
            "position": [round(float(x), 3) for x in nd["end_pos"]],
        })

    # 节点平均间距 (相邻节点代表位置的欧氏距离)
    centers = [np.array(n["position"], np.float32) for n in out_nodes]
    gaps = [float(np.linalg.norm(centers[i] - centers[i - 1])) for i in range(1, len(centers))]
    avg_gap = round(float(np.mean(gaps)), 3) if gaps else 0.0
    node_rgbs = None
    if capture_rgb:
        node_rgbs, prev = [], None
        for n in out_nodes:
            s = min(n["step"], len(locs) - 1)
            if mid_off_thresh is not None:
                # 仅当距上一节点步数 > 阈值(长段)才用偏移帧, 否则用本帧
                use = max(0, s - cap_off) if (prev is not None and (n["step"] - prev) > mid_off_thresh) else s
            else:
                use = max(0, s - cap_off)
            node_rgbs.append(rgb_at.get(use))
            prev = n["step"]
    return out_nodes, n_steps, len(raw), avg_gap, [round(g, 3) for g in gaps], node_rgbs




ID2HALF = {}
SKIP = SKIP_CATS
LM_BLACKLIST = {"", "?", "unlabeled", "void", "unknown", "remove", "objects", "misc",
                "object", "three", "stuff", "clutter", "decoration",
                "clothes", "shoes", "book", "books", "keyboard", "skateboard",
                "toy", "toys", "paper", "papers", "folders"}
STRUCT_LM = {"door", "window", "column", "railing"}
RAW_LM_BASES = set()
_HOUSE_RAW_CACHE = {}


def build_maps_inst(sim):
    """build_maps plus per-instance labels id -> 'cat#k'."""
    id2cat, id2ctr, id2reg, id2half, floors = build_maps(sim)
    per_cat, id2label, inv = {}, {}, {}
    ID2HALF.clear()
    for o in sim.semantic_scene.objects:
        if o is None or not o.category:
            continue
        cat = o.category.name()
        if not cat or cat in SKIP:
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        id2label[o.semantic_id] = f"{cat}#{per_cat[cat]}"
        inv[cat] = inv.get(cat, 0) + 1
        ID2HALF[o.semantic_id] = _cs(o.aabb)
    inventory = ", ".join(f"{c}×{n}" for c, n in sorted(inv.items(), key=lambda x: -x[1]))
    return (id2cat, id2ctr, id2reg, id2half, floors), id2label, inventory


def _house_sid2raw(house_path):
    if house_path in _HOUSE_RAW_CACHE:
        return _HOUSE_RAW_CACHE[house_path]
    cat_name, obj_cat = {}, {}
    with open(house_path) as f:
        for ln in f:
            t = ln.split()
            if not t:
                continue
            if t[0] == "C":
                cat_name[int(t[1])] = t[3]
            elif t[0] == "O":
                obj_cat[int(t[1])] = int(t[3])
    m = {o: cat_name.get(c, "?") for o, c in obj_cat.items()}
    _HOUSE_RAW_CACHE[house_path] = m
    return m


def inject_raw_landmarks(sim, id2cat, id2label, house_path, coarse=("objects", "misc")):
    sid2raw = _house_sid2raw(house_path)
    aabb = {o.semantic_id: o.aabb for o in sim.semantic_scene.objects if o and o.category}
    per, added = {}, []
    for sid, c in list(id2cat.items()):
        if c not in coarse:
            continue
        raw = sid2raw.get(sid)
        if not raw:
            continue
        base = raw.replace("#", " ").strip()
        if "/" in base or not base or base in LM_BLACKLIST:
            continue
        per[base] = per.get(base, 0) + 1
        id2cat[sid] = base
        id2label[sid] = f"{base}#{per[base]}"
        if sid in aabb:
            ID2HALF[sid] = _cs(aabb[sid])
        added.append(base)
        RAW_LM_BASES.add(base)
    return sorted(set(added))


def _inst_dist(sid, cam):
    ch = ID2HALF.get(sid)
    if ch is None:
        return 1e9
    c, h = ch
    d = np.maximum(np.abs(cam - c) - h, 0.0)
    return float(np.linalg.norm(d))


def _inst_occluded(sem, depth, cam, margin=0.6):
    if depth is None:
        return set()
    d = np.asarray(depth)
    bad = set()
    for sid in np.unique(sem):
        sid = int(sid)
        if sid not in ID2HALF:
            continue
        m = (sem == sid) & (d > 1e-3)
        if m.sum() < 16:
            continue
        d_med = float(np.median(d[m]))
        if d_med + margin < _inst_dist(sid, cam):
            bad.add(sid)
    return bad


def _keep_per_base(items, cap):
    seen, out = {}, []
    for k in items:
        b = k.split("#")[0]
        seen[b] = seen.get(b, 0) + 1
        if seen[b] <= cap:
            out.append(k)
    return out


def visible_instances(sem, cam, id2cat, id2ctr, id2label, depth=None, maxdist=5.0, min_frac=0.002, topn=6):
    tot = 480 * 640
    occ = _inst_occluded(sem, depth, cam)
    dist = {}
    for sid, n in zip(*np.unique(sem, return_counts=True)):
        sid = int(sid)
        cat = id2cat.get(sid, "?")
        if cat in SKIP or cat == "?" or sid not in id2label or sid in occ:
            continue
        d = _inst_dist(sid, cam)
        if d > maxdist or n / tot < min_frac:
            continue
        lab = id2label[sid]
        dist[lab] = min(dist.get(lab, 1e9), d)
    labels = sorted(dist, key=lambda k: dist[k])
    base = lambda k: k.split("#")[0]
    furn = [k for k in labels if base(k) not in STRUCT_LM and base(k) not in RAW_LM_BASES]
    lm = _keep_per_base([k for k in labels if base(k) in RAW_LM_BASES], 1)
    struct = _keep_per_base([k for k in labels if base(k) in STRUCT_LM], 1)
    if lm:
        furn = furn[:4]
    return (furn + lm + struct)[:topn]
