import os
import bisect
import math
import pandas as pd
import numpy as np
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==============================================================================
# Constants
# ==============================================================================
TARGET_HZ     = 3.0
T_H           = 6    # 3 sec * 3 Hz (history)
T_F           = 15   # 5 sec * 3 Hz (future)
STRIDE_SEC    = 1.0
MAX_NEIGHBORS = 8
NB_DIM        = 13   # dx, dy, dvx, dvy, dax, day, lc_state, volume, size_bin, gate, I_x, I_y, I

NEIGHBOR_COLS_8 = [
    "leadId", "rearId",
    "leftLeadId", "leftAlongsideId", "leftRearId",
    "rightLeadId", "rightAlongsideId", "rightRearId",
]

VRU_CLASSES = {"motorcycle", "bicycle", "pedestrian"}

# Slot priority for top-N gate tie-breaking: 0 > 2 > 5 > 1 > 4 > 7 > 3 > 6
_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}

# Empirical slot weights (mean I per slot, from dataset analysis)
# Order: preceding, following, leftPreceding, leftAlongside, leftFollowing,
#        rightPreceding, rightAlongside, rightFollowing
SLOT_WEIGHTS = [0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179]

# Conditional slot weights derived from SlotWeightProbe models (mean softmax per slot).
# Used when --slot_importance_conditional is set.

# No-LC case: weights by ego lane level  (0=leftmost/fast, 1=middle, 2=rightmost/slow)
SLOT_WEIGHTS_BY_LANE_LEVEL = [
    [0.4255, 0.0336, 0.0000, 0.0000, 0.0000, 0.4574, 0.0119, 0.1190],  # ll0 leftmost
    [0.4805, 0.0002, 0.0000, 0.0000, 0.0000, 0.3803, 0.0234, 0.1839],  # ll1 middle
    [0.4784, 0.0373, 0.3344, 0.0343, 0.2050, 0.0000, 0.0000, 0.0000],  # ll2 rightmost
]

# LC-in-history case: pre-LC weights per LC group (G0-G3)
# Order: preceding, following, leftPreceding, leftAlongside, leftFollowing,
#        rightPreceding, rightAlongside, rightFollowing
# G0: leftmost→right  (lct0 leftmost→middle,    lct1 leftmost→rightmost)
# G1: middle→right    (lct3 middle→rightmost,    lct6 middle→middle(right))
# G2: middle→left     (lct2 middle→leftmost,     lct7 middle→middle(left))
# G3: rightmost→left  (lct4 rightmost→leftmost,  lct5 rightmost→middle)
SLOT_WEIGHTS_PRE_LC = [
    [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.6253, 0.2663, 0.3117],  # G0 leftmost→right
    [0.0072, 0.0263, 0.0006, 0.0000, 0.0000, 0.3970, 0.3776, 0.5494],  # G1 middle→right
    [0.0183, 0.1326, 0.6745, 0.5179, 0.2365, 0.0000, 0.0000, 0.0000],  # G2 middle→left
    [0.0381, 0.0233, 0.5755, 0.3548, 0.4799, 0.0000, 0.0000, 0.0000],  # G3 rightmost→left
]

# LC-in-history case: post-LC weights per LC group (G0-G3)
SLOT_WEIGHTS_POST_LC = [
    [0.0460, 0.3983, 0.0000, 0.0023, 0.0762, 0.2338, 0.2022, 0.3281],  # G0 leftmost→right
    [0.1036, 0.0851, 0.4832, 0.0540, 0.3810, 0.0013, 0.0000, 0.0002],  # G1 middle→right
    [0.6018, 0.3591, 0.0115, 0.0013, 0.0099, 0.1709, 0.0069, 0.0014],  # G2 middle→left
    [0.2618, 0.0000, 0.0036, 0.0000, 0.0000, 0.6545, 0.2032, 0.1449],  # G3 rightmost→left
]

# lc_type → LC group index  (G0=0, G1=1, G2=2, G3=3)
# lc_type: 0=leftmost→middle, 1=leftmost→rightmost, 2=middle→leftmost,
#           3=middle→rightmost, 4=rightmost→leftmost, 5=rightmost→middle,
#           6=middle→middle(right), 7=middle→middle(left)
_LC_TYPE_TO_GROUP = {
    0: 0, 1: 0,  # G0 leftmost→right
    3: 1, 6: 1,  # G1 middle→right
    2: 2, 7: 2,  # G2 middle→left
    4: 3, 5: 3,  # G3 rightmost→left
}

# (from_level, to_level) → lc_type
_LC_TYPE_MAP_LEVEL = {
    (0, 1): 0, (0, 2): 1,
    (1, 0): 2, (1, 2): 3,
    (2, 0): 4, (2, 1): 5,
}

# ==============================================================================
# Vehicle size bin
# ==============================================================================
_VOLUME_BIN_EDGES = [12.0, 20.0, 90.0, 150.0]  # 4 inner cuts → 5 bins (0~4)


def _volume_bin(phys_length, phys_width, vehicle_class):
    """Return (size_bin 0~4, raw volume m³) for a neighbor vehicle.

    height estimated from vehicle class and physical length:
      Car:   length < 4.5m → 1.45m,  < 5.0m → 1.70m,  >= 5.0m → 1.90m
      Truck: length < 12.0m → 2.75m, >= 12.0m → 3.75m
    """
    vehicle_class = str(vehicle_class).strip().lower()
    if vehicle_class in {"car", "van"}:
        if phys_length < 4.5:   height = 1.45
        elif phys_length < 5.0: height = 1.70
        else:                   height = 1.90
    else:
        height = 2.75 if phys_length < 12.0 else 3.75
    volume = phys_width * phys_length * height
    for i, edge in enumerate(_VOLUME_BIN_EDGES):
        if volume < edge:
            return float(i), volume
    return 4.0, volume


# ==============================================================================
# LIS binning
# ==============================================================================
LIS_BINS = {
    '3': {'cuts': [-5.8639, 4.9525],
          'vals': [-1.0, 0.0, 1.0]},
    '5': {'cuts': [-13.7033, -3.0238, 2.2735, 13.0957],
          'vals': [-2.0, -1.0, 0.0, 1.0, 2.0]},
    '7': {'cuts': [-18.7902, -8.2922, -1.9963, 1.3381, 7.3744, 18.5267],
          'vals': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]},
    '9': {'cuts': [-22.7661, -12.1209, -5.8639, -1.4829, 0.9127, 4.9525, 11.4115, 22.7702],
          'vals': [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]},
}


def _lit_to_lis(lit, lis_mode):
    cfg = LIS_BINS[lis_mode]
    return cfg['vals'][bisect.bisect_right(cfg['cuts'], lit)]


def _apply_topn_gate(nb_feats_ti, mask_ti, n):
    """Apply top-N gate in-place for one timestep.
    nb_feats_ti: (MAX_NEIGHBORS, NB_DIM) — gate at [9], I at [12]
    mask_ti:     (MAX_NEIGHBORS,) bool
    """
    K_local = nb_feats_ti.shape[0]
    valid = [k for k in range(K_local) if mask_ti[k]]
    valid.sort(key=lambda k: (-nb_feats_ti[k, 12], _TOPN_SLOT_PRIORITY.get(k, K_local)))
    selected = set(valid[:n])
    for k in valid:
        if k not in selected:
            nb_feats_ti[k, 9]  = 0.0
            nb_feats_ti[k, 10] = 0.0
            nb_feats_ti[k, 11] = 0.0
            nb_feats_ti[k, 12] = 0.0


def _lane_id_to_level(lid, dd, sorted_lids, post_flip):
    """lane_id → lane_level (0=leftmost/fast, 1=middle, 2=rightmost/slow)."""
    n = len(sorted_lids)
    if n == 0 or lid not in sorted_lids:
        return -1
    idx = sorted_lids.index(lid)
    if n == 1:
        return 1
    if post_flip or dd == 2:
        if idx == 0:     return 0
        if idx == n - 1: return 2
        return 1
    else:  # dd=1, no flip
        if idx == 0:     return 2
        if idx == n - 1: return 0
        return 1


def _ego_lc_context(ego_lane_arr, dd, lane_ids_per_dd, post_flip):
    """history window 내 ego LC 상태를 판단한다.

    Returns (lane_level, lc_frame_ti, lc_type)
      lane_level  : 0/1/2 (no-LC, ego의 t0 차선), -2 (LC in history), -1 (unknown)
      lc_frame_ti : LC가 처음 일어난 hist frame 인덱스 (None = no LC)
      lc_type     : 0-5  (-1 = no LC or unknown)
    """
    sorted_lids = lane_ids_per_dd.get(dd, [])
    lc_frame_ti = None
    lc_type = -1
    for ti in range(1, len(ego_lane_arr)):
        if ego_lane_arr[ti] != ego_lane_arr[ti - 1]:
            lc_frame_ti = ti
            from_lvl = _lane_id_to_level(int(ego_lane_arr[ti - 1]), dd, sorted_lids, post_flip)
            to_lvl   = _lane_id_to_level(int(ego_lane_arr[ti]),     dd, sorted_lids, post_flip)
            lc_type  = _LC_TYPE_MAP_LEVEL.get((from_lvl, to_lvl), -1)
            break
    if lc_frame_ti is None:
        lane_level = _lane_id_to_level(int(ego_lane_arr[-1]), dd, sorted_lids, post_flip)
    else:
        lane_level = -2
    return lane_level, lc_frame_ti, lc_type


def _get_slot_weight(ki, ti, lane_level, lc_frame_ti, lc_type):
    """slot ki / timestep ti에 대응하는 조건부 slot weight를 반환."""
    if lc_frame_ti is not None and lc_type >= 0:
        lc_group = _LC_TYPE_TO_GROUP.get(lc_type, -1)
        if lc_group < 0:
            return SLOT_WEIGHTS[ki]  # unknown lc_type fallback
        if ti < lc_frame_ti:
            return SLOT_WEIGHTS_PRE_LC[lc_group][ki]
        else:
            return SLOT_WEIGHTS_POST_LC[lc_group][ki]
    elif 0 <= lane_level <= 2:
        return SLOT_WEIGHTS_BY_LANE_LEVEL[lane_level][ki]
    else:
        return SLOT_WEIGHTS[ki]  # fallback


# ==============================================================================
# Importance parameters and functions
# ==============================================================================
# [importance_mode='lis']  — default
# I_x = exp(-(lis^2)/(2*sx^2)) * exp(-ax*lc_state) * exp(-bx*delta_lane)
# I_y = exp(-(lc_state^2)/(2*sy^2)) * exp(-ay*|lis|^py) * exp(-by*delta_lane)
IMPORTANCE_PARAMS_LIS = {
    'sx': 1.0, 'ax': 0.15, 'bx': 0.2,
    'sy': 2.0, 'ay': 0.1,  'by': 0.1, 'py': 1.5,
}

# [importance_mode='lit']  — legacy params
# I_x = exp(-(lit^2)/(2*sx^2)) * exp(-ax*lc_state) * exp(-bx*delta_lane)
# I_y = exp(-(lc_state^2)/(2*sy^2)) * exp(-ay*|lit|^1.5) * exp(-by*delta_lane)
IMPORTANCE_PARAMS_LIT = {
    'sx': 15.0, 'ax': 0.2,  'bx': 0.25,
    'sy':  2.0, 'ay': 0.01, 'by': 0.1,
}


def compute_importance_lis(lis, delta_lane, lc_state):
    p = IMPORTANCE_PARAMS_LIS
    ix = (np.exp(-(lis ** 2) / (2.0 * p['sx'] ** 2))
          * np.exp(-p['ax'] * lc_state)
          * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state ** 2) / (2.0 * p['sy'] ** 2))
          * np.exp(-p['ay'] * (abs(lis) ** p['py']))
          * np.exp(-p['by'] * delta_lane))
    return float(ix), float(iy), float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))


def compute_importance_lit(lit, delta_lane, lc_state):
    p = IMPORTANCE_PARAMS_LIT
    ix = (np.exp(-(lit ** 2) / (2.0 * p['sx'] ** 2))
          * np.exp(-p['ax'] * lc_state)
          * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state ** 2) / (2.0 * p['sy'] ** 2))
          * np.exp(-p['ay'] * (abs(lit) ** 1.5))
          * np.exp(-p['by'] * delta_lane))
    return float(ix), float(iy), float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))


# ==============================================================================
# Feature mode index map
# NB_DIM=13: [0]dx [1]dy [2]dvx [3]dvy [4]dax [5]day
#            [6]lc_state [7]volume [8]size_bin [9]gate [10]I_x [11]I_y [12]I
# ==============================================================================
EXTRA_FEATURE_MAP = {
    'baseline':   [0, 1, 2, 3, 4, 5],        # dx, dy, dvx, dvy, dax, day
    'importance': [0, 1, 2, 3, 4, 5, 12],    # dx, dy, dvx, dvy, dax, day, I
    'sy':         [0, 1, 2, 3, 4, 5, 6],     # dx, dy, dvx, dvy, dax, day, lc_state
    'iy':         [0, 1, 2, 3, 4, 5, 11],    # dx, dy, dvx, dvy, dax, day, I_y
    'dimI':       [0, 1, 2, 3, 4, 5, 8, 11],              # dx, dy, dvx, dvy, dax, day, 
    'dim':        [0, 1, 2, 3, 4, 5, 8]
}


def get_num_channels(feature_mode):
    """총 채널 수: ego_vel(2) + nb_feats(N) + is_ego(1)"""
    return 2 + len(EXTRA_FEATURE_MAP[feature_mode]) + 1


# ==============================================================================
# Utilities
# ==============================================================================

def parse_semicolon_floats(s):
    if not isinstance(s, str):
        return []
    return [float(p) for p in s.strip().split(";") if p.strip()]


def process_wrapper(args_tuple):
    rid, raw_path, args = args_tuple
    samples = process_recording(rid, raw_path, args)
    return rid, samples


def _first_numeric_series(df, col, default, dtype):
    if col not in df.columns:
        return np.full(len(df), default, dtype=dtype)
    s = df[col].astype(str).str.strip().str.split(";").str[0]
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(dtype).to_numpy()


def _first_available_numeric_series(df, cols, default, dtype):
    for col in cols:
        if col in df.columns:
            return _first_numeric_series(df, col, default, dtype)
    return np.full(len(df), default, dtype=dtype)


def _fill_derivatives_from_position(per_vid_rows, frame_arr, pos_arr, out_arr, frame_rate):
    for idxs in per_vid_rows.values():
        if len(idxs) < 2:
            continue
        t = frame_arr[idxs].astype(np.float32) / float(frame_rate)
        p = pos_arr[idxs].astype(np.float32)
        out_arr[idxs] = np.gradient(p, t).astype(np.float32)


def _rot2d(vx, vy, h_rad):
    c, s = math.cos(h_rad), math.sin(h_rad)
    return c * vx + s * vy, -s * vx + c * vy


def _norm_pos(gx, gy, ref_x, ref_y, ref_hdg_rad):
    return _rot2d(gx - ref_x, gy - ref_y, ref_hdg_rad)


def _local_to_norm_frame(lon, lat, veh_hdg_rad, ref_hdg_rad):
    delta = veh_hdg_rad - ref_hdg_rad
    c, s = math.cos(delta), math.sin(delta)
    return c * lon - s * lat, s * lon + c * lat


def _rel_local_in_ego_frame(nb_lon, nb_lat, nb_hdg, ego_lon, ego_lat, ego_hdg):
    nb_vx = nb_lon * math.cos(nb_hdg) - nb_lat * math.sin(nb_hdg)
    nb_vy = nb_lon * math.sin(nb_hdg) + nb_lat * math.cos(nb_hdg)
    ego_vx = ego_lon * math.cos(ego_hdg) - ego_lat * math.sin(ego_hdg)
    ego_vy = ego_lon * math.sin(ego_hdg) + ego_lat * math.cos(ego_hdg)
    return _rot2d(nb_vx - ego_vx, nb_vy - ego_vy, ego_hdg)


def balanced_recording_split(ds_counts, ratios=(0.7, 0.1, 0.2), seed=42):
    rng = np.random.default_rng(seed)
    total_samples = sum(ds_counts.values())
    targets = [total_samples * r for r in ratios]
    items = sorted(ds_counts.items(), key=lambda x: x[1], reverse=True)
    rng.shuffle(items)

    splits = {"train": [], "val": [], "test": []}
    sums   = {"train": 0,  "val": 0,  "test": 0}
    keys   = ["train", "val", "test"]
    for rec_id, cnt in items:
        deficits = {k: (targets[i] - sums[k]) for i, k in enumerate(keys)}
        best = max(deficits.items(), key=lambda kv: kv[1])[0]
        splits[best].append(rec_id)
        sums[best] += cnt
    return splits


# ==============================================================================
# Per-recording processing  (raw CSV -> list of sample dicts)
# ==============================================================================

def process_recording(rec_id, raw_dir, args):
    tracks = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv", low_memory=False)
    tmeta  = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    rmeta  = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    tracks.columns = [c.strip() for c in tracks.columns]
    tmeta.columns  = [c.strip() for c in tmeta.columns]
    rmeta.columns  = [c.strip() for c in rmeta.columns]

    frame_rate = float(rmeta.loc[0, "frameRate"]) if "frameRate" in rmeta.columns else 25.0
    step   = max(1, int(round(frame_rate / TARGET_HZ)))
    stride = max(1, int(round(STRIDE_SEC * TARGET_HZ)))

    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns: tracks[c] = -1
    for c in ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"]:
        if c not in tracks.columns: tracks[c] = 0.0
    if "laneletId" not in tracks.columns: tracks["laneletId"] = -1

    meta_id_col = "trackId" if "trackId" in tmeta.columns else "id"
    vid_to_dd    = {int(v): 2 for v in tracks["trackId"].astype(int).unique()}
    vid_to_class = {
        int(k): str(v).strip().lower()
        for k, v in zip(tmeta[meta_id_col].astype(int), tmeta["class"].astype(str))
    } if "class" in tmeta.columns else {}

    tracks = tracks.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)
    vid_arr   = tracks["trackId"].astype(np.int32).to_numpy()
    frame_arr = tracks["frame"].astype(np.int32).to_numpy()
    x_arr     = _first_available_numeric_series(tracks, ["xCenter", "x"], 0.0, np.float32).copy()
    y_arr     = _first_available_numeric_series(tracks, ["yCenter", "y"], 0.0, np.float32).copy()

    # GRIP baseline channels must live in the same coordinate frame as target
    # (xCenter/yCenter deltas). Prefer exiD's global kinematics when present;
    # otherwise derive them from the global positions below.
    has_global_velocity = "xVelocity" in tracks.columns and "yVelocity" in tracks.columns
    has_global_accel = "xAcceleration" in tracks.columns and "yAcceleration" in tracks.columns
    xv_arr    = _first_available_numeric_series(tracks, ["xVelocity"], 0.0, np.float32)
    yv_arr    = _first_available_numeric_series(tracks, ["yVelocity"], 0.0, np.float32)
    xa_arr    = _first_available_numeric_series(tracks, ["xAcceleration"], 0.0, np.float32)
    ya_arr    = _first_available_numeric_series(tracks, ["yAcceleration"], 0.0, np.float32)
    lon_v_arr = _first_available_numeric_series(tracks, ["lonVelocity", "xVelocity"], 0.0, np.float32)
    lat_v_arr = _first_available_numeric_series(tracks, ["latVelocity", "yVelocity"], 0.0, np.float32)
    lon_a_arr = _first_available_numeric_series(tracks, ["lonAcceleration", "xAcceleration"], 0.0, np.float32)
    lat_a_arr = _first_available_numeric_series(tracks, ["latAcceleration", "yAcceleration"], 0.0, np.float32)

    # Lane-change heuristics are lateral-lane based, so keep a separate lateral
    # velocity stream instead of reusing global yVelocity on intersection scenes.
    lat_lc_v_arr = lat_v_arr
    lane_arr  = tracks["laneletId"].fillna(-1).astype(np.int32).to_numpy().copy()
    dd_arr   = np.array([vid_to_dd.get(int(v), 0) for v in vid_arr], np.int8)
    width_arr  = tracks["width"].astype(np.float32).to_numpy() if "width" in tracks.columns else np.zeros(len(tracks), np.float32)
    length_arr = tracks["length"].astype(np.float32).to_numpy() if "length" in tracks.columns else np.zeros(len(tracks), np.float32)
    heading_arr = np.deg2rad(
        _first_available_numeric_series(tracks, ["heading"], 0.0, np.float32)
    ).astype(np.float32)

    if "latLaneCenterOffset" in tracks.columns:
        lat_lane_offset_arr = _first_numeric_series(tracks, "latLaneCenterOffset", 0.0, np.float32)
    else:
        lat_lane_offset_arr = np.zeros(len(tracks), np.float32)
    if "laneWidth" in tracks.columns:
        lat_lane_width_arr = _first_numeric_series(tracks, "laneWidth", 3.5, np.float32)
    else:
        lat_lane_width_arr = np.full(len(tracks), 3.5, np.float32)

    vid_to_w, vid_to_h = {}, {}
    for v, idxs in tracks.groupby("trackId").indices.items():
        r0 = int(np.array(list(idxs), np.int32)[0])
        vid_to_w[int(v)] = float(length_arr[r0])
        vid_to_h[int(v)] = float(width_arr[r0])

    # Build per-vehicle sorted row-index arrays and frame->row dicts
    per_vid_rows: dict = {}
    per_vid_frame_to_row: dict = {}
    for v, idxs in tracks.groupby("trackId").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame_arr[idxs])]
        per_vid_rows[int(v)] = idxs
        per_vid_frame_to_row[int(v)] = {int(frame_arr[r]): int(r) for r in idxs}

    per_vid_frame_to_hdg = {
        int(v): {int(frame_arr[r]): float(heading_arr[r]) for r in idxs}
        for v, idxs in per_vid_rows.items()
    }

    if not has_global_velocity:
        _fill_derivatives_from_position(per_vid_rows, frame_arr, x_arr, xv_arr, frame_rate)
        _fill_derivatives_from_position(per_vid_rows, frame_arr, y_arr, yv_arr, frame_rate)
    if not has_global_accel:
        _fill_derivatives_from_position(per_vid_rows, frame_arr, xv_arr, xa_arr, frame_rate)
        _fill_derivatives_from_position(per_vid_rows, frame_arr, yv_arr, ya_arr, frame_rate)

    nb_ids_all = np.stack([_first_numeric_series(tracks, c, -1, np.int32)
                           for c in NEIGHBOR_COLS_8], axis=1)

    # Feature selection setup
    selected_indices = EXTRA_FEATURE_MAP[args.feature_mode]
    nb_feat_dim = len(selected_indices)
    num_c       = get_num_channels(args.feature_mode)
    ego_vel_ch  = slice(0, 2)
    nb_feat_ch  = slice(2, 2 + nb_feat_dim)
    is_ego_ch   = 2 + nb_feat_dim

    # ── per-dd sorted lane IDs (for conditional slot weights) ─────────────────
    lane_ids_per_dd: dict = {}
    if args.slot_importance_conditional:
        for dd_val in [1, 2]:
            lids = sorted(set(int(x) for x in lane_arr[dd_arr == dd_val] if x > 0))
            lane_ids_per_dd[dd_val] = lids

    samples = []

    for v, idxs in per_vid_rows.items():
        if args.drop_vru and vid_to_class.get(int(v), "") in VRU_CLASSES:
            continue

        frs = frame_arr[idxs]
        if len(frs) < (T_H + T_F) * step:
            continue
        fr_set    = set(map(int, frs.tolist()))
        start_min = int(frs[0]  + (T_H - 1) * step)
        end_max   = int(frs[-1] - T_F * step)
        if start_min > end_max:
            continue

        t0_frame = start_min
        while t0_frame <= end_max:
            hist_frames = [t0_frame - (T_H - 1 - i) * step for i in range(T_H)]
            fut_frames  = [t0_frame + (i + 1) * step for i in range(T_F)]

            if (not all(hf in fr_set for hf in hist_frames) or
                    not all(ff in fr_set for ff in fut_frames)):
                t0_frame += stride * step
                continue

            ego_rows = [per_vid_frame_to_row[v][hf] for hf in hist_frames]
            fut_rows = [per_vid_frame_to_row[v][ff] for ff in fut_frames]

            ex  = x_arr[ego_rows]
            ey  = y_arr[ego_rows]
            exv = xv_arr[ego_rows]
            eyv = yv_arr[ego_rows]
            exa = xa_arr[ego_rows]
            eya = ya_arr[ego_rows]
            ego_lanes = lane_arr[ego_rows].astype(np.int32)
            len_ego   = float(vid_to_w.get(v, 0.0))
            ego_hdg_arr = np.array(
                [per_vid_frame_to_hdg.get(int(v), {}).get(int(hf), 0.0) for hf in hist_frames],
                np.float32,
            )
            ref_hdg = float(ego_hdg_arr[-1])

            # ── conditional slot weight context ───────────────────────────────
            _lc_lane_lv, _lc_frame_ti, _lc_type = -1, None, -1
            if args.slot_importance_conditional and args.slot_importance_alpha > 0.0:
                _ego_dd = vid_to_dd.get(v, 2)
                _lc_lane_lv, _lc_frame_ti, _lc_type = _ego_lc_context(
                    ego_lanes, _ego_dd, lane_ids_per_dd, args.normalize_flip
                )

            # GRIP tensor: (1 + MAX_NEIGHBORS, T_H, num_c)
            tensor = np.zeros((1 + MAX_NEIGHBORS, T_H, num_c), dtype=np.float32)
            adj    = np.eye(1 + MAX_NEIGHBORS, dtype=np.float32)

            # Ego node: ch 0,1 = vx,vy  /  is_ego = 1
            norm_center = np.array([ex[-1], ey[-1]], np.float32)
            if args.normalize_heading:
                ego_vel = np.array([
                    _local_to_norm_frame(
                        float(lon_v_arr[r]), float(lat_v_arr[r]),
                        float(ego_hdg_arr[ti]), ref_hdg,
                    )
                    for ti, r in enumerate(ego_rows)
                ], dtype=np.float32)
            else:
                ego_vel = np.stack([exv, eyv], axis=1)
            tensor[0, :, ego_vel_ch] = ego_vel
            tensor[0, :, is_ego_ch]  = 1.0

            # Neighbor IDs determined at obs_frame (last history frame)
            ids8_obs = nb_ids_all[ego_rows[-1]]

            # Identify valid neighbors (must be present at ALL T_H history frames)
            valid_nbs = {}
            for ki in range(MAX_NEIGHBORS):
                nid = int(ids8_obs[ki])
                if nid <= 0: continue
                if args.drop_vru and vid_to_class.get(nid, "") in VRU_CLASSES:
                    continue
                rm = per_vid_frame_to_row.get(nid)
                if rm is None: continue
                nb_rows_ki = [rm.get(int(hf)) for hf in hist_frames]
                if any(r is None for r in nb_rows_ki): continue
                valid_nbs[ki] = (nid, nb_rows_ki)

            # All-neighbor feature matrix: (MAX_NEIGHBORS, T_H, NB_DIM)
            nb_all_feats = np.zeros((MAX_NEIGHBORS, T_H, NB_DIM), np.float32)
            nb_mask_mat  = np.zeros((MAX_NEIGHBORS, T_H), bool)

            for ti, hf in enumerate(hist_frames):
                for ki, (nid, nb_rows_ki) in valid_nbs.items():
                    nr = nb_rows_ki[ti]
                    if args.normalize_heading:
                        ego_hdg_ti = float(ego_hdg_arr[ti])
                        nb_hdg = float(
                            per_vid_frame_to_hdg.get(nid, {}).get(int(hf), ego_hdg_ti)
                        )
                        dx, dy = _norm_pos(
                            float(x_arr[nr]), float(y_arr[nr]),
                            float(ex[ti]), float(ey[ti]), ego_hdg_ti,
                        )
                        dvx, dvy = _rel_local_in_ego_frame(
                            float(lon_v_arr[nr]), float(lat_v_arr[nr]), nb_hdg,
                            float(lon_v_arr[ego_rows[ti]]), float(lat_v_arr[ego_rows[ti]]),
                            ego_hdg_ti,
                        )
                        dax, day = _rel_local_in_ego_frame(
                            float(lon_a_arr[nr]), float(lat_a_arr[nr]), nb_hdg,
                            float(lon_a_arr[ego_rows[ti]]), float(lat_a_arr[ego_rows[ti]]),
                            ego_hdg_ti,
                        )
                    else:
                        dx  = float(x_arr[nr]  - ex[ti])
                        dy  = float(y_arr[nr]  - ey[ti])
                        dvx = float(xv_arr[nr] - exv[ti])
                        dvy = float(yv_arr[nr] - eyv[ti])
                        dax = float(xa_arr[nr] - exa[ti])
                        day = float(ya_arr[nr] - eya[ti])

                    # ── lc_state ─────────────────────────────────────────────
                    if args.lc_version == "v1":
                        vyn = float(lat_lc_v_arr[nr])
                        if ki < 2:
                            lc_state = 1.0
                        elif abs(vyn) < args.vy_eps:
                            lc_state = 1.0
                        elif ki < 5:
                            lc_state = 0.0 if vyn < 0 else 2.0
                        else:
                            lc_state = 0.0 if vyn > 0 else 2.0
                    elif args.lc_version == "v2":
                        dlatv = float(lat_lc_v_arr[nr] - lat_lc_v_arr[ego_rows[ti]])
                        abs_dvy = abs(dlatv)
                        if ki < 2 and abs(dy) < args.dy_same:
                            lc_state = 2.0 if abs_dvy > args.dvy_eps_same else 1.0
                        elif ki >= 2:
                            lc_state = (0.0 if dy * dlatv < 0 else 2.0) \
                                if abs_dvy > args.dvy_eps_cross else 1.0
                        else:
                            lc_state = 0.0 if dy * dlatv < 0 else 2.0
                    elif args.lc_version == "v3":
                        nb_lat_v = float(lat_lc_v_arr[nr])
                        nb_lco   = float(lat_lane_offset_arr[nr])
                        if ki < 2:   # same lane (lead / rear)
                            if (nb_lco < -1.0 and nb_lat_v > 0.0) or \
                               (nb_lco >  1.0 and nb_lat_v < 0.0):
                                lc_state = 0.0
                            elif (nb_lco < -1.0 and nb_lat_v < 0.0) or \
                                 (nb_lco >  1.0 and nb_lat_v > 0.0) or \
                                 abs(nb_lat_v) > 0.029:
                                lc_state = 2.0
                            else:
                                lc_state = 1.0
                        elif ki < 5:  # left lane (slots 2,3,4)
                            if   nb_lat_v < -0.029: lc_state = 0.0
                            elif nb_lat_v >  0.029: lc_state = 2.0
                            else:                   lc_state = 1.0
                        else:         # right lane (slots 5,6,7)
                            if   nb_lat_v < -0.029: lc_state = 2.0
                            elif nb_lat_v >  0.029: lc_state = 0.0
                            else:                   lc_state = 1.0
                    else:  # v4: lco_norm 기반 경계 판단 + slot별 방향 결정
                        nb_lat_v    = float(lat_lc_v_arr[nr])
                        nb_lco      = float(lat_lane_offset_arr[nr])
                        nb_lw       = float(lat_lane_width_arr[nr])
                        nb_lco_norm = nb_lco / (nb_lw * 0.5) if nb_lw > 0.5 else 0.0
                        if abs(nb_lco_norm) <= 0.5:
                            lc_state = 1.0
                        elif ki < 2:   # same lane
                            lc_state = 0.0 if nb_lco_norm * nb_lat_v < 0 else 2.0
                        elif ki < 5:   # left lane (slots 2,3,4)
                            lc_state = 0.0 if nb_lat_v < 0 else 2.0
                        else:          # right lane (slots 5,6,7)
                            lc_state = 0.0 if nb_lat_v > 0 else 2.0

                    # ── LIT: gap-based (bumper-to-bumper) ─────────────────────
                    len_nb   = float(vid_to_w.get(nid, 0.0))
                    half_sum = 0.5 * (len_ego + len_nb)
                    if dx >= 0:  # nb ahead: gap = x_rear_nb - x_front_ego
                        gap        = abs(dx - half_sum)
                        denom_base = dvx
                    else:        # nb behind: gap = x_rear_ego - x_front_nb
                        gap        = abs(-dx - half_sum)
                        denom_base = -dvx
                    lit = gap / (denom_base + (args.eps_gate if denom_base >= 0 else -args.eps_gate))
                    lis = _lit_to_lis(lit, args.lis_mode)

                    nb_class  = vid_to_class.get(nid, "Car")
                    nb_phys_l = vid_to_w.get(nid, 0.0)   # CSV width = physical length
                    nb_phys_w = vid_to_h.get(nid, 0.0)   # CSV height = physical width
                    size_bin, nb_volume = _volume_bin(nb_phys_l, nb_phys_w, nb_class)

                    delta_lane = float(abs(int(lane_arr[nr]) - int(ego_lanes[ti])))

                    # ── importance ────────────────────────────────────────────
                    if args.importance_mode == 'lit':
                        ix, iy, i_total = compute_importance_lit(lit, delta_lane, lc_state)
                    else:  # 'lis' (default)
                        ix, iy, i_total = compute_importance_lis(lis, delta_lane, lc_state)

                    # ── slot importance boost: I_new = min(I * (1 + alpha * w_slot), 1.0) ──
                    if args.slot_importance_alpha > 0.0:
                        if args.slot_importance_conditional:
                            w_slot = _get_slot_weight(ki, ti, _lc_lane_lv, _lc_frame_ti, _lc_type)
                        else:
                            w_slot = SLOT_WEIGHTS[ki]
                        i_total = min(
                            i_total * (1.0 + args.slot_importance_alpha * w_slot),
                            1.0,
                        )

                    # ── gate ──────────────────────────────────────────────────
                    gate    = 1.0 if (args.gate_theta <= 0.0 or i_total >= args.gate_theta) else 0.0

                    nb_all_feats[ki, ti] = [dx, dy, dvx, dvy, dax, day,
                                            lc_state, nb_volume, size_bin, gate,
                                            ix * gate, iy * gate, i_total * gate]
                    nb_mask_mat[ki, ti]  = True

                # Apply gate_topn per timestep after all slots are filled
                if args.gate_topn > 0:
                    _apply_topn_gate(nb_all_feats[:, ti, :], nb_mask_mat[:, ti], args.gate_topn)

            # Fill tensor with selected features
            for ki in valid_nbs:
                tensor[ki + 1, :, nb_feat_ch] = nb_all_feats[ki, :, :][:, selected_indices]
                adj[0, ki + 1] = adj[ki + 1, 0] = 1.0

            # Target: future (x, y) relative to last observed ego position.
            # With heading normalization, rotate into the last-observed ego frame.
            if args.normalize_heading:
                target = np.array([
                    _norm_pos(
                        float(x_arr[r]), float(y_arr[r]),
                        float(norm_center[0]), float(norm_center[1]), ref_hdg,
                    )
                    for r in fut_rows
                ], dtype=np.float32)
            else:
                fut_xy = np.stack([x_arr[fut_rows], y_arr[fut_rows]], axis=1)  # (T_F, 2)
                target = fut_xy - norm_center

            samples.append({"input": tensor, "adj": adj, "target": target,
                            "recordingId": int(rec_id), "trackId": int(v),
                            "t0_frame": int(t0_frame)})
            t0_frame += stride * step

    return samples


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         type=str,   default="exiD/raw")
    parser.add_argument("--out_dir",         type=str,   default="exiD")
    parser.add_argument("--feature_mode",    type=str,   default="baseline",
                        choices=['baseline', 'importance', 'sy', 'iy', 'dimI'])
    parser.add_argument("--normalize_flip",  action="store_true", default=False,
                        help="Kept for CLI compatibility; exiD coordinates are not highD-flipped.")
    parser.add_argument("--normalize_heading", "--normalize-heading",
                        action="store_true", default=True,
                        help="Normalize exiD coordinates into the ego-heading local frame.")
    parser.add_argument("--no_normalize_heading", "--no-normalize-heading",
                        action="store_false",
                        dest="normalize_heading",
                        help="Keep raw global x/y axes instead of ego-heading normalization.")
    parser.add_argument("--drop_vru", "--drop-vru",
                        action="store_true", default=True,
                        help="Drop ego samples and neighbors whose class is motorcycle/bicycle/pedestrian.")
    parser.add_argument("--keep_vru", "--keep-vru",
                        action="store_true", default=False,
                        help="Disable --drop_vru and keep VRU tracks.")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--eps_gate",        type=float, default=1.0,
                        help="eps for LIT denominator (makes gap dominant over dvx)")
    parser.add_argument("--gate_theta",      type=float, default=0.0,
                        help="importance threshold for gate (0.0 = disabled)")
    parser.add_argument("--gate_topn",            type=int,   default=0,
                        help="keep top-N neighbors by I per timestep (0 = disabled)")
    parser.add_argument("--slot_importance_alpha", type=float, default=0.0,
                        help="slot importance boost: I_new = min(I*(1+alpha*w_slot),1.0) (0.0 = disabled)")
    parser.add_argument("--slot_importance_conditional", action="store_true", default=False,
                        help="use lane-level/pre-LC/post-LC conditional slot weights")
    parser.add_argument("--lc_version",      type=str,   default="v3",
                        choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--lis_mode",        type=str,   default="3",
                        choices=list(LIS_BINS.keys()))
    parser.add_argument("--importance_mode", type=str,   default="lis",
                        choices=["lis", "lit"])
    # v1 lc_state params
    parser.add_argument("--vy_eps",          type=float, default=0.27)
    # v2 lc_state params
    parser.add_argument("--dvy_eps_cross",   type=float, default=0.26)
    parser.add_argument("--dvy_eps_same",    type=float, default=1.03)
    parser.add_argument("--dy_same",         type=float, default=1.5)
    args = parser.parse_args()
    if args.keep_vru:
        args.drop_vru = False

    raw_path = Path(args.raw_dir)
    rec_ids  = sorted(set([
        f.name.split("_")[0] for f in raw_path.glob("*_tracks.csv")
    ]))

    nb_feat_dim = len(EXTRA_FEATURE_MAP[args.feature_mode])
    num_c       = get_num_channels(args.feature_mode)
    print(f"Found {len(rec_ids)} recordings")
    print(f"Feature mode     : {args.feature_mode}")
    print(f"drop_vru         : {args.drop_vru}")
    print(f"normalize_heading: {args.normalize_heading}")
    print(f"lc_version       : {args.lc_version}")
    print(f"importance_mode  : {args.importance_mode}"
          + (f"  lis_mode={args.lis_mode}" if args.importance_mode == 'lis' else ""))
    print(f"gate_theta={args.gate_theta}  gate_topn={args.gate_topn}")
    print(f"Target Hz        : {TARGET_HZ}  |  T_H={T_H}  |  T_F={T_F}")
    print(f"Channel layout   : [ego_vx, ego_vy | {nb_feat_dim} nb_feats | is_ego]  ->  total {num_c}ch")
    print(f"Using {cpu_count()} CPU cores")

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_wrapper,
                      [(rid, raw_path, args) for rid in rec_ids]),
            total=len(rec_ids),
            desc="Preprocessing"
        ))

    all_rec_samples = {rid: s for rid, s in results if s}

    rec_counts = {rid: len(s) for rid, s in all_rec_samples.items()}
    splits     = balanced_recording_split(rec_counts, seed=args.seed)

    out_dir = Path(args.out_dir) / args.feature_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rec_ids in splits.items():
        split_data = [s for rid in split_rec_ids for s in all_rec_samples.get(rid, [])]
        if not split_data:
            print(f"-> {split_name}: empty, skipping")
            continue
        with h5py.File(out_dir / f"{split_name}.h5", "w") as f:
            f.attrs["drop_vru"] = bool(args.drop_vru)
            f.attrs["normalize_heading"] = bool(args.normalize_heading)
            f.attrs["feature_mode"] = args.feature_mode
            f.create_dataset("input",  data=np.array([s["input"]  for s in split_data]), compression="gzip")
            f.create_dataset("adj",    data=np.array([s["adj"]    for s in split_data]), compression="gzip")
            f.create_dataset("target", data=np.array([s["target"] for s in split_data]), compression="gzip")
            f.create_dataset("meta_recordingId", data=np.array([s["recordingId"] for s in split_data], dtype=np.int32))
            f.create_dataset("meta_trackId",     data=np.array([s["trackId"]     for s in split_data], dtype=np.int32))
            f.create_dataset("meta_t0_frame",    data=np.array([s["t0_frame"]    for s in split_data], dtype=np.int32))
        print(f"-> {split_name}.h5 saved ({len(split_data)} samples, input shape: {split_data[0]['input'].shape})")


if __name__ == "__main__":
    main()
