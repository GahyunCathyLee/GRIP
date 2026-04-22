#!/usr/bin/env python3
"""
GRIP/NGSIM/preprocess.py  —  NGSIM preprocessing for GRIP model

Reads a single combined NGSIM CSV (e.g. ../neighformer/data/NGSIM/NGSIM_data.csv),
applies anomaly filtering, and writes HDF5 files in the same format as
GRIP/highD/preprocess.py.

Only freeway locations (i-80, us-101) are processed.

Output HDF5 datasets per split (train / val / test):
  input  : (N, 1+MAX_NEIGHBORS, T_H, num_c)   float32
  adj    : (N, 1+MAX_NEIGHBORS, 1+MAX_NEIGHBORS) float32
  target : (N, T_F, 2)                          float32  — future (x,y) relative to t0
  meta_recordingId, meta_trackId, meta_t0_frame : (N,) int32

Feature channel layout (num_c = 2 + nb_feat_dim + 1):
  ch 0,1        : vx, vy  (ego) | 0, 0  (neighbor)
  ch 2..2+D-1   : 0  (ego) | selected nb features (neighbor)
  ch 2+D        : is_ego flag  (1 for ego node, 0 for neighbors)
"""
import argparse
import bisect
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── NGSIM raw column names ───────────────────────────────────────────────────
NC_ID    = "Vehicle_ID"
NC_FRAME = "Frame_ID"
NC_X     = "Local_X"    # lateral,      feet, front-center from left edge
NC_Y     = "Local_Y"    # longitudinal, feet, front-center from highway entry
NC_LEN   = "v_length"   # feet  (lowercase 'l' in this combined dataset)
NC_WID   = "v_Width"    # feet
NC_CLASS = "v_Class"    # 1=motorcycle, 2=auto, 3=truck
NC_VEL   = "v_Vel"      # feet/s  (forward speed magnitude)
NC_ACC   = "v_Acc"      # feet/s²
NC_LANE  = "Lane_ID"
NC_LOC   = "Location"   # "i-80" | "us-101" | "lankershim" | "peachtree"

FT2M = 0.3048
FPS  = 10.0
DT   = 1.0 / FPS

FREEWAY_LOCATIONS = {"i-80", "us-101"}
MAX_LANES = {"i-80": 6, "us-101": 5}

# ─── GRIP constants ───────────────────────────────────────────────────────────
TARGET_HZ     = 3.0
T_H           = 6     # 2 sec history
T_F           = 15    # 5 sec future
STRIDE_SEC    = 1.0
MAX_NEIGHBORS = 8
NB_DIM        = 13    # dx dy dvx dvy dax day lc_state volume size_bin gate I_x I_y I

NEIGHBOR_COLS_8 = [
    "precedingId", "followingId",
    "leftPrecedingId",  "leftAlongsideId",  "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
]

_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}

SLOT_WEIGHTS = [0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179]

SLOT_WEIGHTS_BY_LANE_LEVEL = [
    [0.4255, 0.0336, 0.0000, 0.0000, 0.0000, 0.4574, 0.0119, 0.1190],
    [0.4805, 0.0002, 0.0000, 0.0000, 0.0000, 0.3803, 0.0234, 0.1839],
    [0.4784, 0.0373, 0.3344, 0.0343, 0.2050, 0.0000, 0.0000, 0.0000],
]
SLOT_WEIGHTS_PRE_LC = [
    [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.6253, 0.2663, 0.3117],
    [0.0072, 0.0263, 0.0006, 0.0000, 0.0000, 0.3970, 0.3776, 0.5494],
    [0.0183, 0.1326, 0.6745, 0.5179, 0.2365, 0.0000, 0.0000, 0.0000],
    [0.0381, 0.0233, 0.5755, 0.3548, 0.4799, 0.0000, 0.0000, 0.0000],
]
SLOT_WEIGHTS_POST_LC = [
    [0.0460, 0.3983, 0.0000, 0.0023, 0.0762, 0.2338, 0.2022, 0.3281],
    [0.1036, 0.0851, 0.4832, 0.0540, 0.3810, 0.0013, 0.0000, 0.0002],
    [0.6018, 0.3591, 0.0115, 0.0013, 0.0099, 0.1709, 0.0069, 0.0014],
    [0.2618, 0.0000, 0.0036, 0.0000, 0.0000, 0.6545, 0.2032, 0.1449],
]
_LC_TYPE_TO_GROUP = {0: 0, 1: 0, 3: 1, 6: 1, 2: 2, 7: 2, 4: 3, 5: 3}
_LC_TYPE_MAP_LEVEL = {
    (0, 1): 0, (0, 2): 1, (1, 0): 2,
    (1, 2): 3, (2, 0): 4, (2, 1): 5,
}

_VOLUME_BIN_EDGES = [12.0, 20.0, 90.0, 150.0]

LIS_BINS = {
    '3': {'cuts': [-5.8639, 4.9525],          'vals': [-1., 0., 1.]},
    '5': {'cuts': [-13.7033, -3.0238, 2.2735, 13.0957],
          'vals': [-2., -1., 0., 1., 2.]},
    '7': {'cuts': [-18.7902, -8.2922, -1.9963, 1.3381, 7.3744, 18.5267],
          'vals': [-3., -2., -1., 0., 1., 2., 3.]},
    '9': {'cuts': [-22.7661, -12.1209, -5.8639, -1.4829,
                    0.9127, 4.9525, 11.4115, 22.7702],
          'vals': [-4., -3., -2., -1., 0., 1., 2., 3., 4.]},
}

IMPORTANCE_PARAMS_LIS = {
    'sx': 1.0, 'ax': 0.15, 'bx': 0.2,
    'sy': 2.0, 'ay': 0.1,  'by': 0.1, 'py': 1.5,
}
IMPORTANCE_PARAMS_LIT = {
    'sx': 15.0, 'ax': 0.2, 'bx': 0.25,
    'sy':  2.0, 'ay': 0.01, 'by': 0.1,
}

EXTRA_FEATURE_MAP = {
    'baseline':   [0, 1, 2, 3, 4, 5],
    'importance': [0, 1, 2, 3, 4, 5, 12],
    'sy':         [0, 1, 2, 3, 4, 5, 6],
    'iy':         [0, 1, 2, 3, 4, 5, 11],
    'dimI':       [0, 1, 2, 3, 4, 5, 8, 11],
    'dim':        [0, 1, 2, 3, 4, 5, 8],
}


def get_num_channels(feature_mode):
    return 2 + len(EXTRA_FEATURE_MAP[feature_mode]) + 1


# ─── Utilities ────────────────────────────────────────────────────────────────

def _volume_bin(phys_length, phys_width, vehicle_class):
    if vehicle_class == "Car":
        height = 1.45 if phys_length < 4.5 else (1.70 if phys_length < 5.0 else 1.90)
    else:
        height = 2.75 if phys_length < 12.0 else 3.75
    volume = phys_width * phys_length * height
    for i, edge in enumerate(_VOLUME_BIN_EDGES):
        if volume < edge:
            return float(i), volume
    return 4.0, volume


def _lit_to_lis(lit, lis_mode):
    c = LIS_BINS[lis_mode]
    return c['vals'][bisect.bisect_right(c['cuts'], lit)]


def _apply_topn_gate(nb_feats_ti, mask_ti, n):
    K = nb_feats_ti.shape[0]
    valid = [k for k in range(K) if mask_ti[k]]
    valid.sort(key=lambda k: (-nb_feats_ti[k, 12], _TOPN_SLOT_PRIORITY.get(k, K)))
    selected = set(valid[:n])
    for k in valid:
        if k not in selected:
            nb_feats_ti[k, 9:13] = 0.0


def _lane_id_to_level(lid, sorted_lids):
    """NGSIM single-direction (dd=2): lane 1=leftmost/fast (level 0)."""
    n = len(sorted_lids)
    if n == 0 or lid not in sorted_lids:
        return -1
    idx = sorted_lids.index(lid)
    if n == 1: return 1
    if idx == 0:     return 0
    if idx == n - 1: return 2
    return 1


def _ego_lc_context(ego_lane_arr, sorted_lids):
    lc_frame_ti, lc_type = None, -1
    for ti in range(1, len(ego_lane_arr)):
        if ego_lane_arr[ti] != ego_lane_arr[ti - 1]:
            lc_frame_ti = ti
            from_lvl = _lane_id_to_level(int(ego_lane_arr[ti - 1]), sorted_lids)
            to_lvl   = _lane_id_to_level(int(ego_lane_arr[ti]),     sorted_lids)
            lc_type  = _LC_TYPE_MAP_LEVEL.get((from_lvl, to_lvl), -1)
            break
    lane_level = (-2 if lc_frame_ti is not None
                  else _lane_id_to_level(int(ego_lane_arr[-1]), sorted_lids))
    return lane_level, lc_frame_ti, lc_type


def _get_slot_weight(ki, ti, lane_level, lc_frame_ti, lc_type):
    if lc_frame_ti is not None and lc_type >= 0:
        grp = _LC_TYPE_TO_GROUP.get(lc_type, -1)
        if grp < 0: return SLOT_WEIGHTS[ki]
        return (SLOT_WEIGHTS_PRE_LC[grp][ki] if ti < lc_frame_ti
                else SLOT_WEIGHTS_POST_LC[grp][ki])
    elif 0 <= lane_level <= 2:
        return SLOT_WEIGHTS_BY_LANE_LEVEL[lane_level][ki]
    return SLOT_WEIGHTS[ki]


def compute_importance_lis(lis, delta_lane, lc_state):
    p = IMPORTANCE_PARAMS_LIS
    ix = (np.exp(-(lis**2) / (2*p['sx']**2))
          * np.exp(-p['ax'] * lc_state) * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state**2) / (2*p['sy']**2))
          * np.exp(-p['ay'] * abs(lis)**p['py']) * np.exp(-p['by'] * delta_lane))
    return float(ix), float(iy), float(np.sqrt((ix**2 + iy**2) / 2.0))


def compute_importance_lit(lit, delta_lane, lc_state):
    p = IMPORTANCE_PARAMS_LIT
    ix = (np.exp(-(lit**2) / (2*p['sx']**2))
          * np.exp(-p['ax'] * lc_state) * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state**2) / (2*p['sy']**2))
          * np.exp(-p['ay'] * abs(lit)**1.5) * np.exp(-p['by'] * delta_lane))
    return float(ix), float(iy), float(np.sqrt((ix**2 + iy**2) / 2.0))


def balanced_split(rec_counts, ratios=(0.7, 0.1, 0.2), seed=42):
    rng   = np.random.default_rng(seed)
    total = sum(rec_counts.values())
    tgts  = [total * r for r in ratios]
    items = list(rec_counts.items())
    rng.shuffle(items)
    splits = {"train": [], "val": [], "test": []}
    sums   = {"train": 0,  "val": 0,  "test": 0}
    keys   = ["train", "val", "test"]
    for rid, cnt in sorted(items, key=lambda x: -x[1]):
        deficits = {k: tgts[i] - sums[k] for i, k in enumerate(keys)}
        best = max(deficits, key=deficits.get)
        splits[best].append(rid)
        sums[best] += cnt
    return splits


# ─── NGSIM data loading & filtering ─────────────────────────────────────────

def _smooth_deriv(arr, dt):
    d = np.gradient(arr, dt)
    return np.convolve(d, np.ones(3) / 3.0, mode='same')


def load_freeway_data(ngsim_path, args):
    """Read NGSIM CSV, keep freeway locations, apply basic filters.
    Returns dict {location: dataframe}.
    """
    needed = [NC_ID, NC_FRAME, NC_X, NC_Y, NC_LEN, NC_WID,
              NC_CLASS, NC_VEL, NC_ACC, NC_LANE, NC_LOC]
    print(f"Reading {ngsim_path} …")
    df = pd.read_csv(ngsim_path, usecols=needed)
    print(f"  Total rows: {len(df):,}")

    df = df[df[NC_LOC].isin(FREEWAY_LOCATIONS)].copy()
    print(f"  Freeway rows: {len(df):,}")

    step      = max(1, int(round(FPS / TARGET_HZ)))
    min_frames = (T_H + T_F) * step + 1

    result = {}
    for loc in sorted(df[NC_LOC].unique()):
        sub = df[df[NC_LOC] == loc].copy()

        # Duplicate rows
        before = len(sub)
        sub = sub.drop_duplicates(subset=[NC_ID, NC_FRAME])
        if len(sub) < before:
            print(f"  [{loc}] removed {before-len(sub):,} duplicate rows")

        # Invalid lane IDs
        sub = sub[(sub[NC_LANE] >= 1) & (sub[NC_LANE] <= MAX_LANES[loc])].copy()
        # Motorcycles
        sub = sub[sub[NC_CLASS] != 1].copy()

        # feet → meters
        for col in [NC_X, NC_Y, NC_LEN, NC_WID, NC_VEL, NC_ACC]:
            sub[col] = sub[col] * FT2M

        # Longitudinal center (Local_Y is front-center)
        sub['xc'] = sub[NC_Y] - 0.5 * sub[NC_LEN]
        sub['yc'] = sub[NC_X]   # lateral center

        sub = sub.sort_values([NC_ID, NC_FRAME]).reset_index(drop=True)

        # ── Per-track anomaly filter + yV/yA computation ──────────────────
        yv_out  = np.zeros(len(sub), np.float32)
        ya_out  = np.zeros(len(sub), np.float32)
        bad_ids = set()
        drop_reasons = dict(too_short=0, jump=0, speed=0, acc=0)

        for vid, grp in sub.groupby(NC_ID, sort=False):
            idx   = grp.index.to_numpy()
            order = np.argsort(grp[NC_FRAME].to_numpy())
            idx   = idx[order]

            if len(idx) < min_frames:
                bad_ids.add(vid); drop_reasons['too_short'] += 1; continue

            xc = sub.loc[idx, 'xc'].to_numpy(np.float64)
            yc = sub.loc[idx, 'yc'].to_numpy(np.float64)
            xv = sub.loc[idx, NC_VEL].to_numpy(np.float64)
            xa = sub.loc[idx, NC_ACC].to_numpy(np.float64)

            if (np.abs(np.diff(xc)) > args.jump_lon_m).any() or \
               (np.abs(np.diff(yc)) > args.jump_lat_m).any():
                bad_ids.add(vid); drop_reasons['jump'] += 1; continue

            if xv.mean() > args.speed_max_ms or xv.mean() < args.speed_min_ms:
                bad_ids.add(vid); drop_reasons['speed'] += 1; continue

            if (np.abs(xa) > args.acc_max_ms2).any():
                bad_ids.add(vid); drop_reasons['acc'] += 1; continue

            yv_out[idx] = _smooth_deriv(yc, DT).astype(np.float32)
            ya_out[idx] = _smooth_deriv(yv_out[idx].astype(np.float64), DT).astype(np.float32)

        n_orig = sub[NC_ID].nunique()
        sub = sub[~sub[NC_ID].isin(bad_ids)].copy()
        print(f"  [{loc}] {n_orig} veh → {sub[NC_ID].nunique()} kept  "
              f"(short={drop_reasons['too_short']}, jump={drop_reasons['jump']}, "
              f"speed={drop_reasons['speed']}, acc={drop_reasons['acc']})")

        sub['xVelocity']     = sub[NC_VEL].astype(np.float32)
        sub['yVelocity']     = yv_out[sub.index]
        sub['xAcceleration'] = sub[NC_ACC].astype(np.float32)
        sub['yAcceleration'] = ya_out[sub.index]
        result[loc] = sub.reset_index(drop=True)

    return result


# ─── Per-frame neighbor computation ──────────────────────────────────────────

def _neighbors_one_frame(vids, lids, xc, hl):
    N  = len(vids)
    nb = np.zeros((N, 8), dtype=np.int32)
    for i in range(N):
        lid = lids[i]; xi = xc[i]; hli = hl[i]
        xf = xi + hli; xr = xi - hli
        for slot, tlane, rel in (
            (0, lid,     'prec'), (1, lid,     'foll'),
            (2, lid - 1, 'prec'), (3, lid - 1, 'along'), (4, lid - 1, 'foll'),
            (5, lid + 1, 'prec'), (6, lid + 1, 'along'), (7, lid + 1, 'foll'),
        ):
            mask = (lids == tlane); mask[i] = False
            if not np.any(mask): continue
            idx_n = np.where(mask)[0]
            xc_n  = xc[idx_n]; hl_n = hl[idx_n]
            if rel == 'prec':
                ahead = xc_n > xi
                if np.any(ahead):
                    c = idx_n[ahead]
                    nb[i, slot] = vids[c[np.argmin(xc[c])]]
            elif rel == 'foll':
                behind = xc_n < xi
                if np.any(behind):
                    c = idx_n[behind]
                    nb[i, slot] = vids[c[np.argmax(xc[c])]]
            else:
                ov = (xc_n - hl_n < xf) & (xc_n + hl_n > xr)
                if np.any(ov):
                    c = idx_n[ov]
                    nb[i, slot] = vids[c[np.argmin(np.abs(xc[c] - xi))]]
    return nb


def compute_all_neighbors(df):
    df = df.reset_index(drop=True)
    nb_array = np.zeros((len(df), 8), dtype=np.int32)
    vids = df[NC_ID].to_numpy(np.int32)
    fids = df[NC_FRAME].to_numpy(np.int64)
    lids = df[NC_LANE].to_numpy(np.int32)
    xc   = df['xc'].to_numpy(np.float64)
    hl   = (df[NC_LEN] * 0.5).to_numpy(np.float64)
    for fr in tqdm(np.unique(fids), desc="  neighbors", leave=False):
        rows = np.where(fids == fr)[0]
        nb_array[rows] = _neighbors_one_frame(
            vids[rows], lids[rows], xc[rows], hl[rows])
    for k, col in enumerate(NEIGHBOR_COLS_8):
        df[col] = nb_array[:, k]
    return df


# ─── Per-location sample extraction ──────────────────────────────────────────

def process_location(loc, df, rec_id, args):
    print(f"  [rec {rec_id}] {loc}: computing neighbors …")
    df = compute_all_neighbors(df)
    df = df.reset_index(drop=True)

    # Sorted lane IDs (for conditional slot weights + lane level)
    sorted_lids = sorted(int(l) for l in df[NC_LANE].unique())

    # Lane centers → lat_lane_offset and lat_lane_width
    lane_centers = {lid: float(df.loc[df[NC_LANE] == lid, 'yc'].median())
                    for lid in sorted_lids}
    y_min = float(df['yc'].min())
    x_min = float(df['xc'].min())
    sh_centers = {lid: c - y_min for lid, c in lane_centers.items()}

    # Global shift
    df['xc'] = (df['xc'] - x_min).astype(np.float32)
    df['yc'] = (df['yc'] - y_min).astype(np.float32)

    # Flat arrays
    frame_arr   = df[NC_FRAME].to_numpy(np.int32)
    vid_arr     = df[NC_ID].to_numpy(np.int32)
    x_arr       = df['xc'].to_numpy(np.float32)
    y_arr       = df['yc'].to_numpy(np.float32)
    xv_arr      = df['xVelocity'].to_numpy(np.float32)
    yv_arr      = df['yVelocity'].to_numpy(np.float32)
    xa_arr      = df['xAcceleration'].to_numpy(np.float32)
    ya_arr      = df['yAcceleration'].to_numpy(np.float32)
    lane_id_arr = df[NC_LANE].to_numpy(np.int16)
    len_arr     = df[NC_LEN].to_numpy(np.float32)
    wid_arr     = df[NC_WID].to_numpy(np.float32)
    class_arr   = df[NC_CLASS].to_numpy(np.int8)
    nb_ids_all  = np.stack([df[c].to_numpy(np.int32) for c in NEIGHBOR_COLS_8], axis=1)

    # lat_lane_offset and lat_lane_width
    lat_lco = np.zeros(len(df), np.float32)
    lat_lw  = np.full(len(df), 3.75, np.float32)
    for i, lid in enumerate(sorted_lids):
        mask = (lane_id_arr == lid)
        lat_lco[mask] = y_arr[mask] - float(sh_centers[lid])
        if i + 1 < len(sorted_lids):
            w = abs(lane_centers[sorted_lids[i+1]] - lane_centers[lid])
        elif i > 0:
            w = abs(lane_centers[lid] - lane_centers[sorted_lids[i-1]])
        else:
            w = 3.75
        lat_lw[mask] = max(w, 0.5)

    # Per-vehicle lookup
    vid_to_len   = {}
    vid_to_wid   = {}
    vid_to_class = {}
    per_vid_rows: Dict[int, np.ndarray]    = {}
    per_vid_f2r:  Dict[int, Dict[int,int]] = {}

    for v, grp_idx in df.groupby(NC_ID).indices.items():
        grp_idx = np.array(grp_idx, np.int32)[np.argsort(frame_arr[np.array(grp_idx)])]
        per_vid_rows[int(v)] = grp_idx
        per_vid_f2r[int(v)]  = {int(frame_arr[r]): int(r) for r in grp_idx}
        vid_to_len[int(v)]   = float(len_arr[grp_idx[0]])
        vid_to_wid[int(v)]   = float(wid_arr[grp_idx[0]])
        vid_to_class[int(v)] = "Car" if int(class_arr[grp_idx[0]]) == 2 else "Truck"

    step        = max(1, int(round(FPS / TARGET_HZ)))
    stride      = max(1, int(round(STRIDE_SEC * TARGET_HZ)))
    selected_idx = EXTRA_FEATURE_MAP[args.feature_mode]
    nb_feat_dim  = len(selected_idx)
    num_c        = get_num_channels(args.feature_mode)
    ego_vel_ch   = slice(0, 2)
    nb_feat_ch   = slice(2, 2 + nb_feat_dim)
    is_ego_ch    = 2 + nb_feat_dim

    samples = []

    for v, idxs in tqdm(per_vid_rows.items(), desc=f"  [{loc}] windowing", leave=False):
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

            ego_rows = [per_vid_f2r[v][hf] for hf in hist_frames]
            fut_rows = [per_vid_f2r[v][ff] for ff in fut_frames]

            ex  = x_arr[ego_rows];  ey  = y_arr[ego_rows]
            exv = xv_arr[ego_rows]; eyv = yv_arr[ego_rows]
            exa = xa_arr[ego_rows]; eya = ya_arr[ego_rows]
            ego_lanes = lane_id_arr[ego_rows].astype(np.int32)
            len_ego   = vid_to_len.get(v, 0.0)

            norm_center = np.array([ex[-1], ey[-1]], np.float32)

            # ── conditional slot weight context ───────────────────────────
            _lc_lv, _lc_fti, _lc_type = -1, None, -1
            if args.slot_importance_conditional and args.slot_importance_alpha > 0.0:
                _lc_lv, _lc_fti, _lc_type = _ego_lc_context(ego_lanes, sorted_lids)

            # ── GRIP tensor ───────────────────────────────────────────────
            tensor = np.zeros((1 + MAX_NEIGHBORS, T_H, num_c), dtype=np.float32)
            adj    = np.eye(1 + MAX_NEIGHBORS, dtype=np.float32)

            # Ego node
            tensor[0, :, ego_vel_ch] = np.stack([exv, eyv], axis=1)
            tensor[0, :, is_ego_ch]  = 1.0

            # Neighbor IDs at obs_frame (last history frame)
            ids8_obs = nb_ids_all[ego_rows[-1]]

            # Valid neighbors: present at ALL T_H frames
            valid_nbs = {}
            for ki in range(MAX_NEIGHBORS):
                nid = int(ids8_obs[ki])
                if nid <= 0: continue
                rm = per_vid_f2r.get(nid)
                if rm is None: continue
                nb_rows_ki = [rm.get(int(hf)) for hf in hist_frames]
                if any(r is None for r in nb_rows_ki): continue
                valid_nbs[ki] = (nid, nb_rows_ki)

            # All-neighbor feature matrix
            nb_all = np.zeros((MAX_NEIGHBORS, T_H, NB_DIM), np.float32)
            nb_msk = np.zeros((MAX_NEIGHBORS, T_H), bool)

            for ti, hf in enumerate(hist_frames):
                for ki, (nid, nb_rows_ki) in valid_nbs.items():
                    nr  = nb_rows_ki[ti]
                    dx  = float(x_arr[nr]  - ex[ti])
                    dy  = float(y_arr[nr]  - ey[ti])
                    dvx = float(xv_arr[nr] - exv[ti])
                    dvy = float(yv_arr[nr] - eyv[ti])
                    dax = float(xa_arr[nr] - exa[ti])
                    day = float(ya_arr[nr] - eya[ti])

                    # ── lc_state ─────────────────────────────────────────
                    if args.lc_version == 'v1':
                        vyn = float(yv_arr[nr])
                        if ki < 2:
                            lc_state = 1.0
                        elif abs(vyn) < args.vy_eps:
                            lc_state = 1.0
                        elif ki < 5:
                            lc_state = 0.0 if vyn < 0 else 2.0
                        else:
                            lc_state = 0.0 if vyn > 0 else 2.0

                    elif args.lc_version == 'v2':
                        abs_dvy = abs(dvy)
                        if ki < 2 and abs(dy) < args.dy_same:
                            lc_state = 2.0 if abs_dvy > args.dvy_eps_same else 1.0
                        elif ki >= 2:
                            lc_state = (0.0 if dy * dvy < 0 else 2.0) \
                                if abs_dvy > args.dvy_eps_cross else 1.0
                        else:
                            lc_state = 0.0 if dy * dvy < 0 else 2.0

                    elif args.lc_version == 'v3':
                        nb_lat_v = float(yv_arr[nr])
                        nb_lco   = float(lat_lco[nr])
                        if ki < 2:
                            if (nb_lco < -1.0 and nb_lat_v > 0.0) or \
                               (nb_lco >  1.0 and nb_lat_v < 0.0):
                                lc_state = 0.0
                            elif (nb_lco < -1.0 and nb_lat_v < 0.0) or \
                                 (nb_lco >  1.0 and nb_lat_v > 0.0) or \
                                 abs(nb_lat_v) > 0.029:
                                lc_state = 2.0
                            else:
                                lc_state = 1.0
                        elif ki < 5:
                            lc_state = (0.0 if nb_lat_v < -0.029 else
                                        2.0 if nb_lat_v >  0.029 else 1.0)
                        else:
                            lc_state = (2.0 if nb_lat_v < -0.029 else
                                        0.0 if nb_lat_v >  0.029 else 1.0)

                    else:  # v4
                        nb_lat_v    = float(yv_arr[nr])
                        nb_lco      = float(lat_lco[nr])
                        nb_lw_      = float(lat_lw[nr])
                        nb_lco_norm = nb_lco / (nb_lw_ * 0.5) if nb_lw_ > 0.5 else 0.0
                        if abs(nb_lco_norm) <= 0.5:
                            lc_state = 1.0
                        elif ki < 2:
                            lc_state = 0.0 if nb_lco_norm * nb_lat_v < 0 else 2.0
                        elif ki < 5:
                            lc_state = 0.0 if nb_lat_v < 0 else 2.0
                        else:
                            lc_state = 0.0 if nb_lat_v > 0 else 2.0

                    # ── LIT / LIS / importance ────────────────────────────
                    len_nb   = vid_to_len.get(nid, 0.0)
                    half_sum = 0.5 * (len_ego + len_nb)
                    if dx >= 0:
                        gap, denom_base = abs(dx - half_sum), dvx
                    else:
                        gap, denom_base = abs(-dx - half_sum), -dvx
                    eps = args.eps_gate if denom_base >= 0 else -args.eps_gate
                    lit = gap / (denom_base + eps)
                    lis = _lit_to_lis(lit, args.lis_mode)

                    nb_phys_l = vid_to_len.get(nid, 0.0)
                    nb_phys_w = vid_to_wid.get(nid, 0.0)
                    nb_class  = vid_to_class.get(nid, "Car")
                    size_bin, nb_volume = _volume_bin(nb_phys_l, nb_phys_w, nb_class)

                    delta_lane = float(abs(int(lane_id_arr[nr]) - int(ego_lanes[ti])))

                    if args.importance_mode == 'lit':
                        ix, iy, i_total = compute_importance_lit(lit, delta_lane, lc_state)
                    else:
                        ix, iy, i_total = compute_importance_lis(lis, delta_lane, lc_state)

                    if args.slot_importance_alpha > 0.0:
                        if args.slot_importance_conditional:
                            w = _get_slot_weight(ki, ti, _lc_lv, _lc_fti, _lc_type)
                        else:
                            w = SLOT_WEIGHTS[ki]
                        i_total = min(i_total * (1.0 + args.slot_importance_alpha * w), 1.0)

                    gate = 1.0 if (args.gate_theta <= 0.0 or i_total >= args.gate_theta) else 0.0

                    nb_all[ki, ti] = [dx, dy, dvx, dvy, dax, day,
                                      lc_state, nb_volume, size_bin, gate,
                                      ix * gate, iy * gate, i_total * gate]
                    nb_msk[ki, ti] = True

                if args.gate_topn > 0:
                    _apply_topn_gate(nb_all[:, ti, :], nb_msk[:, ti], args.gate_topn)

            # Fill tensor
            for ki in valid_nbs:
                tensor[ki + 1, :, nb_feat_ch] = nb_all[ki, :, :][:, selected_idx]
                adj[0, ki + 1] = adj[ki + 1, 0] = 1.0

            fut_xy = np.stack([x_arr[fut_rows], y_arr[fut_rows]], axis=1)
            target = (fut_xy - norm_center).astype(np.float32)

            samples.append({
                "input":       tensor,
                "adj":         adj,
                "target":      target,
                "recordingId": rec_id,
                "trackId":     int(v),
                "t0_frame":    int(t0_frame),
            })
            t0_frame += stride * step

    print(f"  [rec {rec_id}] {loc}: {len(samples):,} samples")
    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NGSIM preprocessing for GRIP model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ngsim_csv", default="../neighformer/data/NGSIM/NGSIM_data.csv",
                        help="Path to combined NGSIM CSV")
    parser.add_argument("--out_dir",   default="NGSIM",
                        help="Output directory for HDF5 files")
    parser.add_argument("--feature_mode", default="baseline",
                        choices=list(EXTRA_FEATURE_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)

    # Anomaly filter thresholds
    parser.add_argument("--speed_max_ms", type=float, default=45.0)
    parser.add_argument("--speed_min_ms", type=float, default=-0.5)
    parser.add_argument("--acc_max_ms2",  type=float, default=9.0)
    parser.add_argument("--jump_lon_m",   type=float, default=5.0)
    parser.add_argument("--jump_lat_m",   type=float, default=1.5)

    # Importance / gate
    parser.add_argument("--eps_gate",   type=float, default=1.0)
    parser.add_argument("--gate_theta", type=float, default=0.0)
    parser.add_argument("--gate_topn",  type=int,   default=0)
    parser.add_argument("--slot_importance_alpha", type=float, default=0.0,
                        dest="slot_importance_alpha")
    parser.add_argument("--slotImportanceConditional", action="store_true", default=False,
                        dest="slot_importance_conditional")

    # lc_state
    parser.add_argument("--lc_version",      default="v3", choices=["v1","v2","v3","v4"])
    parser.add_argument("--lis_mode",        default="3",  choices=list(LIS_BINS.keys()))
    parser.add_argument("--importance_mode", default="lis", choices=["lis","lit"])
    parser.add_argument("--vy_eps",          type=float, default=0.27)
    parser.add_argument("--dvy_eps_cross",   type=float, default=0.26)
    parser.add_argument("--dvy_eps_same",    type=float, default=1.03)
    parser.add_argument("--dy_same",         type=float, default=1.5)

    args = parser.parse_args()

    nb_feat_dim = len(EXTRA_FEATURE_MAP[args.feature_mode])
    num_c       = get_num_channels(args.feature_mode)
    print(f"[NGSIM → GRIP]  feature_mode={args.feature_mode}  "
          f"lc_version={args.lc_version}  importance={args.importance_mode}")
    print(f"T_H={T_H}  T_F={T_F}  target_hz={TARGET_HZ}")
    print(f"Channel layout: [ego_vx,ego_vy | {nb_feat_dim} nb_feats | is_ego] → {num_c} ch")

    # Load data
    loc_dfs = load_freeway_data(Path(args.ngsim_csv), args)

    # Process each location
    all_loc_samples: Dict[int, list] = {}
    for rec_id, (loc, df) in enumerate(loc_dfs.items()):
        samples = process_location(loc, df, rec_id, args)
        if samples:
            all_loc_samples[rec_id] = samples

    if not all_loc_samples:
        raise RuntimeError("No samples produced.")

    all_samples = [s for samples in all_loc_samples.values() for s in samples]
    total = len(all_samples)
    print(f"\nTotal samples: {total:,}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(total)
    n_train = int(total * 0.7)
    n_val   = int(total * 0.1)
    idx_splits = {
        "train": perm[:n_train],
        "val":   perm[n_train:n_train + n_val],
        "test":  perm[n_train + n_val:],
    }

    out_dir = Path(args.out_dir) / args.feature_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, idxs in idx_splits.items():
        split_data = [all_samples[i] for i in idxs]
        if not split_data:
            print(f"  {split_name}: empty, skipping")
            continue
        out_path = out_dir / f"{split_name}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("input",  data=np.array([s["input"]  for s in split_data]),
                             compression="gzip")
            f.create_dataset("adj",    data=np.array([s["adj"]    for s in split_data]),
                             compression="gzip")
            f.create_dataset("target", data=np.array([s["target"] for s in split_data]),
                             compression="gzip")
            f.create_dataset("meta_recordingId",
                             data=np.array([s["recordingId"] for s in split_data], np.int32))
            f.create_dataset("meta_trackId",
                             data=np.array([s["trackId"] for s in split_data], np.int32))
            f.create_dataset("meta_t0_frame",
                             data=np.array([s["t0_frame"] for s in split_data], np.int32))
        print(f"  {split_name}.h5  ({len(split_data):,} samples)  → {out_path}")


if __name__ == "__main__":
    main()
