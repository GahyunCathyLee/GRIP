import os
import bisect
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
NB_DIM        = 13   # dx, dy, dvx, dvy, dax, day, lc_state, lit, lis, gate, I_x, I_y, I

NEIGHBOR_COLS_8 = [
    "precedingId", "followingId",
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
]

# Slot priority for top-N gate tie-breaking: 0 > 2 > 5 > 1 > 4 > 7 > 3 > 6
_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}

# Empirical slot weights (mean I per slot, from dataset analysis)
# Order: preceding, following, leftPreceding, leftAlongside, leftFollowing,
#        rightPreceding, rightAlongside, rightFollowing
SLOT_WEIGHTS = [0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179]

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
#            [6]lc_state [7]lit [8]lis [9]gate [10]I_x [11]I_y [12]I
# ==============================================================================
EXTRA_FEATURE_MAP = {
    'baseline':   [0, 1, 2, 3, 4, 5],        # dx, dy, dvx, dvy, dax, day
    'importance': [0, 1, 2, 3, 4, 5, 12],    # dx, dy, dvx, dvy, dax, day, I
    'sy':         [0, 1, 2, 3, 4, 5, 6],     # dx, dy, dvx, dvy, dax, day, lc_state
    'iy':         [0, 1, 2, 3, 4, 5, 11],    # dx, dy, dvx, dvy, dax, day, I_y
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
    tracks = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")
    tmeta  = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    rmeta  = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")

    frame_rate = float(rmeta.loc[0, "frameRate"])
    step   = max(1, int(round(frame_rate / TARGET_HZ)))
    stride = max(1, int(round(STRIDE_SEC * TARGET_HZ)))

    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns: tracks[c] = 0
    for c in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        if c not in tracks.columns: tracks[c] = 0.0
    if "laneId" not in tracks.columns: tracks["laneId"] = 0

    vid_to_dd = dict(zip(tmeta["id"].astype(int), tmeta["drivingDirection"].astype(int)))
    vid_to_w  = dict(zip(tmeta["id"].astype(int), tmeta["width"].astype(float)))
    vid_to_h  = dict(zip(tmeta["id"].astype(int), tmeta["height"].astype(float)))

    vid_arr   = tracks["id"].astype(np.int32).to_numpy()
    frame_arr = tracks["frame"].astype(np.int32).to_numpy()
    x_arr     = tracks["x"].astype(np.float32).to_numpy().copy()
    y_arr     = tracks["y"].astype(np.float32).to_numpy().copy()
    w_row = np.array([vid_to_w.get(int(v), 0.0) for v in vid_arr], np.float32)
    h_row = np.array([vid_to_h.get(int(v), 0.0) for v in vid_arr], np.float32)
    x_arr += 0.5 * w_row   # bounding-box corner -> center
    y_arr += 0.5 * h_row
    xv_arr   = tracks["xVelocity"].astype(np.float32).to_numpy()
    yv_arr   = tracks["yVelocity"].astype(np.float32).to_numpy()
    xa_arr   = tracks["xAcceleration"].astype(np.float32).to_numpy()
    ya_arr   = tracks["yAcceleration"].astype(np.float32).to_numpy()
    lane_arr = tracks["laneId"].astype(np.int16).to_numpy().copy()
    dd_arr   = np.array([vid_to_dd.get(int(v), 0) for v in vid_arr], np.int8)

    # Parse lane markings (needed for lat_lane_offset and flip)
    upper_marks_raw = parse_semicolon_floats(str(rmeta.loc[0, "upperLaneMarkings"])) \
        if "upperLaneMarkings" in rmeta.columns else []
    lower_marks_raw = parse_semicolon_floats(str(rmeta.loc[0, "lowerLaneMarkings"])) \
        if "lowerLaneMarkings" in rmeta.columns else []
    upper_mark = np.array(upper_marks_raw, np.float32)
    lower_mark = np.array(lower_marks_raw, np.float32)
    _N_upper   = len(upper_mark)

    # Lat-lane offset and width — computed in pre-flip coords (post center-correction)
    _lid_arr            = lane_arr.astype(np.int32)
    lat_lane_offset_arr = np.zeros(len(y_arr), np.float32)
    lat_lane_width_arr  = np.full(len(y_arr), 3.75, np.float32)

    # lower-direction vehicles (dd==2): j = lid - N_upper - 2
    _mask_lo = (dd_arr == 2)
    _j_lo    = _lid_arr - _N_upper - 2
    _ok_lo   = _mask_lo & (_j_lo >= 0) & (_j_lo < len(lower_mark) - 1)
    if np.any(_ok_lo):
        lat_lane_offset_arr[_ok_lo] = (
            y_arr[_ok_lo]
            - 0.5 * (lower_mark[_j_lo[_ok_lo]] + lower_mark[_j_lo[_ok_lo] + 1])
        )
        lat_lane_width_arr[_ok_lo] = np.abs(
            lower_mark[_j_lo[_ok_lo] + 1] - lower_mark[_j_lo[_ok_lo]]
        )

    # upper-direction vehicles (dd==1): j = lid - 2  (Lane1 → j=-1 → invalid)
    _mask_up = (dd_arr == 1)
    _j_up    = _lid_arr - 2
    _ok_up   = _mask_up & (_j_up >= 0) & (_j_up < len(upper_mark) - 1)
    if np.any(_ok_up):
        lat_lane_offset_arr[_ok_up] = (
            y_arr[_ok_up]
            - 0.5 * (upper_mark[_j_up[_ok_up]] + upper_mark[_j_up[_ok_up] + 1])
        )
        lat_lane_width_arr[_ok_up] = np.abs(
            upper_mark[_j_up[_ok_up] + 1] - upper_mark[_j_up[_ok_up]]
        )

    # maybe_flip negates y for upper vehicles → negate lco to match
    lat_lane_offset_arr[dd_arr == 1] *= -1.0

    # Normalize: flip upper-lane vehicles (drivingDirection==1) so all drive left-to-right
    if args.normalize_flip:
        if len(upper_mark) and len(lower_mark):
            C_y   = float(upper_mark[-1] + lower_mark[0])
            x_max = float(np.nanmax(x_arr))
            mask  = dd_arr == 1
            x_arr[mask]  = x_max - x_arr[mask]
            y_arr[mask]  = C_y   - y_arr[mask]
            xv_arr[mask] = -xv_arr[mask]
            yv_arr[mask] = -yv_arr[mask]
            xa_arr[mask] = -xa_arr[mask]
            ya_arr[mask] = -ya_arr[mask]
            n_upper = len(upper_mark) - 1
            if n_upper > 0:
                ok = mask & (lane_arr > 0)
                lane_arr[ok] = np.int16(1 + n_upper) - lane_arr[ok]

    # Build per-vehicle sorted row-index arrays and frame->row dicts
    per_vid_rows: dict = {}
    per_vid_frame_to_row: dict = {}
    for v, idxs in tracks.groupby("id").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame_arr[idxs])]
        per_vid_rows[int(v)] = idxs
        per_vid_frame_to_row[int(v)] = {int(frame_arr[r]): int(r) for r in idxs}

    nb_ids_all = np.stack(
        [tracks[c].astype(np.int32).to_numpy() for c in NEIGHBOR_COLS_8], axis=1
    )

    # Feature selection setup
    selected_indices = EXTRA_FEATURE_MAP[args.feature_mode]
    nb_feat_dim = len(selected_indices)
    num_c       = get_num_channels(args.feature_mode)
    ego_vel_ch  = slice(0, 2)
    nb_feat_ch  = slice(2, 2 + nb_feat_dim)
    is_ego_ch   = 2 + nb_feat_dim

    samples = []

    for v, idxs in per_vid_rows.items():
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

            # GRIP tensor: (1 + MAX_NEIGHBORS, T_H, num_c)
            tensor = np.zeros((1 + MAX_NEIGHBORS, T_H, num_c), dtype=np.float32)
            adj    = np.eye(1 + MAX_NEIGHBORS, dtype=np.float32)

            # Ego node: ch 0,1 = vx,vy  /  is_ego = 1
            norm_center = np.array([ex[-1], ey[-1]], np.float32)
            tensor[0, :, ego_vel_ch] = np.stack([exv, eyv], axis=1)
            tensor[0, :, is_ego_ch]  = 1.0

            # Neighbor IDs determined at obs_frame (last history frame)
            ids8_obs = nb_ids_all[ego_rows[-1]]

            # Identify valid neighbors (must be present at ALL T_H history frames)
            valid_nbs = {}
            for ki in range(MAX_NEIGHBORS):
                nid = int(ids8_obs[ki])
                if nid <= 0: continue
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

                    dx  = float(x_arr[nr]  - ex[ti])
                    dy  = float(y_arr[nr]  - ey[ti])
                    dvx = float(xv_arr[nr] - exv[ti])
                    dvy = float(yv_arr[nr] - eyv[ti])
                    dax = float(xa_arr[nr] - exa[ti])
                    day = float(ya_arr[nr] - eya[ti])

                    # ── lc_state ─────────────────────────────────────────────
                    if args.lc_version == "v1":
                        vyn = float(yv_arr[nr])
                        if ki < 2:
                            lc_state = 1.0
                        elif abs(vyn) < args.vy_eps:
                            lc_state = 1.0
                        elif ki < 5:
                            lc_state = 0.0 if vyn < 0 else 2.0
                        else:
                            lc_state = 0.0 if vyn > 0 else 2.0
                    elif args.lc_version == "v2":
                        abs_dvy = abs(dvy)
                        if ki < 2 and abs(dy) < args.dy_same:
                            lc_state = 2.0 if abs_dvy > args.dvy_eps_same else 1.0
                        elif ki >= 2:
                            lc_state = (0.0 if dy * dvy < 0 else 2.0) \
                                if abs_dvy > args.dvy_eps_cross else 1.0
                        else:
                            lc_state = 0.0 if dy * dvy < 0 else 2.0
                    elif args.lc_version == "v3":
                        nb_lat_v = float(yv_arr[nr])
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
                        nb_lat_v    = float(yv_arr[nr])
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

                    delta_lane = float(abs(int(lane_arr[nr]) - int(ego_lanes[ti])))

                    # ── importance ────────────────────────────────────────────
                    if args.importance_mode == 'lit':
                        ix, iy, i_total = compute_importance_lit(lit, delta_lane, lc_state)
                    else:  # 'lis' (default)
                        ix, iy, i_total = compute_importance_lis(lis, delta_lane, lc_state)

                    # ── slot importance boost: I_new = min(I * (1 + alpha * w_slot), 1.0) ──
                    if args.slot_importance_alpha > 0.0:
                        i_total = min(
                            i_total * (1.0 + args.slot_importance_alpha * SLOT_WEIGHTS[ki]),
                            1.0,
                        )

                    # ── gate ──────────────────────────────────────────────────
                    gate    = 1.0 if (args.gate_theta <= 0.0 or i_total >= args.gate_theta) else 0.0

                    nb_all_feats[ki, ti] = [dx, dy, dvx, dvy, dax, day,
                                            lc_state, lit, lis, gate,
                                            ix * gate, iy * gate, i_total * gate]
                    nb_mask_mat[ki, ti]  = True

                # Apply gate_topn per timestep after all slots are filled
                if args.gate_topn > 0:
                    _apply_topn_gate(nb_all_feats[:, ti, :], nb_mask_mat[:, ti], args.gate_topn)

            # Fill tensor with selected features
            for ki in valid_nbs:
                tensor[ki + 1, :, nb_feat_ch] = nb_all_feats[ki, :, :][:, selected_indices]
                adj[0, ki + 1] = adj[ki + 1, 0] = 1.0

            # Target: future (x, y) relative to last observed ego position
            fut_xy = np.stack([x_arr[fut_rows], y_arr[fut_rows]], axis=1)  # (T_F, 2)
            target = fut_xy - norm_center

            samples.append({"input": tensor, "adj": adj, "target": target})
            t0_frame += stride * step

    return samples


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         type=str,   default="highD/raw")
    parser.add_argument("--out_dir",         type=str,   default="highD")
    parser.add_argument("--feature_mode",    type=str,   default="baseline",
                        choices=['baseline', 'importance', 'sy', 'iy'])
    parser.add_argument("--normalize_flip",  action="store_true", default=True)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--eps_gate",        type=float, default=1.0,
                        help="eps for LIT denominator (makes gap dominant over dvx)")
    parser.add_argument("--gate_theta",      type=float, default=0.0,
                        help="importance threshold for gate (0.0 = disabled)")
    parser.add_argument("--gate_topn",            type=int,   default=0,
                        help="keep top-N neighbors by I per timestep (0 = disabled)")
    parser.add_argument("--slot_importance_alpha", type=float, default=0.0,
                        help="slot importance boost: I_new = min(I*(1+alpha*w_slot),1.0) (0.0 = disabled)")
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

    raw_path = Path(args.raw_dir)
    rec_ids  = sorted(set([
        f.name.split("_")[0] for f in raw_path.glob("*_tracks.csv")
    ]))

    nb_feat_dim = len(EXTRA_FEATURE_MAP[args.feature_mode])
    num_c       = get_num_channels(args.feature_mode)
    print(f"Found {len(rec_ids)} recordings")
    print(f"Feature mode     : {args.feature_mode}")
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
            f.create_dataset("input",  data=np.array([s["input"]  for s in split_data]), compression="gzip")
            f.create_dataset("adj",    data=np.array([s["adj"]    for s in split_data]), compression="gzip")
            f.create_dataset("target", data=np.array([s["target"] for s in split_data]), compression="gzip")
        print(f"-> {split_name}.h5 saved ({len(split_data)} samples, input shape: {split_data[0]['input'].shape})")


if __name__ == "__main__":
    main()
