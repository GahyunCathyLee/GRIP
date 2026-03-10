import os
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
T_H           = 9    # 3 sec * 3 Hz (history)
T_F           = 15   # 5 sec * 3 Hz (future)
STRIDE_SEC    = 1.0
MAX_NEIGHBORS = 8
NB_DIM        = 12   # dx, dy, dvx, dvy, dax, day, lc_state, dx_time, gate, I_x, I_y, I

NEIGHBOR_COLS_8 = [
    "precedingId", "followingId",
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
]

# Importance formula params
# I_x = exp(-(dx_time^2)/(2*sx^2)) * exp(-ax*lc_state) * exp(-bx*delta_lane)
# I_y = exp(-(lc_state^2)/(2*sy^2)) * exp(-ay*|dx_time|^1.5) * exp(-by*delta_lane)
# I   = sqrt((I_x^2 + I_y^2) / 2)
IMPORTANCE_PARAMS = {"sx": 15.0, "ax": 0.2, "bx": 0.25, "sy": 2.0, "ay": 0.01, "by": 0.1}

# Neighbor feature index selection from NB_DIM=12 features:
#   [0]dx [1]dy [2]dvx [3]dvy [4]dax [5]day [6]lc_state [7]dx_time [8]gate [9]I_x [10]I_y [11]I
EXTRA_FEATURE_MAP = {
    'baseline':      [0, 1],                        # dx, dy
    'baseline_v':    [2, 3],                        # dvx, dvy
    'baseline_full': [0, 1, 2, 3, 4, 5],           # dx, dy, dvx, dvy, dax, day
    'importance':    [0, 1, 2, 3, 4, 5, 11],       # dx, dy, dvx, dvy, dax, day, I
    'exp1':          [0, 1, 8],                     # dx, dy, gate
    'exp2':          [0, 1, 6, 7],                  # dx, dy, lc_state, dx_time
    'exp3':          [6, 7],                        # lc_state, dx_time
    'exp4':          [4, 5, 6, 7, 8],               # dax, day, lc_state, dx_time, gate
    'exp5':          [0, 1, 2, 3, 4, 5, 8],         # 6 kinematics + gate
    'exp6':          [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 6 kinematics + lc_state + dx_time + gate
}


def get_num_channels(feature_mode):
    """총 채널 수: ego_vel(2) + nb_feats(N) + is_ego(1)"""
    nb_feats = len(EXTRA_FEATURE_MAP[feature_mode])
    return 2 + nb_feats + 1


# ==============================================================================
# Utilities
# ==============================================================================

def parse_semicolon_floats(s):
    if not isinstance(s, str):
        return []
    return [float(p) for p in s.strip().split(";") if p.strip()]


def compute_importance(dx_time, delta_lane, lc_state):
    p = IMPORTANCE_PARAMS
    ix = (np.exp(-(dx_time ** 2) / (2.0 * p["sx"] ** 2))
          * np.exp(-p["ax"] * lc_state)
          * np.exp(-p["bx"] * delta_lane))
    iy = (np.exp(-(lc_state ** 2) / (2.0 * p["sy"] ** 2))
          * np.exp(-p["ay"] * (abs(dx_time) ** 1.5))
          * np.exp(-p["by"] * delta_lane))
    i_total = np.sqrt((ix ** 2 + iy ** 2) / 2.0)
    return float(ix), float(iy), float(i_total)


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

    # Ensure required columns exist
    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns: tracks[c] = 0
    for c in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        if c not in tracks.columns: tracks[c] = 0.0
    if "laneId" not in tracks.columns: tracks["laneId"] = 0

    vid_to_dd = dict(zip(tmeta["id"].astype(int), tmeta["drivingDirection"].astype(int)))
    vid_to_w  = dict(zip(tmeta["id"].astype(int), tmeta["width"].astype(float)))
    vid_to_h  = dict(zip(tmeta["id"].astype(int), tmeta["height"].astype(float)))

    # Extract raw arrays (one entry per row in tracks, aligned by index)
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

    # Normalize: flip upper-lane vehicles (drivingDirection==1) so all drive left-to-right
    if args.normalize_flip:
        upper_marks = parse_semicolon_floats(str(rmeta.loc[0, "upperLaneMarkings"]))
        lower_marks = parse_semicolon_floats(str(rmeta.loc[0, "lowerLaneMarkings"]))
        if upper_marks and lower_marks:
            C_y   = upper_marks[-1] + lower_marks[0]
            x_max = float(np.nanmax(x_arr))
            mask  = dd_arr == 1
            x_arr[mask]  = x_max - x_arr[mask]
            y_arr[mask]  = C_y   - y_arr[mask]
            xv_arr[mask] = -xv_arr[mask]
            yv_arr[mask] = -yv_arr[mask]
            xa_arr[mask] = -xa_arr[mask]
            ya_arr[mask] = -ya_arr[mask]
            # Flip lane IDs for upper-lane vehicles (y-flip reverses lane ordering)
            n_upper = len(upper_marks) - 1
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

    # Neighbor ID array: shape (len(tracks), 8)
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

    W = max(1, int(round(TARGET_HZ)))  # rolling window size for lc_state v2 (1-sec)

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

            # GRIP tensor: (1 + MAX_NEIGHBORS, T_H, num_c)
            tensor = np.zeros((1 + MAX_NEIGHBORS, T_H, num_c), dtype=np.float32)
            adj    = np.eye(1 + MAX_NEIGHBORS, dtype=np.float32)

            # Ego node: ch 0,1 = vx,vy  /  is_ego = 1
            norm_center = np.array([ex[-1], ey[-1]], np.float32)
            tensor[0, :, ego_vel_ch] = np.stack([exv, eyv], axis=1)
            tensor[0, :, is_ego_ch]  = 1.0

            # Neighbor IDs determined at obs_frame (last history frame)
            ids8_obs = nb_ids_all[ego_rows[-1]]

            for ki in range(MAX_NEIGHBORS):
                nid = int(ids8_obs[ki])
                if nid <= 0:
                    continue
                rm = per_vid_frame_to_row.get(nid)
                if rm is None:
                    continue

                # Require neighbor present at all T_H history frames
                nb_rows = [rm.get(int(hf)) for hf in hist_frames]
                if any(r is None for r in nb_rows):
                    continue

                # Compute NB_DIM=12 features for each timestep
                nb_feat_mat = np.zeros((T_H, NB_DIM), np.float32)
                for ti, (hf, nr) in enumerate(zip(hist_frames, nb_rows)):
                    dx  = float(x_arr[nr]  - ex[ti])
                    dy  = float(y_arr[nr]  - ey[ti])
                    dvx = float(xv_arr[nr] - exv[ti])
                    dvy = float(yv_arr[nr] - eyv[ti])
                    dax = float(xa_arr[nr] - exa[ti])
                    day = float(ya_arr[nr] - eya[ti])

                    # lc_state v2: rolling avg yV over 1-sec window + dy sign
                    yv_vals = []
                    for w in range(W):
                        wr = rm.get(int(hf - w * step))
                        if wr is not None:
                            yv_vals.append(float(yv_arr[wr]))
                    vyn = float(np.mean(yv_vals)) if yv_vals else float(yv_arr[nr])

                    if abs(vyn) < args.vy_eps:
                        lc_state = 1.0   # staying
                    elif dy * vyn > 0:
                        lc_state = 2.0   # moving away from ego
                    else:
                        lc_state = 2.0 if ki < 2 else 0.0  # slots 0,1=same-lane; rest=closing in

                    # dx_time: eps raised to 1.0 to make dx dominant over dvx
                    denom   = dvx + (args.eps_gate if dvx >= 0 else -args.eps_gate)
                    dx_time = dx / denom
                    gate    = 1.0 if (-args.t_back < dx_time < args.t_front) else 0.0

                    # Importance
                    delta_lane = float(abs(int(lane_arr[nr]) - int(ego_lanes[ti])))
                    ix, iy, i_total = compute_importance(dx_time, delta_lane, lc_state)

                    nb_feat_mat[ti] = [dx, dy, dvx, dvy, dax, day,
                                       lc_state, dx_time, gate, ix, iy, i_total]

                tensor[ki + 1, :, nb_feat_ch] = nb_feat_mat[:, selected_indices]
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
    parser.add_argument("--raw_dir",        type=str,   default="highD/raw")
    parser.add_argument("--out_dir",        type=str,   default="highD")
    parser.add_argument("--feature_mode",   type=str,   default="baseline",
                        choices=list(EXTRA_FEATURE_MAP.keys()))
    parser.add_argument("--normalize_flip", action="store_true", default=True)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--t_front",        type=float, default=3.0)
    parser.add_argument("--t_back",         type=float, default=5.0)
    parser.add_argument("--vy_eps",         type=float, default=0.27)
    parser.add_argument("--eps_gate",       type=float, default=1.0,
                        help="eps for dx_time denominator (1.0 makes dx dominant over dvx)")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    rec_ids  = sorted(set([
        f.name.split("_")[0] for f in raw_path.glob("*_tracks.csv")
    ]))

    nb_feat_dim = len(EXTRA_FEATURE_MAP[args.feature_mode])
    num_c       = get_num_channels(args.feature_mode)
    print(f"Found {len(rec_ids)} recordings")
    print(f"Feature mode  : {args.feature_mode}")
    print(f"Target Hz     : {TARGET_HZ}  |  T_H={T_H}  |  T_F={T_F}")
    print(f"Channel layout: [ego_vx, ego_vy | {nb_feat_dim} nb_feats | is_ego]  ->  total {num_c}ch")
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
