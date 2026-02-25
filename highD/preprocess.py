import os
import pandas as pd
import numpy as np
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# 1. 실험 설정 및 인덱스 정의
# ==============================================================================
TARGET_HZ = 5.0  # GRIP++ 기준 5Hz
T_H = 15         # 3초 관측 (15 steps)
T_F = 25         # 5초 예측 (25 steps)
MAX_NEIGHBORS = 8 # highD 슬롯 기반 8대

# [0:dx, 1:dy, 2:dvx, 3:dvy, 4:dax, 5:day, 6:lc_state, 7:dx_time, 8:gate]
EXTRA_FEATURE_MAP = {
    'baseline': [0, 1],
    'exp1': [0, 1, 8],
    'exp2': [0, 1, 6, 7],
    'exp3': [6, 7],
    'exp4': [4, 5, 6, 7, 8],
    'exp5': [0, 1, 2, 3, 4, 5, 8],
    'exp6': [0, 1, 2, 3, 4, 5, 6, 7, 8],
}

# ==============================================================================
# 2. 유틸리티 함수 (Flip, Feature Compute, Split)
# ==============================================================================
def get_neighbor_features(ego_hist, nb_window, args):
    """주변 차량의 9가지 후보 피처 계산"""
    rel_pos = nb_window[:, 1:3] - ego_hist[:, 1:3] # dx, dy
    rel_vel = nb_window[:, 3:5] - ego_hist[:, 3:5] # dvx, dvy
    rel_acc = nb_window[:, 5:7] - ego_hist[:, 5:7] # dax, day (상대 가속도)

    dx, dy = rel_pos[:, 0], rel_pos[:, 1]
    dvx, dvy = rel_vel[:, 0], rel_vel[:, 1]

    # lc_state (차선 변경 상태)
    lc_state = np.zeros_like(dy)
    nb_vy_abs = nb_window[:, 4]
    mask_l, mask_r = dy < -1.0, dy > 1.0
    lc_state[mask_l & (nb_vy_abs > args.vy_eps)] = -1.0
    lc_state[mask_l & (nb_vy_abs < -args.vy_eps)] = -3.0
    lc_state[mask_r & (nb_vy_abs < -args.vy_eps)] = 1.0
    lc_state[mask_r & (nb_vy_abs > args.vy_eps)] = 3.0

    # dx_time & gate (게이팅 로직)
    denom = dvx.copy()
    denom[dvx >= 0] += args.eps_gate
    denom[dvx < 0] -= args.eps_gate
    dx_time = dx / denom
    gate = np.zeros_like(dx_time)
    gate[(-args.t_back < dx_time) & (dx_time < args.t_front)] = 1.0

    return np.stack([dx, dy, dvx, dvy, rel_acc[:,0], rel_acc[:,1], lc_state, dx_time, gate], axis=-1)

def balanced_recording_split(ds_counts, ratios=(0.7, 0.1, 0.2), seed=42):
    """레코딩별 샘플 수 기반 7:1:2 분할 로직"""
    rng = np.random.default_rng(seed)
    total_samples = sum(ds_counts.values())
    targets = [total_samples * r for r in ratios]
    items = sorted(ds_counts.items(), key=lambda x: x[1], reverse=True)
    rng.shuffle(items)

    splits = {"train": [], "val": [], "test": []}
    sums = {"train": 0, "val": 0, "test": 0}
    keys = ["train", "val", "test"]
    for rec_id, cnt in items:
        deficits = {k: (targets[i] - sums[k]) for i, k in enumerate(keys)}
        best = max(deficits.items(), key=lambda kv: kv[1])[0]
        splits[best].append(rec_id)
        sums[best] += cnt
    return splits

# ==============================================================================
# 3. 메인 프로세싱 함수
# ==============================================================================
def process_recording(rec_id, raw_dir, args):
    df = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")
    tmeta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    rmeta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    
    # 좌표 보정 (Center) 및 방향 통일 (Flip)
    df["x"] = df["x"] + df["width"] / 2.0
    df["y"] = df["y"] + df["height"] / 2.0
    df = df.merge(tmeta[["id", "drivingDirection"]], on="id")
    
    if args.normalize_flip:
        upper_mask = df["drivingDirection"] == 1
        up_m = [float(f) for f in str(rmeta.loc[0, "upperLaneMarkings"]).split(";") if f]
        lo_m = [float(f) for f in str(rmeta.loc[0, "lowerLaneMarkings"]).split(";") if f]
        C_y = up_m[-1] + lo_m[0]
        x_max = df["x"].max()
        for col, sign in zip(["x", "xVelocity", "xAcceleration"], [-1, -1, -1]):
            if col == "x": df.loc[upper_mask, col] = x_max - df.loc[upper_mask, col]
            else: df.loc[upper_mask, col] *= sign
        df.loc[upper_mask, "y"] = C_y - df.loc[upper_mask, "y"]
        df.loc[upper_mask, "yVelocity"] *= -1
        df.loc[upper_mask, "yAcceleration"] *= -1

    raw_fps = rmeta.loc[0, "frameRate"]
    stride = int(round(raw_fps / TARGET_HZ))
    df = df[df["frame"] % stride == 0].sort_values(["id", "frame"])

    selected_indices = EXTRA_FEATURE_MAP[args.feature_mode]
    num_c = len(selected_indices)
    
    samples = []
    agents = {vid: g[["frame", "x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values 
              for vid, g in df.groupby("id")}
    neighbor_cols = ["precedingId", "followingId", "leftPrecedingId", "leftAlongsideId", "leftFollowingId", "rightPrecedingId", "rightAlongsideId", "rightFollowingId"]
    
    for vid, data in agents.items():
        if len(data) < T_H + T_F: continue
        for i in range(0, len(data) - (T_H + T_F) + 1, int(TARGET_HZ)):
            window = data[i : i + T_H + T_F]
            ego_hist = window[:T_H]
            obs_frame = ego_hist[-1, 0]
            
            # 인접 행렬 구성을 위한 이웃 ID 확보
            nbr_ids = df[(df["id"] == vid) & (df["frame"] == obs_frame)][neighbor_cols].values.flatten()
            
            # 입력 텐서 (9, 15, num_c) 및 인접 행렬 (9, 9)
            tensor = np.zeros((1 + MAX_NEIGHBORS, T_H, num_c), dtype=np.float32)
            adj = np.eye(1 + MAX_NEIGHBORS, dtype=np.float32)
            
            # Ego 피처
            ego_xy = ego_hist[:, 1:3] 
            if num_c >= 2: tensor[0, :, :2] = ego_xy
            
            #ego_vx_vy = ego_hist[:, 3:5]
            #if num_c >= 2: tensor[0, :, :2] = ego_vx_vy
            
            for nb_idx, nb_id in enumerate(nbr_ids):
                if nb_id <= 0 or nb_id not in agents: continue
                nb_data = agents[nb_id]
                nb_win = nb_data[np.isin(nb_data[:, 0], ego_hist[:, 0])]
                if len(nb_win) < T_H: continue
                
                # 피처 계산 및 선택된 인덱스 추출
                all_nb_feats = get_neighbor_features(ego_hist, nb_win, args)
                tensor[nb_idx + 1, :, :] = all_nb_feats[:, selected_indices]
                adj[0, nb_idx + 1] = adj[nb_idx + 1, 0] = 1 # 상호작용 연결
            
            samples.append({"input": tensor, "adj": adj, "target": window[T_H:, 1:3]})
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="highD/raw")
    parser.add_argument("--out_dir", type=str, default="highD")
    parser.add_argument("--feature_mode", type=str, default="baseline", choices=EXTRA_FEATURE_MAP.keys())
    parser.add_argument("--normalize_flip", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    # Gating params
    parser.add_argument("--t_front", type=float, default=3.0)
    parser.add_argument("--t_back", type=float, default=5.0)
    parser.add_argument("--vy_eps", type=float, default=0.27)
    parser.add_argument("--eps_gate", type=float, default=0.1)
    
    args = parser.parse_args()
    raw_path = Path(args.raw_dir)
    rec_ids = sorted(set([f.name.split("_")[0] for f in raw_path.glob("*_tracks.csv")]))
    
    # 1. 전처리 및 샘플 수 카운트
    all_rec_samples = {}
    for rid in tqdm(rec_ids, desc="Preprocessing"):
        samples = process_recording(rid, raw_path, args)
        if samples: all_rec_samples[rid] = samples

    # 2. 7:1:2 분할
    rec_counts = {rid: len(s) for rid, s in all_rec_samples.items()}
    splits = balanced_recording_split(rec_counts, seed=args.seed)

    # 3. HDF5 저장
    out_dir = Path(args.out_dir) / args.feature_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_rec_ids in splits.items():
        split_data = [s for rid in split_rec_ids for s in all_rec_samples[rid]]
        with h5py.File(out_dir / f"{split_name}.h5", 'w') as f:
            f.create_dataset("input", data=np.array([s["input"] for s in split_data]), compression="gzip")
            f.create_dataset("adj", data=np.array([s["adj"] for s in split_data]), compression="gzip")
            f.create_dataset("target", data=np.array([s["target"] for s in split_data]), compression="gzip")
        print(f"-> {split_name}.h5 saved ({len(split_data)} samples)")

if __name__ == "__main__":
    main()