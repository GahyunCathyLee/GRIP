import os
import pandas as pd
import numpy as np
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. 실험 설정 및 인덱스 정의
# ==============================================================================
TARGET_HZ = 5.0   # GRIP++ 기준 5Hz
T_H = 15          # 3초 관측 (15 steps)
T_F = 25          # 5초 예측 (25 steps)
MAX_NEIGHBORS = 8 # highD 슬롯 기반 8대

# neighbor 후보 피처 인덱스
# [0:dx, 1:dy, 2:dvx, 3:dvy, 4:dax, 5:day, 6:lc_state, 7:dx_time, 8:gate]
EXTRA_FEATURE_MAP = {
    'baseline':   [0, 1],
    'baseline_v': [2, 3],
    'exp1':       [0, 1, 8],
    'exp2':       [0, 1, 6, 7],
    'exp3':       [6, 7],
    'exp4':       [4, 5, 6, 7, 8],
    'exp5':       [0, 1, 2, 3, 4, 5, 8],
    'exp6':       [0, 1, 2, 3, 4, 5, 6, 7, 8],
}

# ==============================================================================
# 채널 레이아웃 (feature_mode에 따라 동적으로 결정)
#
#  ch 0        : ego_vx        (ego 전용, neighbor는 0)
#  ch 1        : ego_vy        (ego 전용, neighbor는 0)
#  ch 2 ~ 2+N-1: neighbor 피처 (neighbor 전용, ego는 0)
#  ch 2+N      : is_ego mask   (ego=1.0, neighbor=0.0)
#
#  → 모든 피처가 0이 될 수 있으므로, is_ego 채널로만 ego/neighbor를 명확히 구분
# ==============================================================================

def get_num_channels(feature_mode):
    """총 채널 수: ego(vx,vy) + neighbor_feats + is_ego_mask"""
    nb_feats = len(EXTRA_FEATURE_MAP[feature_mode])
    return 2 + nb_feats + 1  # ego_vel(2) + nb_feats(N) + is_ego(1)

# ==============================================================================
# 2. 유틸리티 함수
# ==============================================================================

def process_wrapper(args_tuple):
    rid, raw_path, args = args_tuple
    samples = process_recording(rid, raw_path, args)
    return rid, samples


def get_neighbor_features(ego_hist, nb_window, args):
    """
    주변 차량의 9가지 후보 피처 계산 (모두 ego 기준 상대값)
    반환: (T_H, 9) 배열
      [0:dx, 1:dy, 2:dvx, 3:dvy, 4:dax, 5:day, 6:lc_state, 7:dx_time, 8:gate]
    """
    rel_pos = nb_window[:, 1:3] - ego_hist[:, 1:3]  # dx, dy
    rel_vel = nb_window[:, 3:5] - ego_hist[:, 3:5]  # dvx, dvy
    rel_acc = nb_window[:, 5:7] - ego_hist[:, 5:7]  # dax, day

    dx,  dy  = rel_pos[:, 0], rel_pos[:, 1]
    dvx, dvy = rel_vel[:, 0], rel_vel[:, 1]

    # lc_state: neighbor의 차선 위치 + 횡방향 움직임
    # dy < -1 : neighbor가 ego 왼쪽 차선
    # dy >  1 : neighbor가 ego 오른쪽 차선
    # dy 사이  : ego와 같은 차선 → lc_state = 0
    lc_state = np.zeros_like(dy)
    nb_vy    = nb_window[:, 4]
    mask_l   = dy < -1.0
    mask_r   = dy >  1.0

    lc_state[mask_l & (nb_vy  > args.vy_eps)]            = -1.0  # 왼쪽→ego쪽 cut-in
    lc_state[mask_l & (nb_vy  < -args.vy_eps)]           = -3.0  # 왼쪽으로 이탈
    lc_state[mask_l & (np.abs(nb_vy) <= args.vy_eps)]    = -2.0  # 왼쪽 차선 유지

    lc_state[mask_r & (nb_vy  < -args.vy_eps)]           =  1.0  # 오른쪽→ego쪽 cut-in
    lc_state[mask_r & (nb_vy  >  args.vy_eps)]           =  3.0  # 오른쪽으로 이탈
    lc_state[mask_r & (np.abs(nb_vy) <= args.vy_eps)]    =  2.0  # 오른쪽 차선 유지

    # dx_time: 종방향 접근 시간 (dx / dvx), gate: 위험 범위 내 여부
    denom = dvx.copy()
    denom[dvx >= 0] += args.eps_gate
    denom[dvx  < 0] -= args.eps_gate
    dx_time = dx / denom
    gate = np.zeros_like(dx_time)
    gate[(-args.t_back < dx_time) & (dx_time < args.t_front)] = 1.0

    return np.stack(
        [dx, dy, dvx, dvy, rel_acc[:, 0], rel_acc[:, 1], lc_state, dx_time, gate],
        axis=-1
    )  # (T_H, 9)


def balanced_recording_split(ds_counts, ratios=(0.7, 0.1, 0.2), seed=42):
    """레코딩별 샘플 수 기반 7:1:2 분할"""
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
# 3. 메인 프로세싱 함수
# ==============================================================================

def process_recording(rec_id, raw_dir, args):
    df    = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")
    tmeta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    rmeta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")

    # 좌표 보정: bounding box 좌상단 → 중심점
    df["x"] = df["x"] + df["width"]  / 2.0
    df["y"] = df["y"] + df["height"] / 2.0
    df = df.merge(tmeta[["id", "drivingDirection"]], on="id")

    # 방향 통일: 상행선(drivingDirection==1)을 하행선 기준으로 flip
    if args.normalize_flip:
        upper_mask = df["drivingDirection"] == 1
        up_m  = [float(f) for f in str(rmeta.loc[0, "upperLaneMarkings"]).split(";") if f]
        lo_m  = [float(f) for f in str(rmeta.loc[0, "lowerLaneMarkings"]).split(";") if f]
        C_y   = up_m[-1] + lo_m[0]
        x_max = df["x"].max()
        df.loc[upper_mask, "x"]            = x_max - df.loc[upper_mask, "x"]
        df.loc[upper_mask, "xVelocity"]   *= -1
        df.loc[upper_mask, "xAcceleration"] *= -1
        df.loc[upper_mask, "y"]            = C_y - df.loc[upper_mask, "y"]
        df.loc[upper_mask, "yVelocity"]   *= -1
        df.loc[upper_mask, "yAcceleration"] *= -1

    # 5Hz 다운샘플링
    raw_fps = rmeta.loc[0, "frameRate"]
    stride  = int(round(raw_fps / TARGET_HZ))
    df = df[df["frame"] % stride == 0].sort_values(["id", "frame"])

    # 채널 레이아웃 결정
    selected_indices = EXTRA_FEATURE_MAP[args.feature_mode]
    nb_feat_dim = len(selected_indices)
    num_c       = get_num_channels(args.feature_mode)  # 2 + nb_feat_dim + 1
    ego_vel_ch  = slice(0, 2)           # ch 0,1 : ego vx, vy
    nb_feat_ch  = slice(2, 2+nb_feat_dim)  # ch 2~  : neighbor 피처
    is_ego_ch   = 2 + nb_feat_dim       # ch -1  : is_ego mask

    neighbor_cols = [
        "precedingId", "followingId",
        "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
        "rightPrecedingId", "rightAlongsideId", "rightFollowingId"
    ]

    agents = {
        vid: g[["frame", "x", "y", "xVelocity", "yVelocity",
                "xAcceleration", "yAcceleration"]].values
        for vid, g in df.groupby("id")
    }

    samples = []
    for vid, data in agents.items():
        if len(data) < T_H + T_F:
            continue

        for i in range(0, len(data) - (T_H + T_F) + 1, int(TARGET_HZ)):
            window    = data[i : i + T_H + T_F]
            ego_hist  = window[:T_H]
            obs_frame = ego_hist[-1, 0]

            # 이웃 ID 확보
            nbr_ids = df[
                (df["id"] == vid) & (df["frame"] == obs_frame)
            ][neighbor_cols].values.flatten()

            # 텐서 초기화: (1+MAX_NEIGHBORS, T_H, num_c)
            tensor = np.zeros((1 + MAX_NEIGHBORS, T_H, num_c), dtype=np.float32)
            adj    = np.eye(1 + MAX_NEIGHBORS, dtype=np.float32)

            # ── Ego 노드 (ch 0,1: vx,vy  /  nb_feat 채널: 0  /  is_ego: 1) ──
            norm_center         = ego_hist[-1, 1:3]          # 마지막 관측 위치 (정규화 기준)
            tensor[0, :, ego_vel_ch] = ego_hist[:, 3:5]      # vx, vy
            tensor[0, :, is_ego_ch]  = 1.0                   # is_ego = True

            # ── Neighbor 노드 (ego_vel 채널: 0  /  nb_feat 채널: 실제값  /  is_ego: 0) ──
            for nb_idx, nb_id in enumerate(nbr_ids):
                if nb_id <= 0 or nb_id not in agents:
                    continue
                nb_data = agents[nb_id]
                nb_win  = nb_data[np.isin(nb_data[:, 0], ego_hist[:, 0])]
                if len(nb_win) < T_H:
                    continue

                all_nb_feats = get_neighbor_features(ego_hist, nb_win, args)  # (T_H, 9)
                tensor[nb_idx + 1, :, nb_feat_ch] = all_nb_feats[:, selected_indices]
                # is_ego_ch는 0 (기본값 유지)

                adj[0, nb_idx + 1] = adj[nb_idx + 1, 0] = 1.0

            samples.append({
                "input":  tensor,
                "adj":    adj,
                "target": window[T_H:, 1:3] - norm_center   # ego 중심 상대 좌표
            })

    return samples


# ==============================================================================
# 4. 메인
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",        type=str,   default="highD/raw")
    parser.add_argument("--out_dir",        type=str,   default="highD")
    parser.add_argument("--feature_mode",   type=str,   default="baseline",
                        choices=EXTRA_FEATURE_MAP.keys())
    parser.add_argument("--normalize_flip", action="store_true", default=True)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--t_front",        type=float, default=3.0)
    parser.add_argument("--t_back",         type=float, default=5.0)
    parser.add_argument("--vy_eps",         type=float, default=0.27)
    parser.add_argument("--eps_gate",       type=float, default=0.1)
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    rec_ids  = sorted(set([
        f.name.split("_")[0] for f in raw_path.glob("*_tracks.csv")
    ]))

    num_c = get_num_channels(args.feature_mode)
    nb_feat_dim = len(EXTRA_FEATURE_MAP[args.feature_mode])
    print(f"Found {len(rec_ids)} recordings")
    print(f"Feature mode : {args.feature_mode}")
    print(f"Channel layout: [ego_vx, ego_vy | {nb_feat_dim} nb_feats | is_ego]  →  total {num_c}ch")
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
        split_data = [s for rid in split_rec_ids for s in all_rec_samples[rid]]
        with h5py.File(out_dir / f"{split_name}.h5", "w") as f:
            f.create_dataset("input",  data=np.array([s["input"]  for s in split_data]), compression="gzip")
            f.create_dataset("adj",    data=np.array([s["adj"]    for s in split_data]), compression="gzip")
            f.create_dataset("target", data=np.array([s["target"] for s in split_data]), compression="gzip")
        print(f"-> {split_name}.h5 saved ({len(split_data)} samples, input shape: {split_data[0]['input'].shape})")


if __name__ == "__main__":
    main()