#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scenario_labels.py  —  GRIP용 scenario_labels.csv 생성 스크립트

h5 데이터 파일에서 meta 키(recordingId, trackId, t0_frame)를 읽고,
../neighformer/data/highD/scenario_label.py 와 동일한 로직으로 라벨링한다.

Event labels (3-class):
    cut_in          : LC 발생 + 목표 차선 측 rear/alongside 차량 있음
    lane_change     : LC 발생 + 목표 차선 측 rear/alongside 차량 없음 (공백으로 진입)
    lane_following  : 윈도우 내 LC 없음

State labels (2-class):
    dense     : 인접 전방/alongside 슬롯 점유율 <= 0.40
    free_flow : 점유율 > 0.40
    (STATE_SLOTS = [0,2,3,5,6] → preceding, leftPreceding, leftAlongside,
                                  rightPreceding, rightAlongside)

Usage:
    # 단일 h5 파일
    python highD/generate_scenario_labels.py \\
        --h5      highD/baseline/test.h5 \\
        --raw_dir highD/raw \\
        --out_csv highD/baseline/scenario_labels.csv

    # 여러 split을 한 번에 (test + val 합쳐서 하나의 CSV)
    python highD/generate_scenario_labels.py \\
        --h5      highD/baseline/test.h5 highD/baseline/val.h5 \\
        --raw_dir highD/raw \\
        --out_csv highD/baseline/scenario_labels.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# preprocess.py와 동기화: TARGET_HZ=3.0, T_H=6(2sec), T_F=15(5sec)
TARGET_HZ = 3.0
T_H       = 6
T_F       = 15


# ─────────────────────────────────────────────────────────────────────────────
# State label constants  (neighformer와 동일)
# ─────────────────────────────────────────────────────────────────────────────

# NEIGHBOR_COLS_8 순서:
#  0: preceding      1: following
#  2: leftPreceding  3: leftAlongside  4: leftFollowing
#  5: rightPreceding 6: rightAlongside 7: rightFollowing
STATE_SLOTS     = [0, 2, 3, 5, 6]   # front + adj lead/alongside (no rear)
STATE_THRESHOLD = 0.40


def compute_state_labels_from_adj(adj_arr: np.ndarray) -> np.ndarray:
    """
    adj 행렬에서 state label을 계산한다.

    Parameters
    ----------
    adj_arr : (N, 9, 9) — GRIP h5의 adj 데이터셋
        adj[i, 0, k+1] > 0  이면 neighbor k가 윈도우에 존재.

    Returns
    -------
    state_labels : (N,) str array  —  'dense' or 'free_flow'
    occ          : (N,) float32
    """
    # adj[n, 0, k+1] — ego(node 0)와 neighbor k(node k+1)의 연결 여부
    # STATE_SLOTS [0,2,3,5,6] → adj 노드 인덱스 [1,3,4,6,7]
    state_nodes = [s + 1 for s in STATE_SLOTS]          # [1, 3, 4, 6, 7]
    presence = (adj_arr[:, 0, state_nodes] > 0).astype(np.float32)  # (N, 5)
    occ = presence.mean(axis=1)                          # (N,)
    labels = np.where(occ <= STATE_THRESHOLD, "dense", "free_flow")
    return labels, occ


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers  (neighformer와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def smart_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization  (neighformer와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_tracks(tracks: pd.DataFrame, xx: str) -> pd.DataFrame:
    df = tracks.copy()
    df.columns = [c.strip() for c in df.columns]
    df["recordingId"] = int(xx)
    if "trackId" not in df.columns:
        df["trackId"] = df["id"]
    if "latVelocity" not in df.columns:
        df["latVelocity"] = df.get("yVelocity", np.nan)
    if "leadId" not in df.columns:
        df["leadId"] = df.get("precedingId", 0)
    if "rearId" not in df.columns:
        df["rearId"] = df.get("followingId", 0)
    rename = {
        "leftPrecedingId":   "leftLeadId",
        "leftFollowingId":   "leftRearId",
        "rightPrecedingId":  "rightLeadId",
        "rightFollowingId":  "rightRearId",
        "rightAlsongsideId": "rightAlongsideId",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    for col in ["leftLeadId", "leftRearId", "leftAlongsideId",
                "rightLeadId", "rightRearId", "rightAlongsideId"]:
        if col not in df.columns:
            df[col] = 0
    for col in ["frame", "laneId", "trackId", "recordingId"]:
        if col not in df.columns:
            raise KeyError(f"normalize_tracks: missing required column '{col}'")
    return df


def normalize_recmeta(recmeta: pd.DataFrame, xx: str) -> pd.DataFrame:
    rm = recmeta.copy()
    rm.columns = [c.strip() for c in rm.columns]
    if "recordingId" not in rm.columns:
        rm["recordingId"] = rm["id"] if "id" in rm.columns else int(xx)
    rm["recordingId"] = rm["recordingId"].astype(int)
    if "frameRate" not in rm.columns:
        raise KeyError("recordingMeta missing 'frameRate'")
    return rm[["recordingId", "frameRate"]].drop_duplicates()


# ─────────────────────────────────────────────────────────────────────────────
# Lane lookup  (neighformer와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def build_lane_lookup(df: pd.DataFrame) -> Dict[int, Dict[int, int]]:
    tmp = df[["trackId", "frame", "laneId"]].copy()
    tmp["trackId"] = pd.to_numeric(tmp["trackId"], errors="coerce").astype(int)
    tmp["frame"]   = pd.to_numeric(tmp["frame"],   errors="coerce").astype(int)
    tmp["laneId"]  = pd.to_numeric(tmp["laneId"],  errors="coerce")
    lookup: Dict[int, Dict[int, int]] = {}
    for tid, g in tmp.groupby("trackId", sort=False):
        d: Dict[int, int] = {}
        for f, l in zip(g["frame"].to_numpy(), g["laneId"].to_numpy()):
            if not np.isnan(l):
                d[int(f)] = int(l)
        lookup[int(tid)] = d
    return lookup


def get_lane_at(lookup, track_id: int, frame: int) -> Optional[int]:
    d = lookup.get(int(track_id))
    if d is None:
        return None
    return d.get(int(frame))


def is_adjacent_lane(ego_lane: int, nb_lane: int) -> bool:
    return abs(int(ego_lane) - int(nb_lane)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Lane change detection  (neighformer와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def detect_lane_change(w: pd.DataFrame) -> Tuple[bool, int, Optional[int]]:
    if "laneId" not in w.columns or "frame" not in w.columns:
        return False, 0, None
    df = w[["frame", "laneId"]].copy()
    df["frame"]  = pd.to_numeric(df["frame"],  errors="coerce").astype("Int64")
    df["laneId"] = pd.to_numeric(df["laneId"], errors="coerce")
    df = df.dropna().sort_values("frame")
    if len(df) < 2:
        return False, 0, None
    lane   = df["laneId"].to_numpy()
    frames = df["frame"].to_numpy()
    changed = lane[1:] != lane[:-1]
    count   = int(changed.sum())
    if count == 0:
        return False, 0, None
    first_frame = int(frames[1:][changed][0])
    return True, count, first_frame


def infer_lc_direction(w: pd.DataFrame, lc_frame: int, K: int = 5) -> Optional[str]:
    vcol = "yVelocity" if "yVelocity" in w.columns else "latVelocity"
    if vcol not in w.columns:
        return None
    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    win = df[(df["frame"] >= lc_frame - K) & (df["frame"] <= lc_frame + K)]
    v = pd.to_numeric(win[vcol], errors="coerce").dropna()
    if len(v) == 0:
        return None
    mv = float(v.mean())
    if mv > 0:
        return "right"
    if mv < 0:
        return "left"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Adjacent presence check  (neighformer와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def _to_int_id(x) -> Optional[int]:
    try:
        v = int(float(x))
        return None if v in (-1, 0) else v
    except Exception:
        return None


def check_adjacent_rear_or_alongside(
    w: pd.DataFrame,
    lc_frame: int,
    direction: str,
    lookup: Dict[int, Dict[int, int]],
    W: int = 10,
) -> bool:
    if direction == "left":
        rear_col      = "leftRearId"
        alongside_col = "leftAlongsideId"
    else:
        rear_col      = "rightRearId"
        alongside_col = "rightAlongsideId"
    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    pre = df[(df["frame"] >= lc_frame - W) & (df["frame"] < lc_frame)]
    if len(pre) == 0:
        return False
    for _, row in pre.iterrows():
        f        = int(row["frame"])
        ego_lane = row.get("laneId", np.nan)
        if pd.isna(ego_lane):
            continue
        ego_lane = int(ego_lane)
        for col in (rear_col, alongside_col):
            if col not in pre.columns:
                continue
            nb_id = _to_int_id(row.get(col, 0))
            if nb_id is None:
                continue
            nb_lane = get_lane_at(lookup, nb_id, f)
            if nb_lane is None:
                continue
            if is_adjacent_lane(ego_lane, nb_lane):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Window-level labeling  (neighformer와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def label_window(
    w: pd.DataFrame,
    lookup: Dict[int, Dict[int, int]],
    W_adj: int = 25,
    lc_direction_K: int = 5,
) -> Dict:
    out: Dict = {}
    has_lc, lc_count, lc_frame = detect_lane_change(w)
    out["lc_count"] = lc_count
    out["lc_frame"] = int(lc_frame) if lc_frame is not None else -1

    if not has_lc or lc_frame is None:
        out["event_label"]               = "lane_following"
        out["lc_direction"]              = "none"
        out["has_adj_rear_or_alongside"] = False
        return out

    direction = infer_lc_direction(w, lc_frame=lc_frame, K=lc_direction_K)
    out["lc_direction"] = direction if direction is not None else "unknown"

    if direction is None:
        out["event_label"]               = "lane_change"
        out["has_adj_rear_or_alongside"] = False
        return out

    has_adj = check_adjacent_rear_or_alongside(
        w=w, lc_frame=lc_frame, direction=direction, lookup=lookup, W=W_adj,
    )
    out["has_adj_rear_or_alongside"] = bool(has_adj)
    out["event_label"] = "cut_in" if has_adj else "lane_change"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording processing
# ─────────────────────────────────────────────────────────────────────────────

def label_recording(
    xx: str,
    raw_dir: Path,
    keys: List[Tuple[int, int]],   # [(trackId, t0_frame), ...]
    history_sec: float,
    future_sec: float,
    target_hz: float,
    W_adj: int,
) -> List[Dict]:
    tracks_path  = raw_dir / f"{xx}_tracks.csv"
    recmeta_path = raw_dir / f"{xx}_recordingMeta.csv"

    if not tracks_path.exists() or not recmeta_path.exists():
        print(f"  [SKIP] {xx}: raw files not found")
        return []

    tracks  = smart_read_csv(tracks_path)
    recmeta = smart_read_csv(recmeta_path)
    tracks.columns  = [c.strip() for c in tracks.columns]
    recmeta.columns = [c.strip() for c in recmeta.columns]

    tracks_n  = normalize_tracks(tracks, xx)
    recmeta_n = normalize_recmeta(recmeta, xx)

    fr = float(recmeta_n["frameRate"].iloc[0])
    if not np.isfinite(fr) or fr <= 0:
        print(f"  [SKIP] {xx}: invalid frameRate={fr}")
        return []

    tracks_n = tracks_n.merge(recmeta_n, on="recordingId", how="left")
    tracks_n["frame"] = pd.to_numeric(tracks_n["frame"], errors="coerce").astype("Int64")
    tracks_n = tracks_n.dropna(subset=["frame"]).sort_values(["trackId", "frame"])

    ds_step    = max(1, int(round(fr / target_hz)))
    T          = int(round(history_sec * target_hz))
    Tf         = int(round(future_sec  * target_hz))
    win_native = (T + Tf - 1) * ds_step   # inclusive frame span

    lookup  = build_lane_lookup(tracks_n)
    by_tid  = {int(tid): g for tid, g in tracks_n.groupby("trackId", sort=False)}
    rid     = int(xx)
    rows: List[Dict] = []

    for tid, t0_frame in sorted(set(keys)):
        g = by_tid.get(int(tid))
        base = {
            "recordingId": rid,
            "trackId":     int(tid),
            "t0_frame":    int(t0_frame),
        }
        if g is None:
            rows.append({**base, "event_label": "unknown",
                         "lc_frame": -1, "lc_count": 0,
                         "lc_direction": "unknown",
                         "has_adj_rear_or_alongside": False})
            continue

        t1_frame = int(t0_frame) + win_native
        w = g[(g["frame"] >= t0_frame) & (g["frame"] <= t1_frame)]

        if len(w) == 0:
            rows.append({**base, "event_label": "unknown",
                         "lc_frame": -1, "lc_count": 0,
                         "lc_direction": "unknown",
                         "has_adj_rear_or_alongside": False})
            continue

        result = label_window(w, lookup=lookup, W_adj=W_adj)
        rows.append({**base, **result})

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_recording(
    args_tuple: Tuple,
) -> List[Dict]:
    xx, keys, raw_dir, history_sec, future_sec, target_hz, W_adj = args_tuple
    return label_recording(
        xx=xx, raw_dir=raw_dir, keys=keys,
        history_sec=history_sec, future_sec=future_sec,
        target_hz=target_hz, W_adj=W_adj,
    )


# ─────────────────────────────────────────────────────────────────────────────
# H5 reading
# ─────────────────────────────────────────────────────────────────────────────

def read_h5_meta(h5_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    여러 h5 파일에서 meta 및 adj를 읽어 합친다.

    Returns
    -------
    rids, tids, t0s : (N,) int32
    adjs            : (N, 9, 9) float32
    """
    rids_list, tids_list, t0s_list, adjs_list = [], [], [], []
    for p in h5_paths:
        if not p.exists():
            print(f"[WARN] h5 not found: {p}")
            continue
        with h5py.File(p, "r") as f:
            if "meta_recordingId" not in f:
                print(f"[WARN] {p}: meta_recordingId not found — skip")
                continue
            rids_list.append(f["meta_recordingId"][:])
            tids_list.append(f["meta_trackId"][:])
            t0s_list.append(f["meta_t0_frame"][:])
            adjs_list.append(f["adj"][:].astype(np.float32))
        print(f"  Loaded {p.name}: {len(rids_list[-1]):,} samples")

    if not rids_list:
        raise RuntimeError("No valid h5 files found")

    rids = np.concatenate(rids_list).astype(np.int32)
    tids = np.concatenate(tids_list).astype(np.int32)
    t0s  = np.concatenate(t0s_list).astype(np.int32)
    adjs = np.concatenate(adjs_list, axis=0)
    return rids, tids, t0s, adjs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="GRIP highD scenario label generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--h5", nargs="+", required=True,
        help="h5 파일 경로 (test.h5, val.h5 등 여러 개 가능)",
    )
    ap.add_argument(
        "--raw_dir", default="highD/raw",
        help="highD raw CSV 디렉토리 (XX_tracks.csv, XX_recordingMeta.csv 포함)",
    )
    ap.add_argument(
        "--out_csv", default=None,
        help="출력 CSV 경로. 미지정시 첫 번째 h5와 같은 디렉토리에 scenario_labels.csv 저장",
    )
    ap.add_argument("--history_sec", type=float, default=2.0,
                    help=f"히스토리 길이 (초). preprocess T_H={T_H} frames @ {TARGET_HZ}Hz")
    ap.add_argument("--future_sec",  type=float, default=5.0,
                    help=f"예측 길이 (초). preprocess T_F={T_F} frames @ {TARGET_HZ}Hz")
    ap.add_argument("--target_hz",   type=float, default=TARGET_HZ,
                    help="다운샘플링 Hz (preprocess와 동일하게 유지)")
    ap.add_argument("--W_adj",       type=int,   default=25,
                    help="LC 직전 look-back 윈도우 (native frames). 25 ≈ 1sec @ 25fps")
    ap.add_argument("--num_workers", type=int,   default=0,
                    help="병렬 프로세스 수 (0 = os.cpu_count())")
    return ap.parse_args()


def main() -> None:
    args    = parse_args()
    raw_dir = Path(args.raw_dir)
    h5_paths = [Path(p) for p in args.h5]

    if args.out_csv is None:
        out_csv = h5_paths[0].parent / "scenario_labels.csv"
    else:
        out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── H5에서 meta 및 adj 읽기 ───────────────────────────────────────────────
    print(f"[Step 1] h5 파일 읽는 중...")
    rids, tids, t0s, adjs = read_h5_meta(h5_paths)
    n_total = len(rids)
    print(f"  총 {n_total:,} 샘플")

    # ── State labels (adj 기반) ───────────────────────────────────────────────
    print(f"\n[Step 2] State label 계산 중 (adj 기반)...")
    state_labels, state_occ = compute_state_labels_from_adj(adjs)
    print(f"  dense={( state_labels=='dense').mean()*100:.1f}%  "
          f"free_flow={(state_labels=='free_flow').mean()*100:.1f}%")

    # ── recording별로 키 묶기 ─────────────────────────────────────────────────
    keys_by_xx: Dict[str, List[Tuple[int, int, int]]] = {}  # xx -> [(tid, t0, global_idx), ...]
    for i, (rid, tid, t0) in enumerate(zip(rids, tids, t0s)):
        xx = f"{int(rid):02d}"
        keys_by_xx.setdefault(xx, []).append((int(tid), int(t0), i))

    print(f"\n[Step 3] Event label 계산 중 ({len(keys_by_xx)} recordings)...")

    # ── 병렬 처리 ─────────────────────────────────────────────────────────────
    n_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()

    work_items = [
        (xx, [(tid, t0) for tid, t0, _ in key_list],
         raw_dir, args.history_sec, args.future_sec, args.target_hz, args.W_adj)
        for xx, key_list in sorted(keys_by_xx.items())
    ]

    # global_idx 매핑: (xx, tid, t0) → global_idx
    global_idx_map: Dict[Tuple[str, int, int], int] = {}
    for xx, key_list in keys_by_xx.items():
        for tid, t0, gi in key_list:
            global_idx_map[(xx, tid, t0)] = gi

    all_rows: List[Dict] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_process_one_recording, item): item[0]
                for item in work_items}
        for fut in tqdm(concurrent.futures.as_completed(futs),
                        total=len(futs), desc="Labeling"):
            rows = fut.result()
            all_rows.extend(rows)

    # ── state_label 조인 ──────────────────────────────────────────────────────
    for row in all_rows:
        xx  = f"{row['recordingId']:02d}"
        gi  = global_idx_map.get((xx, row["trackId"], row["t0_frame"]))
        if gi is not None:
            row["state_label"] = state_labels[gi]
            row["occupancy"]   = float(state_occ[gi])
        else:
            row["state_label"] = "unknown"
            row["occupancy"]   = float("nan")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["recordingId", "trackId", "t0_frame"])

    col_order = [
        "recordingId", "trackId", "t0_frame",
        "event_label", "state_label", "occupancy",
        "lc_frame", "lc_count", "lc_direction", "has_adj_rear_or_alongside",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] {len(df):,} labels → {out_csv}")

    if len(df) != n_total:
        print(f"[WARN] h5 samples={n_total}, labeled={len(df)} "
              f"(delta={n_total - len(df)})")

    if "event_label" in df.columns:
        print("\nEvent label counts:")
        print(df["event_label"].value_counts().to_string())

    if "state_label" in df.columns:
        print("\nState label counts:")
        print(df["state_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
