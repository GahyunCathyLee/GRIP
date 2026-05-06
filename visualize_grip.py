#!/usr/bin/env python3
"""
Visualize GRIP predictions for one highD/exiD sample.

The script takes one or two checkpoints, picks a sample by meta id or scenario,
reconstructs the sample geometry from raw CSVs, and plots:
  - ego history and GT future
  - model prediction(s)
  - neighbor boxes at every history timestep
  - highD lane markings or exiD lanelet map lines
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from highD.preprocess import MAX_NEIGHBORS, NEIGHBOR_COLS_8 as HIGHD_NB_COLS
from highD.preprocess import TARGET_HZ, T_F, T_H, get_num_channels
from layers.graph import Graph
from model import Model


EXID_NB_COLS = [
    "leadId", "rearId",
    "leftLeadId", "leftAlongsideId", "leftRearId",
    "rightLeadId", "rightAlongsideId", "rightRearId",
]

SLOT_NAMES = [
    "preceding", "following", "leftPreceding", "leftAlongside",
    "leftFollowing", "rightPreceding", "rightAlongside", "rightFollowing",
]


@dataclass
class RawScene:
    dataset: str
    rec_id: int
    frame_rate: float
    tracks: pd.DataFrame
    recmeta: pd.DataFrame
    tmeta: pd.DataFrame
    nb_cols: List[str]
    normalize_heading: bool = False
    ref_hdg: float = 0.0
    norm_center: Tuple[float, float] = (0.0, 0.0)


def read_csv_smart(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


def parse_float_list(value: Any) -> List[float]:
    if not isinstance(value, str):
        return []
    return [float(x) for x in value.split(";") if x.strip()]


def to_int_id(value: Any) -> int:
    try:
        if pd.isna(value):
            return -1
        return int(float(value))
    except Exception:
        return -1


def rot2d(vx: float, vy: float, h_rad: float) -> Tuple[float, float]:
    c, s = math.cos(h_rad), math.sin(h_rad)
    return c * vx + s * vy, -s * vx + c * vy


def norm_pos(gx: float, gy: float, ref_x: float, ref_y: float, ref_hdg_rad: float) -> Tuple[float, float]:
    return rot2d(gx - ref_x, gy - ref_y, ref_hdg_rad)


def denorm_pos(lx: float, ly: float, ref_x: float, ref_y: float, ref_hdg_rad: float) -> Tuple[float, float]:
    c, s = math.cos(ref_hdg_rad), math.sin(ref_hdg_rad)
    return ref_x + c * lx - s * ly, ref_y + s * lx + c * ly


def infer_dataset(cfg: Dict[str, Any], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    base = str(cfg.get("data", {}).get("base_dir", "")).lower()
    if "exid" in base:
        return "exiD"
    return "highD"


def default_raw_dir(dataset: str) -> Path:
    if dataset == "exiD":
        return Path("exiD/raw")
    return Path("highD/raw")


def default_h5_path(cfg: Dict[str, Any], feature_mode: str, split: str) -> Path:
    base_dir = Path(cfg["data"]["base_dir"])
    return base_dir / feature_mode / f"{split}.h5"


def load_checkpoint_state(path: Path) -> Tuple[Dict[str, torch.Tensor], Optional[str]]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("feature_mode")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("feature_mode")
    if isinstance(ckpt, dict):
        return ckpt, ckpt.get("feature_mode")
    raise ValueError(f"Unsupported checkpoint format: {path}")


def build_model_from_cfg(cfg: Dict[str, Any], ckpt_path: Path, device: torch.device) -> Tuple[Model, str]:
    state, ckpt_feature = load_checkpoint_state(ckpt_path)
    feature_mode = ckpt_feature or cfg["exp"]["feature_mode"]
    model = Model(
        in_channels=get_num_channels(feature_mode),
        graph_args={
            "max_hop": int(cfg["model"]["max_hop"]),
            "num_node": int(cfg["model"]["num_node"]),
        },
        edge_importance_weighting=bool(cfg["model"]["edge_importance_weighting"]),
    ).to(device)
    if hasattr(model, "seq2seq"):
        model.seq2seq.isCuda = device.type == "cuda"
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, feature_mode


def load_h5_arrays(h5_path: Path) -> Dict[str, Any]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required to read GRIP .h5 files. Run this script in the same environment "
            "used for training/evaluation, or install h5py there."
        ) from exc

    with h5py.File(h5_path, "r") as f:
        out = {
            "meta_recordingId": f["meta_recordingId"][:],
            "meta_trackId": f["meta_trackId"][:],
            "meta_t0_frame": f["meta_t0_frame"][:],
            "target_shape": tuple(f["target"].shape),
            "path": h5_path,
            "attrs": dict(f.attrs),
        }
    return out


def read_h5_sample(h5: Dict[str, Any], idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required to read GRIP .h5 files. Run this script in the same environment "
            "used for training/evaluation, or install h5py there."
        ) from exc

    with h5py.File(h5["path"], "r") as f:
        input_arr = f["input"][idx:idx + 1]
        adj_arr = f["adj"][idx]
        pred_len = int(f["target"].shape[1])
    return input_arr, adj_arr, pred_len


def load_scenario_labels(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def find_sample_index(
    h5: Dict[str, Any],
    *,
    rec_id: Optional[int],
    track_id: Optional[int],
    t0_frame: Optional[int],
    labels: Optional[pd.DataFrame],
    event: Optional[str],
    state: Optional[str],
    scenario_rank: int,
) -> Tuple[int, str]:
    rids = h5["meta_recordingId"].astype(int)
    tids = h5["meta_trackId"].astype(int)
    t0s = h5["meta_t0_frame"].astype(int)
    mask = np.ones(len(rids), dtype=bool)
    desc: List[str] = []

    if rec_id is not None:
        mask &= rids == rec_id
        desc.append(f"recordingId={rec_id}")
    if track_id is not None:
        mask &= tids == track_id
        desc.append(f"trackId={track_id}")
    if t0_frame is not None:
        mask &= t0s == t0_frame
        desc.append(f"t0_frame={t0_frame}")

    exact = np.where(mask)[0]
    if len(exact) > 0 and not event and not state:
        return int(exact[0]), " / ".join(desc) if desc else "first valid sample"

    if labels is not None and (event or state):
        df = labels.copy()
        if rec_id is not None:
            df = df[df["recordingId"].astype(int) == rec_id]
        if track_id is not None:
            df = df[df["trackId"].astype(int) == track_id]
        if t0_frame is not None:
            df = df[df["t0_frame"].astype(int) == t0_frame]
        if event:
            df = df[df["event_label"].astype(str) == event]
        if state:
            df = df[df["state_label"].astype(str) == state]
        key_to_idx = {
            (int(r), int(t), int(f)): i
            for i, (r, t, f) in enumerate(zip(rids, tids, t0s))
        }
        matches: List[Tuple[int, Tuple[int, int, int]]] = []
        for row in df.itertuples(index=False):
            key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
            if key in key_to_idx:
                matches.append((int(key_to_idx[key]), key))
        if matches:
            if scenario_rank < 0 or scenario_rank >= len(matches):
                preview = "\n  ".join(
                    f"rank={rank} idx={idx} rec={key[0]} track={key[1]} t0={key[2]}"
                    for rank, (idx, key) in enumerate(matches[:10])
                )
                raise ValueError(
                    f"--scenario_rank={scenario_rank} is out of range for this filter "
                    f"({len(matches)} matches). First matches:\n  {preview}"
                )
            idx, key = matches[scenario_rank]
            return idx, f"scenario rank={scenario_rank}/{len(matches)} match {key}"

    if len(exact) > 0:
        return int(exact[0]), "meta match; scenario filter had no matching label"

    if track_id is not None:
        candidates = np.where((rids == rec_id if rec_id is not None else True) & (tids == track_id))[0]
        if len(candidates) > 0 and t0_frame is not None:
            nearest = candidates[np.argsort(np.abs(t0s[candidates] - t0_frame))[:5]]
            lines = [f"(idx={i}, rec={rids[i]}, track={tids[i]}, t0={t0s[i]})" for i in nearest]
            raise ValueError("Requested sample was not found. Nearby valid samples:\n  " + "\n  ".join(lines))

    candidates = np.arange(min(5, len(rids)))
    lines = [f"(idx={i}, rec={rids[i]}, track={tids[i]}, t0={t0s[i]})" for i in candidates]
    raise ValueError("Requested sample was not found. Try one of:\n  " + "\n  ".join(lines))


def find_exact_meta_index(h5: Dict[str, Any], rec_id: int, track_id: int, t0_frame: int) -> int:
    rids = h5["meta_recordingId"].astype(int)
    tids = h5["meta_trackId"].astype(int)
    t0s = h5["meta_t0_frame"].astype(int)
    idxs = np.where((rids == rec_id) & (tids == track_id) & (t0s == t0_frame))[0]
    if len(idxs) > 0:
        return int(idxs[0])
    raise ValueError(
        f"Matching sample not found in {h5['path']} for "
        f"rec={rec_id:02d}, track={track_id}, t0={t0_frame}"
    )


def normalize_highd_scene(rec_id: int, raw_dir: Path) -> RawScene:
    xx = f"{rec_id:02d}"
    tracks = read_csv_smart(raw_dir / f"{xx}_tracks.csv")
    tmeta = read_csv_smart(raw_dir / f"{xx}_tracksMeta.csv")
    recmeta = read_csv_smart(raw_dir / f"{xx}_recordingMeta.csv")
    tracks.columns = [c.strip() for c in tracks.columns]
    tmeta.columns = [c.strip() for c in tmeta.columns]
    recmeta.columns = [c.strip() for c in recmeta.columns]

    id_col = "id"
    vid_to_dd = dict(zip(tmeta[id_col].astype(int), tmeta["drivingDirection"].astype(int)))
    vid_to_w = dict(zip(tmeta[id_col].astype(int), tmeta["width"].astype(float)))
    vid_to_h = dict(zip(tmeta[id_col].astype(int), tmeta["height"].astype(float)))

    df = tracks.copy()
    df["trackId"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["length"] = df["trackId"].map(vid_to_w).fillna(df.get("width", 0.0)).astype(float)
    df["veh_width"] = df["trackId"].map(vid_to_h).fillna(df.get("height", 0.0)).astype(float)
    df["xCenter"] = df["x"].astype(float) + 0.5 * df["length"]
    df["yCenter"] = df["y"].astype(float) + 0.5 * df["veh_width"]
    df["drivingDirection"] = df["trackId"].map(vid_to_dd).fillna(0).astype(int)

    upper = parse_float_list(str(recmeta.loc[0, "upperLaneMarkings"])) if "upperLaneMarkings" in recmeta else []
    lower = parse_float_list(str(recmeta.loc[0, "lowerLaneMarkings"])) if "lowerLaneMarkings" in recmeta else []
    if upper and lower:
        cy = float(upper[-1] + lower[0])
        x_max = float(df["xCenter"].max())
        flip_mask = df["drivingDirection"] == 1
        df.loc[flip_mask, "xCenter"] = x_max - df.loc[flip_mask, "xCenter"]
        df.loc[flip_mask, "yCenter"] = cy - df.loc[flip_mask, "yCenter"]
        for col in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
            if col in df.columns:
                df.loc[flip_mask, col] = -df.loc[flip_mask, col].astype(float)

    return RawScene(
        dataset="highD",
        rec_id=rec_id,
        frame_rate=float(recmeta.loc[0, "frameRate"]),
        tracks=df.sort_values(["trackId", "frame"]),
        recmeta=recmeta,
        tmeta=tmeta,
        nb_cols=list(HIGHD_NB_COLS),
    )


def normalize_exid_scene(rec_id: int, raw_dir: Path, normalize_heading: bool) -> RawScene:
    xx = f"{rec_id:02d}"
    tracks = read_csv_smart(raw_dir / f"{xx}_tracks.csv")
    tmeta = read_csv_smart(raw_dir / f"{xx}_tracksMeta.csv")
    recmeta = read_csv_smart(raw_dir / f"{xx}_recordingMeta.csv")
    tracks.columns = [c.strip() for c in tracks.columns]
    tmeta.columns = [c.strip() for c in tmeta.columns]
    recmeta.columns = [c.strip() for c in recmeta.columns]

    df = tracks.copy()
    df["trackId"] = df["trackId"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["xCenter"] = pd.to_numeric(df.get("xCenter", df.get("x", 0.0)), errors="coerce").fillna(0.0)
    df["yCenter"] = pd.to_numeric(df.get("yCenter", df.get("y", 0.0)), errors="coerce").fillna(0.0)
    df["length"] = pd.to_numeric(df.get("length", 0.0), errors="coerce").fillna(0.0)
    df["veh_width"] = pd.to_numeric(df.get("width", 0.0), errors="coerce").fillna(0.0)
    df["heading_rad"] = np.deg2rad(pd.to_numeric(df.get("heading", 0.0), errors="coerce").fillna(0.0))

    return RawScene(
        dataset="exiD",
        rec_id=rec_id,
        frame_rate=float(recmeta.loc[0, "frameRate"]) if "frameRate" in recmeta.columns else 25.0,
        tracks=df.sort_values(["trackId", "frame"]),
        recmeta=recmeta,
        tmeta=tmeta,
        nb_cols=list(EXID_NB_COLS),
        normalize_heading=normalize_heading,
    )


def rows_by_track_frame(df: pd.DataFrame) -> Dict[int, Dict[int, pd.Series]]:
    out: Dict[int, Dict[int, pd.Series]] = {}
    for tid, group in df.groupby("trackId", sort=False):
        out[int(tid)] = {int(r.frame): r for r in group.itertuples(index=False)}
    return out


def row_xy(row: Any, scene: RawScene) -> Tuple[float, float]:
    x, y = float(row.xCenter), float(row.yCenter)
    if scene.dataset == "exiD" and scene.normalize_heading:
        return norm_pos(x, y, scene.norm_center[0], scene.norm_center[1], scene.ref_hdg)
    return x - scene.norm_center[0], y - scene.norm_center[1]


def row_heading(row: Any, scene: RawScene) -> float:
    if scene.dataset == "exiD":
        hdg = float(getattr(row, "heading_rad", 0.0))
        return hdg - scene.ref_hdg if scene.normalize_heading else hdg
    return 0.0


def get_sample_geometry(scene: RawScene, track_id: int, t0_frame: int) -> Dict[str, Any]:
    by_tf = rows_by_track_frame(scene.tracks)
    if track_id not in by_tf:
        raise ValueError(f"trackId={track_id} not found in raw recording {scene.rec_id:02d}")

    step = max(1, int(round(scene.frame_rate / TARGET_HZ)))
    hist_frames = [t0_frame - (T_H - 1 - i) * step for i in range(T_H)]
    fut_frames = [t0_frame + (i + 1) * step for i in range(T_F)]
    all_frames = hist_frames + fut_frames
    hist_frame_set = set(hist_frames)
    ego_rows = [by_tf[track_id].get(f) for f in hist_frames]
    fut_rows = [by_tf[track_id].get(f) for f in fut_frames]
    if any(r is None for r in ego_rows) or any(r is None for r in fut_rows):
        raise ValueError("Raw CSV does not contain the full requested history/future window.")

    obs = ego_rows[-1]
    scene.norm_center = (float(obs.xCenter), float(obs.yCenter))
    if scene.dataset == "exiD" and scene.normalize_heading:
        scene.ref_hdg = float(getattr(obs, "heading_rad", 0.0))

    history = np.array([row_xy(r, scene) for r in ego_rows], dtype=np.float32)
    future = np.array([row_xy(r, scene) for r in fut_rows], dtype=np.float32)
    ego_size = (float(getattr(obs, "length", 4.8)), float(getattr(obs, "veh_width", 2.0)))
    ego_heading = np.array([row_heading(r, scene) for r in ego_rows], dtype=np.float32)

    obs_ids = [to_int_id(getattr(obs, col, -1)) for col in scene.nb_cols]
    neighbors: List[Dict[str, Any]] = []
    for slot, nid in enumerate(obs_ids):
        if nid <= 0 or nid not in by_tf:
            continue
        nb_items = [(frame, by_tf[nid].get(frame)) for frame in all_frames]
        nb_items = [(frame, row) for frame, row in nb_items if row is not None]
        if not nb_items:
            continue
        nb_frames = [frame for frame, _ in nb_items]
        nb_rows = [row for _, row in nb_items]
        xy = np.array([row_xy(r, scene) for r in nb_rows], dtype=np.float32)
        headings = np.array([row_heading(r, scene) for r in nb_rows], dtype=np.float32)
        is_history = np.array([frame in hist_frame_set for frame in nb_frames], dtype=bool)
        n0 = nb_rows[min(len(nb_rows) - 1, max(0, int(np.sum(is_history)) - 1))]
        neighbors.append({
            "slot": slot,
            "slot_name": SLOT_NAMES[slot],
            "track_id": nid,
            "frames": nb_frames,
            "is_history": is_history,
            "xy": xy,
            "heading": headings,
            "length": float(getattr(n0, "length", 4.8)),
            "width": float(getattr(n0, "veh_width", 2.0)),
        })

    return {
        "history": history,
        "future": future,
        "hist_frames": hist_frames,
        "fut_frames": fut_frames,
        "ego_size": ego_size,
        "ego_heading": ego_heading,
        "neighbors": neighbors,
    }


def predict_one(model: Model, h5: Dict[str, Any], idx: int, cfg: Dict[str, Any], device: torch.device) -> np.ndarray:
    input_arr, adj_raw, pred_len = read_h5_sample(h5, idx)
    x = torch.from_numpy(input_arr).float()
    if x.shape[1] != model.in_channels:
        x = x.permute(0, 3, 2, 1)

    graph = Graph(max_hop=int(cfg["model"]["max_hop"]), num_node=int(cfg["model"]["num_node"]))
    adj_raw = adj_raw.astype(np.float32)
    adj = graph.normalize_adjacency(graph.get_adjacency(adj_raw))
    adj_t = torch.from_numpy(adj[None]).float()

    with torch.no_grad():
        out = model(x.to(device), adj_t.to(device), pred_len)
    pred = out[:, :, :, 0].permute(0, 2, 1)[0].detach().cpu().numpy()
    return pred


def trajectory_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    n = min(len(pred), len(target))
    if n == 0:
        return {"ADE": float("nan"), "FDE": float("nan"), "RMSE": float("nan")}
    diff = pred[:n] - target[:n]
    dist = np.linalg.norm(diff, axis=1)
    return {
        "ADE": float(np.mean(dist)),
        "FDE": float(dist[-1]),
        "RMSE": float(np.sqrt(np.mean(dist ** 2))),
    }


def parse_lanelet_osm(osm_path: Path, recmeta: pd.DataFrame) -> List[np.ndarray]:
    if not osm_path.exists():
        return []
    root = ET.parse(osm_path).getroot()
    lat0 = float(recmeta.loc[0, "latLocation"]) if "latLocation" in recmeta.columns else 0.0
    lon0 = float(recmeta.loc[0, "lonLocation"]) if "lonLocation" in recmeta.columns else 0.0
    x0 = float(recmeta.loc[0, "xUtmOrigin"]) if "xUtmOrigin" in recmeta.columns else 0.0
    y0 = float(recmeta.loc[0, "yUtmOrigin"]) if "yUtmOrigin" in recmeta.columns else 0.0

    transformer = None
    try:
        from pyproj import Transformer  # type: ignore
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    except Exception:
        transformer = None

    nodes: Dict[str, Tuple[float, float]] = {}
    for n in root.findall("node"):
        lat = float(n.attrib["lat"])
        lon = float(n.attrib["lon"])
        if transformer is not None:
            ux, uy = transformer.transform(lon, lat)
            x, y = ux - x0, uy - y0
        else:
            meters_per_lat = 111_320.0
            meters_per_lon = 111_320.0 * math.cos(math.radians(lat0))
            x = (lon - lon0) * meters_per_lon
            y = (lat - lat0) * meters_per_lat
        nodes[n.attrib["id"]] = (float(x), float(y))

    lines: List[np.ndarray] = []
    for way in root.findall("way"):
        tags = {t.attrib.get("k"): t.attrib.get("v") for t in way.findall("tag")}
        if tags.get("type") not in {"line_thin", "road_border", "curbstone"}:
            continue
        pts = [nodes[nd.attrib["ref"]] for nd in way.findall("nd") if nd.attrib["ref"] in nodes]
        if len(pts) >= 2:
            lines.append(np.array(pts, dtype=np.float32))
    return lines


def exid_osm_path(lanelet_root: Path, recmeta: pd.DataFrame) -> Optional[Path]:
    if "locationId" not in recmeta.columns:
        return None
    loc = int(recmeta.loc[0, "locationId"])
    matches = sorted(lanelet_root.glob(f"{loc}_*/location{loc}.osm"))
    if matches:
        return matches[0]
    direct = lanelet_root / f"location{loc}.osm"
    return direct if direct.exists() else None


def transform_map_lines(lines: List[np.ndarray], scene: RawScene) -> List[np.ndarray]:
    out = []
    for line in lines:
        if scene.dataset == "exiD" and scene.normalize_heading:
            pts = [norm_pos(float(x), float(y), scene.norm_center[0], scene.norm_center[1], scene.ref_hdg) for x, y in line]
        else:
            pts = [(float(x) - scene.norm_center[0], float(y) - scene.norm_center[1]) for x, y in line]
        out.append(np.array(pts, dtype=np.float32))
    return out


def draw_rotated_box(ax: plt.Axes, xy: Sequence[float], length: float, width: float, heading: float, **kwargs: Any) -> None:
    from matplotlib.patches import Rectangle

    x, y = float(xy[0]), float(xy[1])
    rect = Rectangle(
        (x - length / 2.0, y - width / 2.0),
        length,
        width,
        angle=math.degrees(heading),
        rotation_point="center",
        fill=False,
        **kwargs,
    )
    ax.add_patch(rect)


def draw_highd_lanes(ax: plt.Axes, scene: RawScene, xlim: Tuple[float, float]) -> None:
    rec = scene.recmeta
    marks = parse_float_list(str(rec.loc[0, "upperLaneMarkings"])) + parse_float_list(str(rec.loc[0, "lowerLaneMarkings"]))
    if not marks:
        return
    upper = parse_float_list(str(rec.loc[0, "upperLaneMarkings"])) if "upperLaneMarkings" in rec else []
    lower = parse_float_list(str(rec.loc[0, "lowerLaneMarkings"])) if "lowerLaneMarkings" in rec else []
    cy = float(upper[-1] + lower[0]) if upper and lower else None
    ego_dd_rows = scene.tracks[np.isclose(scene.tracks["xCenter"], scene.norm_center[0]) & np.isclose(scene.tracks["yCenter"], scene.norm_center[1])]
    ego_dd = int(ego_dd_rows.iloc[0]["drivingDirection"]) if len(ego_dd_rows) else 2
    for y_raw in marks:
        y = (cy - y_raw) if (cy is not None and ego_dd == 1) else y_raw
        ax.hlines(y - scene.norm_center[1], xlim[0], xlim[1], colors="0.15", linestyles="--", linewidth=0.8, alpha=0.45)


def plot_scene(
    scene: RawScene,
    geom: Dict[str, Any],
    preds: List[Tuple[str, np.ndarray]],
    title: str,
    out_path: Path,
    *,
    lanelet_root: Path,
    show: bool,
    invert_y: bool,
    show_legend: bool,
    show_neighbor_labels: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Run this script in the same environment "
            "used for plotting, or install matplotlib there."
        ) from exc

    n_plots = max(1, len(preds))
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(8.2, 1.5 * n_plots),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_list = list(axes[:, 0])
    history = geom["history"]
    future = geom["future"]
    gt_full = np.vstack([history, future])

    all_pts = [gt_full]
    for _, pred in preds:
        all_pts.append(pred)
    pts_all = np.vstack(all_pts)
    xmin, ymin = np.nanmin(pts_all, axis=0)
    xmax, ymax = np.nanmax(pts_all, axis=0)
    padx = max(8.0, 0.12 * (xmax - xmin + 1e-6))
    pady = max(8.0, 0.20 * (ymax - ymin + 1e-6))

    raw_map_lines: List[np.ndarray] = []
    if scene.dataset == "exiD":
        osm = exid_osm_path(lanelet_root, scene.recmeta)
        if osm is not None:
            raw_map_lines = transform_map_lines(parse_lanelet_osm(osm, scene.recmeta), scene)

    cmap = plt.get_cmap("tab10")
    pred_colors = ["tab:blue", "tab:red"]
    for pi, ((label, pred), ax) in enumerate(zip(preds, axes_list)):
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)

        if scene.dataset == "exiD":
            for line in raw_map_lines:
                if len(line) and not (
                    line[:, 0].max() < xmin - 2 * padx or line[:, 0].min() > xmax + 2 * padx
                    or line[:, 1].max() < ymin - 2 * pady or line[:, 1].min() > ymax + 2 * pady
                ):
                    ax.plot(line[:, 0], line[:, 1], color="0.75", linewidth=0.8, alpha=0.8, zorder=0)
        else:
            draw_highd_lanes(ax, scene, ax.get_xlim())

        for nb_i, nb in enumerate(geom["neighbors"]):
            color = cmap((nb_i + 3) % 10)
            hist_xy = nb["xy"][nb["is_history"]]
            fut_xy = nb["xy"][~nb["is_history"]]
            ax.plot(nb["xy"][:, 0], nb["xy"][:, 1], color=color, linewidth=1.0, alpha=0.5, zorder=2)
            if len(hist_xy) > 0:
                ax.scatter(hist_xy[:, 0], hist_xy[:, 1], color=color, s=12, alpha=0.65, zorder=3)
            if len(fut_xy) > 0:
                ax.scatter(fut_xy[:, 0], fut_xy[:, 1], color=color, s=12, marker="x", alpha=0.55, zorder=3)
            for ti, xy in enumerate(nb["xy"]):
                alpha = 0.45 if bool(nb["is_history"][ti]) else 0.18
                draw_rotated_box(
                    ax, xy, nb["length"], nb["width"], float(nb["heading"][ti]),
                    edgecolor=color, linewidth=0.8, alpha=alpha, zorder=2,
                )
            if show_neighbor_labels:
                ax.text(
                    nb["xy"][-1, 0],
                    nb["xy"][-1, 1],
                    f"{nb['slot_name']}:{nb['track_id']}",
                    fontsize=8,
                    color=color,
                )

        ax.plot(gt_full[:, 0], gt_full[:, 1], color="0.2", linewidth=2.0, marker="o", markersize=3, label="GT hist+future", zorder=5)
        ax.scatter(history[-1, 0], history[-1, 1], color="black", s=45, marker="x", label="t0", zorder=7)
        for ti, xy in enumerate(history):
            draw_rotated_box(
                ax, xy, geom["ego_size"][0], geom["ego_size"][1], float(geom["ego_heading"][ti]),
                edgecolor="black", linewidth=1.0, alpha=0.25 + 0.08 * ti, zorder=4,
            )

        pred_color = pred_colors[pi % len(pred_colors)]
        ax.plot(pred[:, 0], pred[:, 1], color=pred_color, linewidth=2.0, marker="o", markersize=3, label=label, zorder=6)
        ax.scatter(pred[-1, 0], pred[-1, 1], color=pred_color, s=55, marker="x", zorder=7)

        ax.set_aspect("equal", adjustable="box")
        if invert_y:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
        if pi == n_plots - 1:
            ax.set_xlabel("x (m)", fontsize=7, labelpad=1)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("y (m)", fontsize=7, labelpad=1)
        ax.tick_params(axis="both", labelsize=6, pad=1)
        ax.set_title(f"{title} | {label}", fontsize=10, pad=1.5)
        if show_legend:
            ax.legend(loc="best")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.10, top=0.93, hspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"saved: {out_path}", flush=True)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, help="one config, or one config per checkpoint")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or two checkpoint paths")
    ap.add_argument("--dataset", choices=["highD", "exiD"], default=None)
    ap.add_argument("--h5", nargs="+", default=None, help="one h5, or one h5 per checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--scenario_labels", default=None)
    ap.add_argument("--event", choices=["lane_change", "lane_following"], default=None)
    ap.add_argument("--state", choices=["dense", "free_flow"], default=None)
    ap.add_argument("--recordingId", type=int, default=None)
    ap.add_argument("--trackId", type=int, default=None)
    ap.add_argument("--t0_frame", type=int, default=None)
    ap.add_argument("--idx", type=int, default=None, help="direct h5 sample index")
    ap.add_argument("--scenario_rank", type=int, default=0, help="nth sample among matches for --event/--state filters")
    ap.add_argument("--out", default="vis_grip")
    ap.add_argument("--lanelet_root", default="exiD/lanelet2")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--show_legend", action="store_true")
    ap.add_argument("--show_neighbor_labels", action="store_true")
    ap.add_argument("--no_invert_y", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if len(args.ckpt) not in (1, 2):
        raise ValueError("--ckpt accepts one or two paths")
    if len(args.config) not in (1, len(args.ckpt)):
        raise ValueError("--config must be either one config or one config per checkpoint")
    if args.h5 is not None and len(args.h5) not in (1, len(args.ckpt)):
        raise ValueError("--h5 must be either one h5 or one h5 per checkpoint")

    cfg_paths = [Path(p) for p in (args.config if len(args.config) == len(args.ckpt) else args.config * len(args.ckpt))]
    cfgs = [yaml.safe_load(p.read_text()) for p in cfg_paths]
    dataset = infer_dataset(cfgs[0], args.dataset)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else str(cfgs[0].get("exp", {}).get("device", "cuda:0")))

    models: List[Tuple[str, Model, str, Dict[str, Any]]] = []
    for ckpt_str, cfg in zip(args.ckpt, cfgs):
        ckpt = Path(ckpt_str).expanduser().resolve()
        m, feature_mode = build_model_from_cfg(cfg, ckpt, device)
        models.append((ckpt.parent.name if ckpt.name == "best.pt" else ckpt.stem, m, feature_mode, cfg))

    h5_paths: List[Path] = []
    if args.h5 is None:
        h5_paths = [default_h5_path(cfg, feature_mode, args.split) for _, _, feature_mode, cfg in models]
    elif len(args.h5) == 1:
        h5_paths = [Path(args.h5[0])] * len(models)
    else:
        h5_paths = [Path(p) for p in args.h5]
    h5s = [load_h5_arrays(p) for p in h5_paths]
    ref_h5 = h5s[0]
    ref_feature_mode = models[0][2]

    labels_path = Path(args.scenario_labels) if args.scenario_labels else h5_paths[0].parent / "scenario_labels.csv"
    labels = load_scenario_labels(labels_path)
    if (args.event or args.state) and labels is None:
        print(f"[WARN] scenario_labels not found: {labels_path}; falling back to meta/index selection")

    if args.idx is not None:
        idx = int(args.idx)
        if idx < 0 or idx >= len(ref_h5["meta_recordingId"]):
            raise ValueError(f"--idx={idx} is out of range for {h5_paths[0]} ({len(ref_h5['meta_recordingId'])} samples)")
        reason = f"idx={idx}"
    else:
        idx, reason = find_sample_index(
            ref_h5,
            rec_id=args.recordingId,
            track_id=args.trackId,
            t0_frame=args.t0_frame,
            labels=labels,
            event=args.event,
            state=args.state,
            scenario_rank=args.scenario_rank,
        )

    rec_id = int(ref_h5["meta_recordingId"][idx])
    track_id = int(ref_h5["meta_trackId"][idx])
    t0_frame = int(ref_h5["meta_t0_frame"][idx])
    sample_indices = [idx] + [
        find_exact_meta_index(h5_other, rec_id, track_id, t0_frame)
        for h5_other in h5s[1:]
    ]
    print(f"selected sample: ref_idx={idx} rec={rec_id:02d} track={track_id} t0={t0_frame}", flush=True)
    print(
        "model sample indices: "
        + " | ".join(
            f"{feature_mode} idx={sample_idx}"
            for (_, _, feature_mode, _), sample_idx in zip(models, sample_indices)
        ),
        flush=True,
    )

    raw_dir = Path(args.raw_dir) if args.raw_dir else default_raw_dir(dataset)
    normalize_heading = bool(ref_h5.get("attrs", {}).get("normalize_heading", dataset == "exiD"))
    scene = normalize_exid_scene(rec_id, raw_dir, normalize_heading) if dataset == "exiD" else normalize_highd_scene(rec_id, raw_dir)
    geom = get_sample_geometry(scene, track_id, t0_frame)

    preds: List[Tuple[str, np.ndarray]] = []
    metric_rows: List[Tuple[str, Dict[str, float]]] = []
    feature_modes = [fm for _, _, fm, _ in models]
    for (label, m, feature_mode, cfg), h5, sample_idx in zip(models, h5s, sample_indices):
        pred = predict_one(m, h5, sample_idx, cfg, device)
        metrics = trajectory_metrics(pred, geom["future"])
        plot_label = (
            f"feature={feature_mode} | "
            f"ADE={metrics['ADE']:.3f} FDE={metrics['FDE']:.3f} RMSE={metrics['RMSE']:.3f}"
        )
        preds.append((plot_label, pred))
        metric_rows.append((feature_mode, metrics))

    if len(metric_rows) == 1:
        m = metric_rows[0][1]
        metrics_text = f"ADE={m['ADE']:.3f} FDE={m['FDE']:.3f} RMSE={m['RMSE']:.3f}"
    else:
        metrics_text = " | ".join(
            f"ADE={m['ADE']:.3f} FDE={m['FDE']:.3f} RMSE={m['RMSE']:.3f}"
            for _, m in metric_rows
        )
    feature_text = "/".join(feature_modes)
    title = f"{dataset} | rec={rec_id:02d} track={track_id} t0={t0_frame}"
    print(f"{title} | feature={feature_text} | {metrics_text}", flush=True)
    out_name = f"{dataset}_rec{rec_id:02d}_track{track_id}_t0{t0_frame}_{'-vs-'.join(feature_modes)}.png"
    plot_scene(
        scene,
        geom,
        preds,
        title,
        Path(args.out) / out_name,
        lanelet_root=Path(args.lanelet_root),
        show=args.show,
        invert_y=not args.no_invert_y,
        show_legend=args.show_legend,
        show_neighbor_labels=args.show_neighbor_labels,
    )


if __name__ == "__main__":
    main()
