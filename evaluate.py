import argparse
import os
import yaml
import time
from highD.preprocess import EXTRA_FEATURE_MAP, get_num_channels
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from model import Model
from feeder import HighDFeeder as Feeder


# ──────────────────────────────────────────────────────────────────────────────
# Scenario label helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_scenario_labels(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels not found: {path}")
        return None
    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] scenario_labels missing columns {missing}")
        return None
    has_event = "event_label" in df.columns
    has_state = "state_label" in df.columns
    if not has_event and not has_state:
        print("[WARN] scenario_labels has no event_label/state_label")
        return None
    lut = {}
    for row in df.itertuples(index=False):
        key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
        lut[key] = {
            "event_label": getattr(row, "event_label", None) if has_event else None,
            "state_label": getattr(row, "state_label", None) if has_state else None,
        }
    print(f"[INFO] Loaded scenario labels: {len(lut):,} entries from {path}")
    return lut


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sep(widths, left="+", mid="+", right="+", fill="-"):
    return left + mid.join(fill * w for w in widths) + right


def print_scenario_results(stats, label_type):
    if not stats:
        return
    rows = sorted(stats.items(), key=lambda x: (x[0] == "unknown", x[0]))
    c_lbl = max(len(lbl) for lbl, _ in rows)
    c_lbl = max(c_lbl, len(label_type)) + 2
    c_n = 9
    c_m = 11
    ws = [c_lbl, c_n, c_m, c_m, c_m]

    print(f"\n====== Scenario Results [{label_type}] ======")
    print(_sep(ws))
    print(f"|{label_type:^{c_lbl}}|{'n':^{c_n}}"
          f"|{'ADE':^{c_m}}|{'FDE':^{c_m}}|{'RMSE':^{c_m}}|")
    print(_sep(ws))

    for lbl, (sa, sf, sr, n) in rows:
        if n == 0:
            continue
        print(f"|{lbl:^{c_lbl}}|{n:^{c_n},}"
              f"|{sa/n:^{c_m}.4f}|{sf/n:^{c_m}.4f}|{sr/n:^{c_m}.4f}|")
    print(_sep(ws))

    total_sa = sum(v[0] for v in stats.values())
    total_sf = sum(v[1] for v in stats.values())
    total_sr = sum(v[2] for v in stats.values())
    total_n = sum(v[3] for v in stats.values())
    N = max(1, total_n)
    print(f"|{'Total':^{c_lbl}}|{total_n:^{c_n},}"
          f"|{total_sa/N:^{c_m}.4f}|{total_sf/N:^{c_m}.4f}|{total_sr/N:^{c_m}.4f}|")
    print(_sep(ws))


# ──────────────────────────────────────────────────────────────────────────────
# Collate function that handles optional meta
# ──────────────────────────────────────────────────────────────────────────────

def collate_with_meta(batch):
    if len(batch[0]) == 4:
        inputs, adjs, targets, metas = zip(*batch)
        return (torch.stack(inputs), torch.stack(adjs),
                torch.stack(targets), list(metas))
    else:
        inputs, adjs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(adjs), torch.stack(targets), None


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='GRIP++ Evaluation')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--measure_time', action='store_true', help='measure inference time with batch size 1')
    parser.add_argument('--scenario_labels', type=str, default=None,
                        help='path to scenario_labels.csv for per-scenario breakdown')
    return parser.parse_args()


def calculate_metrics(pred, target):
    """
    pred: (N, T, 2)
    target: (N, T, 2)
    Returns per-sample metrics: ade (N,), fde (N,), rmse (N,), step_rmse dict
    """
    diff = pred - target
    dist = torch.norm(diff, p=2, dim=-1)  # (N, T)

    ade_per = dist.mean(dim=1)          # (N,)
    fde_per = dist[:, -1]               # (N,)
    rmse_per = torch.sqrt((dist ** 2).mean(dim=1))  # (N,)

    # RMSE @ 1s ~ 5s
    rmse_steps = {}
    fps = pred.shape[1] // 5
    for s in range(1, 6):
        idx = s * fps - 1
        if idx < pred.shape[1]:
            step_rmse = torch.sqrt(dist[:, idx] ** 2)  # (N,)
            rmse_steps[f'RMSE@{s}s'] = step_rmse.mean().item()

    return ade_per, fde_per, rmse_per, rmse_steps


def evaluate(model, loader, device, labels_lut=None):
    model.eval()
    all_ade, all_fde, all_rmse = [], [], []
    all_step_rmse = {f'RMSE@{s}s': [] for s in range(1, 6)}

    ev_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])

    print(f"평가 시작 (Samples: {len(loader.dataset)})")

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Eval"):
            data, adj, target, metas = batch_data
            data, adj, target = data.to(device).float(), adj.to(device).float(), target.to(device).float()

            if data.shape[1] != model.in_channels:
                data = data.permute(0, 3, 2, 1)

            output = model(data, adj, target.shape[1])
            ego_pred = output[:, :, :, 0].permute(0, 2, 1)  # (N, T, C)

            ade_per, fde_per, rmse_per, step_rmse = calculate_metrics(ego_pred, target)

            all_ade.append(ade_per.mean().item())
            all_fde.append(fde_per.mean().item())
            all_rmse.append(rmse_per.mean().item())
            for k, v in step_rmse.items():
                all_step_rmse[k].append(v)

            # Per-scenario accumulation
            if labels_lut is not None and metas is not None:
                ade_np = ade_per.cpu().numpy()
                fde_np = fde_per.cpu().numpy()
                rmse_np = rmse_per.cpu().numpy()
                B = ade_np.shape[0]
                for i in range(B):
                    meta = metas[i]
                    key = (meta['recordingId'], meta['trackId'], meta['t0_frame'])
                    lab = labels_lut.get(key)
                    if lab is None:
                        continue
                    ev = lab.get("event_label") or "unknown"
                    st = lab.get("state_label") or "unknown"
                    for acc, lbl in ((ev_stats, ev), (st_stats, st)):
                        acc[lbl][0] += float(ade_np[i])
                        acc[lbl][1] += float(fde_np[i])
                        acc[lbl][2] += float(rmse_np[i])
                        acc[lbl][3] += 1

    # Final results
    print("\n" + "=" * 50)
    print(f"Final Results (m)")
    print(f"  ADE  : {np.mean(all_ade):.4f}")
    print(f"  FDE  : {np.mean(all_fde):.4f}")
    print(f"  RMSE : {np.mean(all_rmse):.4f}")
    print("-" * 50)
    for k, v in all_step_rmse.items():
        if v:
            print(f"  {k:8s} : {np.mean(v):.4f}")
    print("=" * 50 + "\n")

    if ev_stats:
        print_scenario_results(ev_stats, label_type="Event")
    if st_stats:
        print_scenario_results(st_stats, label_type="State")


def measure_inference_time(model, loader, device, iterations=10000):
    model.eval()
    print(f"Inference Time 측정 시작 (Batch Size: 1, Iterations: {iterations})")

    data, adj, target, _ = next(iter(loader))
    data, adj = data[0:1].to(device).float(), adj[0:1].to(device).float()
    if data.shape[1] != model.in_channels:
        data = data.permute(0, 3, 2, 1)

    pred_len = target.shape[1]
    times = []

    for _ in range(100):
        with torch.no_grad():
            _ = model(data, adj, pred_len)
    torch.cuda.synchronize()

    print(f"Measuring...")
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            _ = model(data, adj, pred_len)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print("\n" + "=" * 50)
    print(f"Inference Time Statistics (ms)")
    print(f"  Avg Latency : {avg_time:.4f} ms")
    print(f"  Std Dev     : {std_time:.4f} ms")
    print(f"  Min Latency : {min_time:.4f} ms")
    print(f"  Max Latency : {max_time:.4f} ms")
    print("-" * 50)
    print(f"  FPS (Avg)   : {1000/avg_time:.2f} frames/s")
    print("=" * 50 + "\n")


def main():
    args = get_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Checkpoint 먼저 로드해서 feature_mode 확인
    checkpoint = torch.load(args.ckpt, map_location=device)

    cfg_feature_mode  = cfg['exp']['feature_mode']
    ckpt_feature_mode = checkpoint.get('feature_mode', None)

    if ckpt_feature_mode is not None and ckpt_feature_mode != cfg_feature_mode:
        print(f"[WARN] config feature_mode={cfg_feature_mode!r} != "
              f"checkpoint feature_mode={ckpt_feature_mode!r}")
        print(f"[INFO] checkpoint feature_mode={ckpt_feature_mode!r} 으로 override합니다.")
        feature_mode = ckpt_feature_mode
    else:
        feature_mode = cfg_feature_mode

    in_channels = get_num_channels(feature_mode)

    # 2. Model
    model = Model(in_channels=in_channels,
                  graph_args={'max_hop': cfg['model']['max_hop'], 'num_node': cfg['model']['num_node']},
                  edge_importance_weighting=cfg['model']['edge_importance_weighting']).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint Loaded: {args.ckpt}  (feature_mode={feature_mode}, in_channels={in_channels})")

    # 3. Scenario labels
    labels_lut = None
    if args.scenario_labels and not args.measure_time:
        labels_lut = load_scenario_labels(args.scenario_labels)

    # 4. Data loader
    need_meta = labels_lut is not None
    test_path = Path(cfg['data']['base_dir']) / feature_mode / "test.h5"
    if not test_path.exists():
        test_path = Path(cfg['data']['base_dir']) / feature_mode / "val.h5"

    batch_size = cfg['data'].get('batch_size_val', cfg['data'].get('batch_size', 64))

    test_loader = torch.utils.data.DataLoader(
        Feeder(str(test_path), return_meta=need_meta),
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_with_meta
    )

    # 5. Run
    if args.measure_time:
        measure_inference_time(model, test_loader, device)
    else:
        evaluate(model, test_loader, device, labels_lut)

if __name__ == '__main__':
    main()
