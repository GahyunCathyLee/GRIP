import argparse
import os
import yaml
import time
from highD.preprocess import EXTRA_FEATURE_MAP, get_num_channels
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from model import Model
from feeder import HighDFeeder as Feeder

def get_args():
    parser = argparse.ArgumentParser(description='GRIP++ Evaluation')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--measure_time', action='store_true', help='measure inference time with batch size 1')
    return parser.parse_args()

def calculate_metrics(pred, target):
    """
    pred: (N, T, 2)
    target: (N, T, 2)
    """
    # 유클리드 거리 계산: (N, T)
    diff = pred - target
    dist = torch.norm(diff, p=2, dim=-1) 

    # 1. ADE (Average Displacement Error)
    ade = torch.mean(dist).item()

    # 2. FDE (Final Displacement Error)
    fde = torch.mean(dist[:, -1]).item()

    # 3. RMSE (Root Mean Square Error)
    # RMSE = sqrt(mean(dist^2))
    rmse = torch.sqrt(torch.mean(dist**2)).item()

    # 4. RMSE @ 1s ~ 5s (Data freq가 5Hz인 경우: 5, 10, 15, 20, 25 frames)
    # 사용자의 데이터 주기에 맞춰 인덱스를 조절하세요.
    rmse_steps = {}
    fps = pred.shape[1] // 5  # future_frames / 5sec -> 3Hz:3, 5Hz:5
    for s in range(1, 6):
        idx = s * fps - 1
        if idx < pred.shape[1]:
            step_rmse = torch.sqrt(torch.mean(dist[:, idx]**2)).item()
            rmse_steps[f'RMSE@{s}s'] = step_rmse

    return ade, fde, rmse, rmse_steps

def evaluate(model, loader, device):
    model.eval()
    all_ade, all_fde, all_rmse = [], [], []
    all_step_rmse = {f'RMSE@{s}s': [] for s in range(1, 6)}

    print(f"🔍 평가 시작 (Samples: {len(loader.dataset)})")
    
    with torch.no_grad():
        for data, adj, target in tqdm(loader, desc="Eval"):
            data, adj, target = data.to(device).float(), adj.to(device).float(), target.to(device).float()
            
            if data.shape[1] != model.in_channels:
                data = data.permute(0, 3, 2, 1)

            # Inference
            output = model(data, adj, target.shape[1])
            # Ego 차량(0번 노드)만 추출: (N, C, T, V) -> (N, T, C)
            ego_pred = output[:, :, :, 0].permute(0, 2, 1)

            # Metrics
            ade, fde, rmse, step_rmse = calculate_metrics(ego_pred, target)
            
            all_ade.append(ade)
            all_fde.append(fde)
            all_rmse.append(rmse)
            for k, v in step_rmse.items():
                all_step_rmse[k].append(v)

    # 최종 결과 출력
    print("\n" + "="*50)
    print(f"📊 Final Results (m)")
    print(f"  • ADE  : {np.mean(all_ade):.4f}")
    print(f"  • FDE  : {np.mean(all_fde):.4f}")
    print(f"  • RMSE : {np.mean(all_rmse):.4f}")
    print("-" * 50)
    for k, v in all_step_rmse.items():
        print(f"  • {k:8s} : {np.mean(v):.4f}")
    print("="*50 + "\n")

def measure_inference_time(model, loader, device, iterations=10000):
    model.eval()
    print(f"⏱️ Inference Time 측정 시작 (Batch Size: 1, Iterations: {iterations})")
    
    # 1. 단일 샘플 준비
    data, adj, target = next(iter(loader))
    data, adj = data[0:1].to(device).float(), adj[0:1].to(device).float()
    if data.shape[1] != model.in_channels:
        data = data.permute(0, 3, 2, 1)
    
    pred_len = target.shape[1]
    times = []

    # 2. Warm-up (GPU 예열) - 드라이버 및 cuDNN 초기화 시간 제외
    for _ in range(100):
        with torch.no_grad():
            _ = model(data, adj, pred_len)
    torch.cuda.synchronize()

    # 3. 실제 측정 루프
    print(f"🚀 Measuring...")
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            
            _ = model(data, adj, pred_len)
            
            torch.cuda.synchronize()  # GPU 연산이 완료될 때까지 대기
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # ms 단위로 저장

    # 4. 통계 계산 (numpy 활용)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # 5. 결과 출력
    print("\n" + "="*50)
    print(f"📊 Inference Time Statistics (ms)")
    print(f"  • Avg Latency : {avg_time:.4f} ms")
    print(f"  • Std Dev     : {std_time:.4f} ms")
    print(f"  • Min Latency : {min_time:.4f} ms")
    print(f"  • Max Latency : {max_time:.4f} ms")
    print("-" * 50)
    print(f"  • FPS (Avg)   : {1000/avg_time:.2f} frames/s")
    print("="*50 + "\n")

def main():
    args = get_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Feature Mode에 따른 자동 입력 채널 설정
    feature_mode = cfg['exp']['feature_mode']
    in_channels = get_num_channels(feature_mode)
    
    # 2. 모델 로드
    model = Model(in_channels=in_channels, 
                  graph_args={'max_hop': cfg['model']['max_hop'], 'num_node': cfg['model']['num_node']}, 
                  edge_importance_weighting=cfg['model']['edge_importance_weighting']).to(device)

    # 가중치 불러오기
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Checkpoint Loaded: {args.ckpt}")

    # 2. 데이터 로더 준비 (Test/Val 데이터)
    test_path = Path(cfg['data']['base_dir']) / cfg['exp']['feature_mode'] / "test.h5"
    if not test_path.exists():
        test_path = Path(cfg['data']['base_dir']) / cfg['exp']['feature_mode'] / "val.h5"
    
    batch_size = cfg['data'].get('batch_size_val', cfg['data'].get('batch_size', 64))
    
    test_loader = torch.utils.data.DataLoader(
        Feeder(str(test_path)),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 3. 모드별 실행
    if args.measure_time:
        measure_inference_time(model, test_loader, device)
    else:
        evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()