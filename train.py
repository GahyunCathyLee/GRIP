import argparse
import os
import yaml
import model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random

from model import Model
from feeder import HighDFeeder as Feeder
from highD.preprocess import EXTRA_FEATURE_MAP, get_num_channels

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataloader(cfg, split='train'):
    """feature_mode에 맞는 HDF5 데이터를 로드 """
    data_path = Path(cfg['data']['base_dir']) / cfg['exp']['feature_mode'] / f"{split}.h5"
    
    dataset = Feeder(data_path=str(data_path), train_val_test=split)
    
    batch_size = cfg['data']['batch_size'] 
    shuffle = True if split == 'train' else False
    
    persistent_workers = cfg['data'].get('persistent_workers', True)

    print(f"📂 {split.capitalize()} Data Loading from {data_path}...")

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=persistent_workers
    )

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem = total * 4 / (1024**2) # MB
    print("=" * 50)
    print("📊 Model Size Info")
    print(f"  • Total Parameters : {total:,}")
    print(f"  • Model Memory Size: {mem:.2f} MB")
    print("=" * 50)
    print()

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Train", dynamic_ncols=True)
    
    for data, adj, target in pbar:
        data, adj, target = data.to(device, non_blocking=True), adj.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if data.shape[1] != model.in_channels:
            data = data.permute(0, 3, 2, 1)
        
        output = model(data, adj, target.shape[1]) 
        
        ego_pred = output[:, :, :, 0].permute(0, 2, 1)
        loss = criterion(ego_pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(L=f"{loss.item():.4f}")
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Valid", dynamic_ncols=True)
    
    with torch.no_grad():
        for data, adj, target in pbar:
            data = data.to(device).float()
            adj = adj.to(device).float()
            target = target.to(device).float()

            # 2. 입력 차원 확인 및 변경 (N, V, T, C) -> (N, C, T, V)
            if data.shape[1] != model.in_channels:
                data = data.permute(0, 3, 2, 1)
            
            # 3. 모델 예측 (미래 프레임 수 전달)
            output = model(data, adj, target.shape[1])
            
            # 4. Ego 차량(0번 노드) 예측값만 추출하여 Loss 계산
            # output: (N, 2, 25, 9) -> ego_pred: (N, 25, 2)
            ego_pred = output[:, :, :, 0].permute(0, 2, 1)
            loss = criterion(ego_pred, target)
            
            total_loss += loss.item()
            
            pbar.set_postfix(L=f"{loss.item():.4f}")
            
    avg_val_loss = total_loss / len(loader)
    return avg_val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', action='store_true', help='auto resume from best.pt in ckpt_dir')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    seed_everything(cfg['exp']['seed'])
    device = torch.device(cfg['exp']['device'] if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # 1. 환경 및 경로 설정 (Resume을 위해 상단으로 이동)
    feature_mode = cfg['exp']['feature_mode']
    in_channels = get_num_channels(feature_mode)  # ← 동적 계산
    
    ckpt_dir = Path(cfg['train']['ckpt_dir']) / feature_mode
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"  

    active = EXTRA_FEATURE_MAP[feature_mode]
    print("=" * 50)
    exp = f"      Experiment: {feature_mode} (Active slots: {active})      "
    margin = (50 - len(exp)) // 2
    print(f"{' ' * margin}{exp}{' ' * margin}")
    
    # 2. 모델 및 학습 도구 초기화
    model = Model(in_channels=in_channels, 
                  graph_args={'max_hop': cfg['model']['max_hop'], 'num_node': cfg['model']['num_node']}, 
                  edge_importance_weighting=cfg['model']['edge_importance_weighting'])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.SmoothL1Loss()
    
    start_epoch = 1
    best_val_loss = float('inf')

    # 3. 자동 Resume 로직
    if args.resume:
        if best_path.exists():
            print(f"🔄 [Resume] Found existing checkpoint: {best_path}")
            checkpoint = torch.load(best_path, map_location=device)
            
            # 가중치 및 옵티마이저 상태 복구
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            print(f"✅ Successfully resumed from Epoch {checkpoint['epoch']} (Best Val Loss: {best_val_loss:.4f})")
        else:
            print(f"⚠️ [Resume] 'best.pt' not found at {best_path}. Starting from scratch.")

    count_parameters(model)

    # 4. 로그 및 데이터 로더 설정
    log_dir = Path("logs") / feature_mode / datetime.now().strftime("%m%d-%H%M")
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"📊 TensorBoard 로그가 {log_dir}에 기록됩니다.")
    
    train_loader = get_dataloader(cfg, 'train')
    val_loader = get_dataloader(cfg, 'val')
    print(f"✅ [Data] Ready! Train: {len(train_loader.dataset)} / Val: {len(val_loader.dataset)}\n")
    
    # 5. 학습 루프
    epochs = cfg['train']['epochs']
    for epoch in range(start_epoch, epochs + 1):
        print(f"{'=' * 30}  Epoch {epoch}/{epochs}  {'=' * 30}")
        train_l = train_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
        val_l = validate(model, val_loader, criterion, device, epoch, epochs)
        
        print(f"Epoch {epoch:3d} | Train Loss: {train_l:.4f} | Val Loss: {val_l:.4f} |")

        writer.add_scalar('Loss/train', train_l, epoch)
        writer.add_scalar('Loss/val', val_l, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Best 모델 저장 로직
        if val_l < best_val_loss:
            best_val_loss = val_l
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_l,
                'feature_mode': feature_mode
            }, best_path)
            print(f"⭐ Best model updated and saved to {best_path}")
        print("-" * 50)
        print()
    
    writer.close()

if __name__ == '__main__':
    main()