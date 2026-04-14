import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from layers.graph import Graph

class HighDFeeder(Dataset):
    def __init__(self, data_path, graph_args={'max_hop': 2, 'num_node': 9},
                 train_val_test='train', return_meta=False):
        self.data_path = data_path
        self.graph = Graph(**graph_args)
        self.return_meta = return_meta

        print(f"[{train_val_test}] 데이터를 RAM에 로드 중: {data_path}")
        with h5py.File(self.data_path, 'r') as f:
            self.all_input = f['input'][:]
            self.all_adj = f['adj'][:]
            self.all_target = f['target'][:]
            if return_meta and 'meta_recordingId' in f:
                self.meta_rec   = f['meta_recordingId'][:]
                self.meta_track = f['meta_trackId'][:]
                self.meta_frame = f['meta_t0_frame'][:]
            else:
                self.meta_rec = self.meta_track = self.meta_frame = None
                if return_meta:
                    print("[WARN] h5 파일에 meta 정보가 없습니다. 전처리를 다시 실행하세요.")

        self.num_samples = self.all_input.shape[0]
        print(f"✅ {self.num_samples}개의 샘플이 RAM 적재 완료되었습니다.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_data = self.all_input[idx]
        adj_raw = self.all_adj[idx]
        target = self.all_target[idx]

        now_adjacency = self.graph.get_adjacency(adj_raw)
        normalized_A = self.graph.normalize_adjacency(now_adjacency)

        if self.return_meta and self.meta_rec is not None:
            meta = {
                'recordingId': int(self.meta_rec[idx]),
                'trackId':     int(self.meta_track[idx]),
                't0_frame':    int(self.meta_frame[idx]),
            }
            return (torch.from_numpy(input_data).float(),
                    torch.from_numpy(normalized_A).float(),
                    torch.from_numpy(target).float(),
                    meta)

        return (torch.from_numpy(input_data).float(),
                torch.from_numpy(normalized_A).float(),
                torch.from_numpy(target).float())
