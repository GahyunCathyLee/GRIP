import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from layers.graph import Graph

class HighDFeeder(Dataset):
    def __init__(self, data_path, graph_args={'max_hop': 2, 'num_node': 9}, train_val_test='train'):
        self.data_path = data_path
        self.graph = Graph(**graph_args)
        
        print(f"[{train_val_test}] 데이터를 RAM에 로드 중: {data_path}")
        with h5py.File(self.data_path, 'r') as f:
            self.all_input = f['input'][:]
            self.all_adj = f['adj'][:]
            self.all_target = f['target'][:]
            
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
        
        return (torch.from_numpy(input_data).float(), 
                torch.from_numpy(normalized_A).float(), 
                torch.from_numpy(target).float())