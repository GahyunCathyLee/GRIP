import torch
import torch.nn as nn
from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq
import numpy as np 

class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.graph = Graph(**graph_args)
        
        # 기본 인접 행렬 구조 생성
        A = np.ones((graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node']))

        spatial_kernel_size = np.shape(A)[0]
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # 1. 입력 데이터 정규화를 위한 BatchNorm (루프에서 분리하여 관리하는 것이 안전합니다)
        self.input_bn = nn.BatchNorm2d(in_channels)

        # 2. ST-GCN 레이어들
        self.st_gcn_networks = nn.ModuleList((
            Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
        ))

        # 3. Edge Importance 설정
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for _ in range(len(self.st_gcn_networks))]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 4. highD 전용 단일 Seq2Seq 모델 (차량용으로 통합)
        self.seq2seq = Seq2Seq(input_size=64, hidden_size=64, num_layers=2, isCuda=True)

    def reshape_for_lstm(self, x):
        # x: (N, C, T, V) -> (N, V, T, C) -> (N*V, T, C)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        return x.view(n * v, t, c)

    def reshape_from_lstm(self, x):
        # x: (N*V, T, C) -> (N, V, T, C) -> (N, C, T, V)
        nv, t, c = x.size()
        v = 9  # num_node (Ego + 8 Neighbors)
        n = nv // v
        x = x.view(n, v, t, c)
        return x.permute(0, 3, 2, 1).contiguous()

    def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
        # 1. 먼저 BatchNorm으로 입력 정규화
        x = self.input_bn(pra_x)
        
        # 2. ST-GCN 레이어 통과 (각 레이어는 x와 A를 모두 받음)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, pra_A * importance)

        # 3. LSTM 입력을 위한 리쉐이핑
        graph_conv_feature = self.reshape_for_lstm(x)
        
        # 마지막 위치 정보를 LSTM에 전달 (dx, dy 기반이므로 첫 2채널 사용)
        # pra_x: (N, C, T, V)
        last_position = self.reshape_for_lstm(pra_x[:, :2, :, :])

        # 4. Seq2Seq 예측 (highD 전용 단일 모델)
        now_predict = self.seq2seq(
            in_data=graph_conv_feature, 
            last_location=last_position[:, -1:, :], 
            pred_length=pra_pred_length, 
            teacher_forcing_ratio=pra_teacher_forcing_ratio, 
            teacher_location=pra_teacher_location
        )
        
        # 5. 결과 복원 (N, C, T, V)
        return self.reshape_from_lstm(now_predict)