import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda=True):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input):
        output, hidden = self.lstm(input)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=True):
        super(DecoderRNN, self).__init__()
        # 1. 2차원 좌표(output_size=2)를 hidden_size(64)로 변환해주는 레이어 추가
        self.embedding = nn.Linear(output_size, hidden_size)
        
        # 2. GRU 입력 차원을 hidden_size로 설정
        self.lstm = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded_input, hidden):
        # encoded_input: (N, 1, 2)
        embedded = self.embedding(encoded_input) # (N, 1, 64)로 변환
        embedded = self.dropout(embedded)
        
        decoded_output, hidden = self.lstm(embedded, hidden)
        decoded_output = self.linear(decoded_output) # 다시 (N, 1, 2)로 변환
        return decoded_output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, isCuda=True):
        super(Seq2Seq, self).__init__()
        self.isCuda = isCuda
        # Encoder는 GCN 피처(64)를 받음
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        # Decoder는 좌표(2)를 받아서 예측
        self.decoder = DecoderRNN(hidden_size, 2, num_layers, dropout, isCuda)
    
    def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
        batch_size = in_data.shape[0]
        out_dim = 2 
        outputs = torch.zeros(batch_size, pred_length, out_dim)
        if self.isCuda: outputs = outputs.cuda()

        encoded_output, hidden = self.encoder(in_data)
        decoder_input = last_location # (N, 1, 2)
        
        for t in range(pred_length):
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input # Residual 연결 (상대 위치 변화량 학습)
            outputs[:, t:t+1, :] = now_out
            decoder_input = now_out # 다음 스텝의 입력으로 사용
            
        return outputs