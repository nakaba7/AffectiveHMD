import torch.nn.functional as F
import torch.nn as nn

"""
表情識別に使うRNNの学習モデル

入力: headdatanum: 頭部姿勢データの次元数
出力: 5次元表情予測ラベル
"""

SENSOR_NUM = 16 #反射型光センサの数
HIDDEN_DIM = 128 
DROPOUT = 0.3 

class RNNClassifier(nn.Module):
        def __init__(self, headdatanum):
            super(RNNClassifier, self).__init__()
            self.rnn = nn.RNN(SENSOR_NUM+headdatanum, hidden_size = HIDDEN_DIM, batch_first = True)
            self.fc = nn.Linear(HIDDEN_DIM, 5)
            self.dropout = nn.Dropout(DROPOUT)
            self.batchnorm = nn.BatchNorm1d(SENSOR_NUM+headdatanum)

        def forward(self, x):
            #batch_size = x.size(0)
            x = x.permute(0,2,1)
            x = self.batchnorm(x)#時系列方向にバッチ正規化
            x = x.permute(0,2,1)
            x_rnn, hidden = self.rnn(x, None)
            #output = F.relu(x_rnn[:, -1, :])
            output = self.dropout(x_rnn[:, -1, :])
            x = self.fc(output)
            return x