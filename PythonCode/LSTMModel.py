import torch.nn.functional as F
import torch.nn as nn

"""
表情識別に使うLSTMの学習モデル

入力: headdatanum: 頭部姿勢データの次元数
出力: 5次元表情予測ラベル
"""

SENSOR_NUM = 16 #反射型光センサの数
HIDDEN_DIM = 128 
DROPOUT = 0.3 

class LSTMClassifier(nn.Module):
        # モデルで使う各ネットワークをコンストラクタで定義
        def __init__(self, headdatanum):
            super(LSTMClassifier, self).__init__()
            self.input_layer = nn.Linear(SENSOR_NUM + headdatanum, HIDDEN_DIM)
            self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
            self.output_layer = nn.Linear(HIDDEN_DIM, 5)
            self.dropout = nn.Dropout(DROPOUT)
            self.batchnorm = nn.BatchNorm1d(HIDDEN_DIM)

        def forward(self, x):
            b, s, d = x.shape
            x = x.reshape(b*s, d)
            x = self.input_layer(x)#入力層
            x = x.reshape(b, s, HIDDEN_DIM)
            x = x.permute(0,2,1)
            x = self.batchnorm(x)#バッチ正規化(時系列方向なのでLayer Normかも)
            x = x.permute(0,2,1)
            x = F.relu(x)
            x = self.dropout(x)#ドロップアウト
            x, _ = self.lstm(x)#LSTM層
            output = F.relu(x[:,-1,:])
            output = self.dropout(output)#ドロップアウト
            output = self.output_layer(output)#出力層
            #x = F.softmax(x, dim=1)
            return output