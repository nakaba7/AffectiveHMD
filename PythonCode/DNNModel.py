import torch.nn.functional as F
import torch.nn as nn

"""
表情識別に使うDNNの学習モデル

入力: headdatanum: 頭部姿勢データの次元数
出力: 5次元表情予測ラベル
"""

SENSOR_NUM = 16 #反射型光センサの数
HIDDEN_DIM = 128 
DROPOUT = 0.3 

class NeuralNet(nn.Module):    
        def __init__(self, headdatanum):
            super(NeuralNet, self).__init__()
            self.input_layer = nn.Linear(SENSOR_NUM+headdatanum, HIDDEN_DIM)
            self.hidden_layer1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.hidden_layer2 = nn.Linear(HIDDEN_DIM, 64)
            self.output_layer = nn.Linear(64, 5)
            self.dropout = nn.Dropout(DROPOUT)
            self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
            self.bn2 = nn.BatchNorm1d(64)
        
        def forward(self, x):
            x = self.input_layer(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.hidden_layer1(x)
            x = self.bn1(x)#バッチ正規化
            x = F.relu(x)
            #x = self.dropout(x)
            x = self.hidden_layer2(x)
            x = self.bn2(x)
            x = F.relu(x)
            #x = self.dropout(x)
            output = self.output_layer(x)
            #x = F.softmax(x, dim=1)
            return output