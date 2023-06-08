import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from SerialConnection16 import SerialConnection
import socket

SENSOR_NUM = 16
BATCH_SIZE = 64
OPTIMIZER_LEARNING_RATE = 1.0e-4
HIDDEN_DIM = 128
EPOCH_NUM = 500
SEQUENCE_TENSOR_LENGTH = 20
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HOST = '127.0.0.1'
PORT = 50007

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

MODELPATH = "Demo_LSTM_Nakabayashi.pth"

class LSTMClassifier(nn.Module):
        # モデルで使う各ネットワークをコンストラクタで定義
        def __init__(self):
            super(LSTMClassifier, self).__init__()
            self.input_layer = nn.Linear(SENSOR_NUM, HIDDEN_DIM)
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

serialconnection = SerialConnection("COM7",115200)

model = LSTMClassifier().to(device)

model.load_state_dict(torch.load(MODELPATH))
model.eval()
emotion_dict = {0:'Neutral', 1:'Smile', 2:'Surprised', 3:'Sad', 4:'Angry'}
#sequencedata = np.empty(20)
sequencedata = np.arange(20*16).reshape(20,16)
print(sequencedata.shape)
count = 0
while(1):
    serialconnection.UpdateSensorData()
    #sequencedata = np.append(sequencedata, serialconnection.getSensorData())
    sequencedata = np.insert(sequencedata, 20, np.array(serialconnection.getSensorData()), axis = 0)
    count += 1
    #print(sequencedata.shape)
    
    sequencedata = np.delete(sequencedata, 0,axis=0)
    #print(np.array(sequencedata).shape)
    #if len(sequencedata) == 21:#時系列データ作成
    if count > 20:
        sequencedata = np.array(sequencedata).reshape(1,20,16)

        inputs = torch.from_numpy(sequencedata).float()#時系列データをテンソルへ
        
        outputs = model(inputs.to(device))#モデルへ入力
        _, preds = torch.max(outputs, 1)#予測ラベル
        preds = str(preds)
        client.sendto(preds.encode('utf-8'),(HOST,PORT))
        print(emotion_dict[preds.item()])
        sequencedata = np.array(sequencedata).reshape(20,16)
    time.sleep(0.023)