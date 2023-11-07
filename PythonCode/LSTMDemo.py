import numpy as np
import torch
import time
import numpy as np
from SerialConnection16 import SerialConnection
import socket
from LSTMModel import LSTMClassifier

"""
学習したLSTMを使って, リアルタイムにセンサ値から表情を識別するデモ
Unityとシリアル通信を行う
"""

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

serialconnection = SerialConnection("COM7",115200)

model = LSTMClassifier(headdatanum=2).to(device)

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