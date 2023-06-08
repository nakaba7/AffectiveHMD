import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import math
from sklearn.metrics import confusion_matrix
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import itertools


SENSOR_NUM = 16
HEAD_DIRECTION_DATA_NUM = 2
DATA_PER_TENSOR = 10
BATCH_SIZE = 300
#LEARNING_RATE = 0.003
EPOCH_NUM = 1000
EMOTION_NUM = 5
DATASET_SIZE = 10000
SEQUENCE_TENSOR_LENGTH = 60
#DECODER_RATIO = 0.1
HEAD_NUM = 8
HIDDEN_DIM = 512
OPTIMIZER_LEARNING_RATE = 1.0e-8  #Default:0.0001
DROPOUT = 0.3

def mkDataSet(filename, data_size, data_length,is_normalize = True):
    """
    datasize : 訓練＋評価データのサイズ
    data_length : 過去何個分のデータを参考にするか
    train_x : 時系列データをセットにした全データの3次元テンソル
    train_t : 予想されたテンソルの正解ラベル
    """
    print("dataset filename = {}".format(filename))
    df = pd.read_csv(filename, header=None)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)#Delete rows and columns with NaNs
    sensor_data = df.iloc[:, 1:SENSOR_NUM + HEAD_DIRECTION_DATA_NUM + 1]#get data without 1st row
    label_data = df.iloc[:, 0]#get label in 1st row
    
    value_data, label_data = sensor_data[:], label_data[:]
    x_train_list = value_data.to_numpy().tolist()     
    #print(label_data.to_numpy().shape)  
    y_train_list = label_data.to_numpy().tolist()
    #print(len(x_train_list))
    
    train_x = []
    train_t = []
    #print(y_train_list)
    for i in range(0,data_size,data_length):
        isStopToken = False
        tmplist = []#sequence data list
        tmpLabelList = []
        for j in range(data_length):#時系列データ数だけループ
            if y_train_list[i + j] == 'a':#StopTokenがきたら，それを含んだ一まとまりのデータを削除
              isStopToken = True
              break
            tmplist.append(x_train_list[i + j])#1つの時系列データをtmplistへ入れる
            tmpLabelList.append(y_train_list[i + j])
        if isStopToken:
          continue
        train_x.append(tmplist)       
        train_t.append(tmpLabelList)   
    train_x = np.array(train_x)
    train_t = np.array(train_t)
    train_t = train_t.astype(np.int32)
    train_x = train_x.astype(np.float32)
        
    return train_x, train_t

data, target = mkDataSet('C:\\Users\\yukin\\Downloads\\ReducedDataSet.csv', DATASET_SIZE, SEQUENCE_TENSOR_LENGTH, is_normalize=False)
print(data)
#print(data)
"""
# 2次元のデータセットを生成する
X = data[200]
#print(X)
# 主成分分析（PCA）を実行する
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# 主成分分析の結果をプロットする
plt.scatter(X_pca[:, 0], X_pca[:, 1])
#plt.scatter(np.arange(0,X_pca.shape[0]),X_pca[:])
# 特に大きな値を示す点を探す
#max_point = np.argmax(np.abs(X_pca), axis=0)
#print(X_pca)
max_dist = np.linalg.norm(X_pca[1]-X_pca[0])
index = 0
for i in range(1,X_pca.shape[0]):
    dist = np.linalg.norm(X_pca[i]-X_pca[i-1])
    if(dist > max_dist):
        max_dist = dist
        index = i
    print(i,X_pca[i])
print("peak",index)
# 特に大きな値を示す点をプロットする
#plt.scatter(X_pca[max_point[0], 0], X_pca[max_point[1], 1], marker='*', c='r')
plt.show()
"""
