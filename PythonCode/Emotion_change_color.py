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
DATASET_SIZE = 2820
SEQUENCE_RANGE = 100

#ラベルの変わり目をラベルごとに色分けしてグラフ化する

def mkSequenceDataforPCA(filename, data_size, sequence_range):
    """
    datasize : 訓練＋評価データのサイズ
    sequence_range : 過去何個分のデータを参考にするか
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
    y_train_list = label_data.to_numpy().tolist()
    train_x = []
    train_t = []
    #print(y_train_list)
    prev_label = y_train_list[0]
    
    for i in range(0,data_size):
        
        tmplist = []#sequence data list
        tmpLabelList = []
        if y_train_list[i] == 'a':
            if i!=data_size-1:
                prev_label = y_train_list[i+1]
            continue
        if y_train_list[i] != prev_label:
            #print(i)
            tmpLabelList.append(y_train_list[i-sequence_range:i+sequence_range])
            tmplist.append(x_train_list[i-sequence_range:i+sequence_range])
            #print(np.squeeze(np.array(tmplist)).shape)
            train_t.append(np.squeeze(np.array(tmpLabelList)))
            train_x.append(np.squeeze(np.array(tmplist)))
        prev_label = y_train_list[i]
    print("train_x",train_x)
    print("train_t",train_t)
    print("train_x",len(train_x),len(train_x[0]))
    print("train_t",len(train_t),len(train_t[0]))

    train_x = np.array(train_x)
    train_t = np.array(train_t)
    train_x = train_x.astype(np.float32)
    train_t = train_t.astype(np.int32)
    
    return train_x, train_t

data, target = mkSequenceDataforPCA("C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_Mashiyama_DataSet.csv", DATASET_SIZE, SEQUENCE_RANGE)


def RelabelByPCA(x, labels):
    # 主成分分析（PCA）を実行する
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(x)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.ylim(-200, 200) # (4)y軸の表示範囲
    #plt.legend()
    plt.xlabel('Time[s]', fontsize = 16)
    plt.ylabel('First Main Component', fontsize = 16)

    # 主成分分析の結果をプロットする
    for i in range(X_pca.shape[0]):
        color = ''
        if labels[i] == 0:#Neutral
            color = 'k'
        elif labels[i] == 1:#Smile
            color = 'm'
        elif labels[i] == 2:#Surprised
            color = 'y'
        elif labels[i] == 3:#Sad
            color = 'b'
        elif labels[i] == 4:#Angry 
            color = 'r'
        else:
            print("Error")
            break
       
        plt.scatter(i*0.02, X_pca[i][0], c=color)
   
    plt.show()

for i in range(data.shape[0]):
    RelabelByPCA(data[i],target[i])






