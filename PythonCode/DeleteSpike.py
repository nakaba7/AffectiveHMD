import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statistics
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

def delete_spike(filename):
    """
    datasize : 訓練＋評価データのサイズ
    data_length : 過去何個分のデータを参考にするか
    train_x : 時系列データをセットにした全データの3次元テンソル
    train_t : 予想されたテンソルの正解ラベル
    """
    
    df = pd.read_csv(filename, header=None)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)#Delete rows and columns with NaNs
    sensor_data = df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]]#get data without 1st row
    label_data = df.iloc[:, 0]#get label in 1st row
    #print(sensor_data)
    value_data, label_data = sensor_data[:], label_data[:]
    x_train_list = value_data.to_numpy()
    #print(label_data.to_numpy().shape)  
    y_train_list = label_data.to_numpy()
   
    prev_sensor_values = x_train_list[0]
    prev_prev_sensor_values = x_train_list[0]
    next_sensor_values = x_train_list[1]
    next_next_sensor_values = x_train_list[2]

    for i in range(x_train_list.shape[0]-2):#csvファイルのスパイク除去
        if y_train_list[i] == 'a':
            prev_sensor_values = x_train_list[i+1]
            prev_prev_sensor_values = x_train_list[i+1]
            next_sensor_values = x_train_list[i+2]
            next_next_sensor_values = x_train_list[i+3]
            continue
        next_sensor_values = x_train_list[i+1]
        next_next_sensor_values = x_train_list[i+2]
        for j in range(16):
            l = [prev_sensor_values[j], x_train_list[i][j], next_sensor_values[j], prev_prev_sensor_values[j], next_next_sensor_values[j]]
            median = statistics.median(l)
            if x_train_list[i][j] - median > 30 or x_train_list[i][j] - median< -30:
                x_train_list[i][j] = median
        prev_sensor_values = x_train_list[i]
        if i!=0 :prev_prev_sensor_values = x_train_list[i-1]
    labels = y_train_list.reshape(y_train_list.shape[0],1)
    newdata = np.concatenate([labels, x_train_list], 1)
    np.savetxt('C:\\Users\\yukin\\Downloads\\Median_Nakabayashi_Test.csv',newdata, fmt="%s", delimiter=',')
    
delete_spike("C:\\Users\\yukin\\Downloads\\Nakabayashi_Test_DataSet.csv")
