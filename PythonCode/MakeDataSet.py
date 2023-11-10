import numpy as np
import pandas as pd

"""
学習データセットを作る関数
時系列データも作成可能
"""

SENSOR_NUM = 16
SEQUENCE_TENSOR_LENGTH = 20

def zscore(x, axis = None):#標準化
    xmean = np.mean(x, axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd

    return zscore

def mkDataSet(filename, headdatanum, is_sequence_dataset = False, is_normalize = False):
    """
    入力: 
    filename: データを格納したcsvファイル. ラベル1次元, 反射型光センサの値16次元, 頭部姿勢データ headdatanum次元
    headdatanum: 頭部姿勢データの数
    is_sequence_dataset: 時系列を考慮する場合はTrue, しない場合はFalse
    is_normalize: 標準化をする場合はTrue, しないならFalse
    """
    print("dataset filename = {}".format(filename))
    if is_sequence_dataset:
        data_length = SEQUENCE_TENSOR_LENGTH
    else:
        data_length = 0
    df = pd.read_csv(filename, header=None)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)#Delete rows and columns with NaNs
    sensor_data = df.iloc[:, 1:SENSOR_NUM + headdatanum + 1]#get data without 1st row
    label_data = df.iloc[:, 0]#get label in 1st row

    #if is_normalize == True:#列で正規化
    #   sensor_data = (sensor_data - np.mean(sensor_data, axis = 0)) / np.std(sensor_data, axis = 0)
    #print(sensor_data)
    #x_train, y_train = sensor_data[:data_size], label_data[:data_size]
    value_data, label_data = sensor_data[:], label_data[:]
    data_size = label_data.to_numpy().shape[0]
    x_train_list = value_data.to_numpy().tolist()       
    y_train_list = label_data.to_numpy().tolist()
    train_x = []
    train_t = []
    data_size -= data_length
    for i in range(data_size):
        isStopToken = False
        tmplist = []#sequence data list
        for j in range(data_length):
            if y_train_list[i + j] == 'a' or y_train_list[i+data_length] == 'a':
              isStopToken = True
              break
            tmplist.append(x_train_list[i + j])
        if isStopToken:
            continue
        train_x.append(tmplist)
        train_t.append(y_train_list[i+data_length])  #時系列データ数+1番目のラベルを予測
    
    train_x = np.array(train_x)
    train_t = np.array(train_t)
    train_t = train_t.astype(np.int32)
    train_x = train_x.astype(np.float32)
    if is_normalize == True:#列で正規化
        #train_x = (train_x - np.mean(data_list_copy, axis = 0)) / np.std(data_list_copy, axis = 0)
        x0 = train_x.shape[0]
        x1 = train_x.shape[1]
        
        train_x = train_x.reshape([x0*x1, SENSOR_NUM + headdatanum])
        
        train_x = train_x.astype(np.float32)
        
        train_x = zscore(train_x, axis = 0)
        train_x = train_x.reshape([x0, x1, SENSOR_NUM + headdatanum]) 
    
    return train_x, train_t
