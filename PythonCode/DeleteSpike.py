import numpy as np
import statistics
import pandas as pd


SENSOR_NUM = 16
HEAD_DIRECTION_DATA_NUM = 2

def delete_spike(inputfilename, participantname):
    """
    時刻iのデータがスパイクと判定された場合、i-2, i-1, i, i+1, i+2のデータの中央値とすることで、スパイク除去をする.
    データと中央値の絶対値の差が30以上の場合にスパイクと判定される.

    入力: ラベル, 表情, 頭部姿勢の入ったcsvファイル
    出力: スパイク除去のcsvファイル
    """
    
    df = pd.read_csv(inputfilename, header=None)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)#Delete rows and columns with NaNs
    value_data = df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]]#get data without 1st row
    label_data = df.iloc[:, 0]#get label in 1st row
    x_train_list = value_data.to_numpy()
    y_train_list = label_data.to_numpy()
   
    #初期化
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
    np.savetxt('.\\Spike_Removed_csv\\Median_{0}.csv'.format(participantname),newdata, fmt="%s", delimiter=',')
