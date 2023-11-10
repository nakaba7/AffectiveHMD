import numpy as np

import pandas as pd
SENSOR_NUM = 16
HEAD_DIRECTION_DATA_NUM = 2
SEQUENCE_RANGE = 100

name = "Nakabayashi"

#新しいラベルを表情遷移1回分ごとに区切ったデータを全セット格納

def mkSequenceDataforPCA(filename, sequence_range):#[[SEQUENCE_RANGE*2個の18次元連続データ], [同じ]...]をtrain_xとして返す．学習時とは違い，データ間に被っている18次元データなし．new_targetは1次元全ラベルデータのコピー用
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

    #if is_normalize == True:#列で正規化
     #   sensor_data = (sensor_data - np.mean(sensor_data, axis = 0)) / np.std(sensor_data, axis = 0)
    #print(sensor_data)
    #x_train, y_train = sensor_data[:data_size], label_data[:data_size]
    value_data, label_data = sensor_data[:], label_data[:]
    x_train_list = value_data.to_numpy().tolist()       
    y_train_list = label_data.to_numpy().tolist()
    
    y_train_list = np.squeeze(y_train_list)

    train_x = []
    train_t = []
    target = []
    label_change_index_list = []#ラベルの変わった直後の全インデックスを保存
    #print(y_train_list)
    prev_label = y_train_list[0]
    data_size = np.array(y_train_list).shape[0]
    changeFlag=False
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0  
    count=0
    for i in range(0,data_size):
        tmplist = []#sequence data list
        tmpLabelList = []
        if y_train_list[i] == 'a':
            if i!=data_size-1:
                prev_label = y_train_list[i+1]
            continue
        if y_train_list[i] != prev_label:
            #print(i)
            label_change_index_list.append(i)#iはラベルの変わった直後
            #tmpLabelList.append(y_train_list[i-sequence_range-1:i])#変わる前の時系列数個のラベルを保存
            #tmplist.append(x_train_list[i-sequence_range-1:i])#変わる前の時系列数個のデータを保存
            train_t.append(y_train_list[i-sequence_range:i+1])
            train_x.append(x_train_list[i-sequence_range:i+1])
            #target.append(y_train_list[i])
            changeFlag=True
        else:
            if changeFlag:
                count += 1
                if count == 15:
                    if y_train_list[i] == "0":
                        if count_0 < 15:
                            train_t.append(y_train_list[i:i+sequence_range+1])
                            train_x.append(x_train_list[i:i+sequence_range+1])
                            changeFlag=False
                            count_0 += 1
                        count = 0
                    elif y_train_list[i] == "1":
                        if count_1 < 15:
                            train_t.append(y_train_list[i:i+sequence_range+1])
                            train_x.append(x_train_list[i:i+sequence_range+1])
                            changeFlag=False
                            count_1 += 1
                        count = 0
                    elif y_train_list[i] == "2":
                        if count_2 < 15:
                            train_t.append(y_train_list[i:i+sequence_range+1])
                            train_x.append(x_train_list[i:i+sequence_range+1])
                            changeFlag=False
                            count_2 += 1
                        count = 0
                    elif y_train_list[i] == "3":
                        if count_3 < 15:
                            train_t.append(y_train_list[i:i+sequence_range+1])
                            train_x.append(x_train_list[i:i+sequence_range+1])
                            changeFlag=False
                            count_3 += 1
                        count = 0
                    elif y_train_list[i] == "4":
                        if count_4 < 15:
                            train_t.append(y_train_list[i:i+sequence_range+1])
                            train_x.append(x_train_list[i:i+sequence_range+1])
                            changeFlag=False
                            count_4 += 1
                        count = 0
        prev_label = y_train_list[i]
    train_x = np.array(train_x)
    train_t = np.array(train_t)
    #target = np.array(target)
    train_t = train_t.astype(np.int32)
    train_x = train_x.astype(np.float32)
    print("0",count_0)
    print("1",count_1)
    print("2",count_2)
    print("3",count_3)
    print("4",count_4)
        
    return train_x, train_t, target

def saveCSV():
    
    input_filename = 'C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_{}_DataSet.csv'.format(name)
    output_filename = 'C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\ChangeMomentDataSet\\Change_{}_DataSet.csv'.format(name)
   
    data, dec_input,_ = mkSequenceDataforPCA(input_filename, SEQUENCE_RANGE)
    #new_targetは区切りのない全データに対する新しいラベルを格納．これを学習に使うラベルにする．
    print("data",data.shape)
    print("dec_input",dec_input.shape)
    #print("target",target.shape)
    for i in range(dec_input.shape[0]):
      print(dec_input[i])
    #print(target)
    #target = target.reshape(target.shape[0],1)
    dec_input = dec_input.reshape([dec_input.shape[0]*dec_input.shape[1], 1])
    data = data.reshape([data.shape[0]*data.shape[1], data.shape[2]])
    newdata = np.concatenate([dec_input, data], 1)
    np.savetxt(output_filename, newdata, fmt="%s", delimiter=',')
    print("saved!",output_filename)

saveCSV()


