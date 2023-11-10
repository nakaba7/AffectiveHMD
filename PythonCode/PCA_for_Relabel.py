import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

"""
主成分分析をした後, K-means法を使うことで正解ラベルを正しく貼り直す. このファイルを実行することで貼り直しが可能.
"""

SENSOR_NUM = 16
HEAD_DIRECTION_DATA_NUM = 2
SEQUENCE_RANGE = 30
participant_name = "Nakabayashi" #スパイク除去をした誰のデータのラベル貼り直しをするか ./Spike_Removed_csv/Median_{participant_name}.csv 

label_change_index_list = []#ラベルの変わった直後の全インデックスを保存
new_labels_list = []#新しいラベルを表情遷移1回分ごとに区切ったデータを全セット格納

def mkSequenceDataforPCA(filename, sequence_range):#[[SEQUENCE_RANGE*2個の18次元連続データ], [同じ]...]をtrain_xとして返す．学習時とは違い，データ間に被っている18次元データなし．new_targetは1次元全ラベルデータのコピー用
    """
    filename: スパイク除去済みのcsvデータファイル
    sequence_range: 時系列サイズ
    """
    print("dataset filename = {}".format(filename))
    df = pd.read_csv(filename, header=None)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)#Delete rows and columns with NaNs
    sensor_data = df.iloc[:, 1:SENSOR_NUM + HEAD_DIRECTION_DATA_NUM + 1]#get data without 1st row
    label_data = df.iloc[:, 0]#get label in 1st row

    value_data = sensor_data[:]
    x_train_list = value_data.to_numpy().tolist()
    y_train_list = label_data.to_numpy().tolist()
    new_target = label_data.to_numpy()
    train_x = []
    train_t = []
    
    #print(y_train_list)
    prev_label = y_train_list[0]
    data_size = np.array(y_train_list).shape[0]
    
    for i in range(0,data_size):
        tmplist = []#sequence data list
        tmpLabelList = []
        if y_train_list[i] == 'a':
            if i!=data_size-1:
                prev_label = y_train_list[i+1]
            continue
        if y_train_list[i] != prev_label:
            #print(i)
            label_change_index_list.append(i)
            tmpLabelList.append(y_train_list[i-sequence_range:i+sequence_range])
            tmplist.append(x_train_list[i-sequence_range:i+sequence_range])
            #print(np.squeeze(np.array(tmplist)).shape)
            train_t.append(np.squeeze(np.array(tmpLabelList)))
            train_x.append(np.squeeze(np.array(tmplist)))
        prev_label = y_train_list[i]
    train_x = np.array(train_x)
    train_t = np.array(train_t)
    train_t = train_t.astype(np.int32)
    train_x = train_x.astype(np.float32)
        
    return train_x, train_t, new_target


def RelabelByPCA(x, labels):#データセット全体に対し，表情遷移1回ごとに区切った新しいラベルリストを生成．
    # 主成分分析（PCA）を実行する
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(x)
    model = KMeans(n_clusters=2, random_state=0, init='random')
    model.fit(X_pca)
    clusters = model.predict(X_pca)  # データが属するクラスターのラベルを取得
    former_label = labels[0]
    latter_label = labels[-1]
    former_class = clusters[0]
    latter_class = clusters[-1]
    
    new_labels = []#表情遷移1回のみを含むSEQUENCE_RANGE*2個のデータの新しいラベルを格納
    for i in range(X_pca.shape[0]):
        """
        color = ''
        marker = ''
        if labels[i] == 0:#Neutral
            #color = 'k'
            marker = '.'
        elif labels[i] == 1:#Smile
            #color = 'm'
            marker = ','
        elif labels[i] == 2:#Surprised
            #color = 'y'
            marker = 'v'
        elif labels[i] == 3:#Sad
            #color = 'b'
            marker = 'x'
        elif labels[i] == 4:#Angry 
            #color = 'r'
            marker = '*'
        else:
            print("Error")
            break
        """
        if clusters[i] == former_class:#former
            #color = 'r'
            new_labels.append(former_label)
        elif clusters[i] == latter_class:#latter
            #color = 'b'
            new_labels.append(latter_label)
        #plt.scatter(i, X_pca[i][0], c=color,marker=marker)
    #plt.show()
    new_labels_list.append(new_labels)

data, target, new_target = mkSequenceDataforPCA('.\\Spike_Removed_csv\\Median_{0}.csv'.format(participant_name), SEQUENCE_RANGE)#new_targetは区切りのない全データに対する新しいラベルを格納．これを学習に使うラベルにする．

for i in range(data.shape[0]):#全ての時系列データセットに対してPCAを実行
    RelabelByPCA(data[i],target[i])
new_labels_list = np.array(new_labels_list)#変更後のラベルリスト
old_target = new_target.copy()#変更前のラベル

sequence_range = int(SEQUENCE_RANGE)

for i in range(new_labels_list.shape[0]):#1次元リストのラベルデータに書き戻し
    insert_index = label_change_index_list[i]
    first_index = insert_index-sequence_range
    last_index = insert_index+sequence_range
    for j in range(first_index, last_index):#元のラベルをnew_labelで置き換える
        new_target[j] = str(new_labels_list[i][j-first_index])


for i in range(sequence_range, new_target.shape[0], sequence_range*2):
    #if not(np.allclose(new_target[i-5:i+5], old_target[i-5:i+5])):
    print(i)
    print("old",old_target[i-sequence_range:i+sequence_range])
    print("new",new_target[i-sequence_range:i+sequence_range])
    print("-------------------------------------------------------")

#このコードの成果物はnew_target．これを元の18次元データの新しいラベルとすれば，正しくラベルの分けられたデータセットができる．
