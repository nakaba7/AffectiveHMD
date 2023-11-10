import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from DeleteSpike import delete_spike
import matplotlib.pyplot as plt

SENSOR_NUM = 16
HEAD_DIRECTION_DATA_NUM = 2
SEQUENCE_RANGE = 100

name_list = ["Arai", "Nakabayashi", "Ozaki", "Nakagawa", "Nitta", "Kawamura", "Mashiyama", "Haba", "Hamano", "Takechi"]
name_list = ["Ozaki", "Nakagawa", "Nitta", "Kawamura", "Mashiyama", "Haba", "Hamano", "Takechi"]
#新しいラベルを表情遷移1回分ごとに区切ったデータを全セット格納

def mkSequenceDataforPCA(filename, sequence_range):#[[SEQUENCE_RANGE*2個の18次元連続データ], [同じ]...]をtrain_xとして返す．学習時とは違い，データ間に被っている18次元データなし．new_targetは1次元全ラベルデータのコピー用
    """
    datasize : 訓練＋評価データのサイズ
    data_length : 過去何個分のデータを参考にするか
    train_x : 時系列データをセットにした全データの3次元テンソル
    train_t : 予想されたテンソルの正解ラベル
    """

    x_train_list, y_train_list = delete_spike(filename)#メディアンフィルタをかけたセンサデータ x_train_listと無加工のラベル y_train_list
    
    y_train_list = np.squeeze(y_train_list)
    #print(x_train_list)
    #print(y_train_list)
    median_valueData = x_train_list
    new_target = y_train_list
    x_train_list = x_train_list.tolist()
    y_train_list = y_train_list.tolist()

    train_x = []
    train_t = []
    label_change_index_list = []#ラベルの変わった直後の全インデックスを保存
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
        
    return train_x, train_t, new_target, median_valueData, label_change_index_list

def RelabelByPCA(x, labels):#データセット全体に対し，表情遷移1回ごとに区切った新しいラベルリストを生成．
    print("x",x.shape)
    print("labels",labels.shape)
    # 主成分分析（PCA）を実行する
    new_labels_list = []
    
    for i in range(x.shape[0]):#全ての時系列データセットに対してPCAを実行
        
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(x[i])
        model = KMeans(n_clusters=2, random_state=0, init='random')
        model.fit(X_pca)
        clusters = model.predict(X_pca)  # データが属するクラスターのラベルを取得

        prev_class = clusters[0]
        border_index_list = []#前後でクラス変化のあったインデックスを全て格納する．インデックスは変化”後”のもの
        change_counter = 0
        for l in range(len(clusters)):
            if clusters[l] != prev_class:
                border_index_list.append(l)
                change_counter += 1
            prev_class = clusters[l]
        idx = np.abs(np.asarray(border_index_list) - SEQUENCE_RANGE).argmin()#真ん中に一番近いインデックスを取得
        border_index = border_index_list[idx]#クラス変化直後のインデックスを決定

        former_label = labels[i][0]
        latter_label = labels[i][-1]
        former_class = clusters[border_index-1]
        latter_class = clusters[border_index]

        #print("prev clusters",clusters)
        clusters[:border_index] = former_class
        clusters[border_index :] = latter_class#クラス境界以降を一律で同じクラスにする（2回ラベルが変わることを防ぐ）
        new_labels = []#表情遷移1回のみを含むSEQUENCE_RANGE*2個のデータの新しいラベルを格納
        #print("After clusters",clusters)
        for j in range(SEQUENCE_RANGE*2):
            
            if clusters[j] == former_class:#former
                new_labels.append(former_label)
            elif clusters[j] == latter_class:#latter
                new_labels.append(latter_label)
      
        new_labels_list.append(new_labels)
        if change_counter > 100:
            print("border_list", border_index_list)
            print("idx",idx)
            print("border_index",border_index)
            
            for k in range(SEQUENCE_RANGE*2):
                color = ''
                if clusters[k] == 0:
                    color = 'k'
                elif clusters[k] == 1:
                    color = 'm'
                else:
                    print("Error")
                    break
                if labels[i][k] == 0:#Neutral
                    #color = 'k'
                    marker = '.'
                elif labels[i][k] == 1:#Smile
                    #color = 'm'
                    marker = ','
                elif labels[i][k] == 2:#Surprised
                    #color = 'y'
                    marker = 'v'
                elif labels[i][k] == 3:#Sad
                    #color = 'b'
                    marker = 'x'
                elif labels[i][k] == 4:#Angry 
                    #color = 'r'
                    marker = '*'
                else:
                    print("Error")
                    break
                plt.scatter(k, X_pca[k][0], c=color, marker=marker)
            plt.show()

    return np.array(new_labels_list)

def saveCSV(name, order):
    input_filename = 'C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\PreRelabelDataSet\\{0}_DataSet_{1}.csv'.format(name, order)
    output_filename = 'C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_{0}_DataSet_{1}.csv'.format(name, order)
    data, target, new_target, median_valueData, label_change_index_list = mkSequenceDataforPCA(input_filename, SEQUENCE_RANGE)
    #new_targetは区切りのない全データに対する新しいラベルを格納．これを学習に使うラベルにする．

    new_labels_list = RelabelByPCA(data, target)#変更後のラベルリスト

    old_target = new_target.copy()#変更前のラベル

    sequence_range = SEQUENCE_RANGE
    sequence_range = int(sequence_range)
    print("new_target",new_target.shape)
    print("new_labels_list", new_labels_list.shape)
    for i in range(new_labels_list.shape[0]):#PCAで更新されたラベルをnew_targetに書き戻し
        insert_index = label_change_index_list[i]
        first_index = insert_index-sequence_range
        last_index = insert_index+sequence_range
        for j in range(first_index, last_index):#元のラベルをnew_labelで置き換える
            #print("j", j)
            new_target[j] = str(new_labels_list[i][j-first_index])

    new_target = new_target.reshape(new_target.shape[0],1)
    newdata = np.concatenate([new_target, median_valueData], 1)
    np.savetxt(output_filename, newdata, fmt="%s", delimiter=',')
    print("saved!",output_filename)


def saveCSV_1(name):
    input_filename = 'C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\PreRelabelDataSet\\{0}_DataSet.csv'.format(name)
    output_filename = 'C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_{0}_DataSet.csv'.format(name)
    data, target, new_target, median_valueData, label_change_index_list = mkSequenceDataforPCA(input_filename, SEQUENCE_RANGE)
    #new_targetは区切りのない全データに対する新しいラベルを格納．これを学習に使うラベルにする．

    new_labels_list = RelabelByPCA(data, target)#変更後のラベルリスト

    old_target = new_target.copy()#変更前のラベル

    sequence_range = SEQUENCE_RANGE
    sequence_range = int(sequence_range)
    print("new_target",new_target.shape)
    print("new_labels_list", new_labels_list.shape)
    for i in range(new_labels_list.shape[0]):#PCAで更新されたラベルをnew_targetに書き戻し
        insert_index = label_change_index_list[i]
        first_index = insert_index-sequence_range
        last_index = insert_index+sequence_range
        for j in range(first_index, last_index):#元のラベルをnew_labelで置き換える
            #print("j", j)
            new_target[j] = str(new_labels_list[i][j-first_index])

    new_target = new_target.reshape(new_target.shape[0],1)
    newdata = np.concatenate([new_target, median_valueData], 1)
    np.savetxt(output_filename, newdata, fmt="%s", delimiter=',')
    print("saved!",output_filename)

"""

for name in name_list:
    saveCSV(name, "First")
    saveCSV(name, "Second")
    saveCSV(name, "Third")

"""
saveCSV_1("Nakabayashi_Double_Demo")