import numpy as np
import matplotlib.pyplot as plt

"""
表情遷移の境目を可視化するコード
入力: 正解ラベル貼替済みのデータセット
出力: 各表情遷移の境目を可視化したグラフ. 正解ラベルで色分けがされる.
"""

participant_name = "Nakabayashi" #可視化対象の参加者のデータセット

SENSOR_NUM = 16
HEAD_DIRECTION_DATA_NUM = 2
START_INDEX = 500
FIN_INDEX = 1000

neutral_list = []
smile_list = []
surprised_list = []
sad_list = []
angry_list = []

alldata = np.loadtxt('.\\RelabeledDataSet\\Relabeled_{}_DataSet.csv'.format(participant_name), delimiter=',', dtype='str')
for i in range(START_INDEX, FIN_INDEX):
    if alldata[i][0] == 'a':
        continue
    if alldata[i][0] == '0':
        np.put(alldata[i], 0, i)
        neutral_list.append(alldata[i].astype(np.float32))
    elif alldata[i][0] == '1':
        np.put(alldata[i], 0, i)
        smile_list.append(alldata[i].astype(np.float32))
    elif alldata[i][0] == '2':
        np.put(alldata[i], 0, i)
        surprised_list.append(alldata[i].astype(np.float32))
    elif alldata[i][0] == '3':
        np.put(alldata[i], 0, i)
        sad_list.append(alldata[i].astype(np.float32))
    elif alldata[i][0] == '4':
        np.put(alldata[i], 0, i)
        angry_list.append(alldata[i].astype(np.float32))
print("scan done")
def emotion_scatter_plot(list, color):
    list = np.array(list)
    if list.shape[0] > 0:
        for i in range(1,16):
            plt.scatter(list[:,0], list[:,i], c=color, s=0.1)

emotion_scatter_plot(neutral_list, 'k')
print("scatter 0 done")
emotion_scatter_plot(smile_list,'m')
print("scatter 1 done")
emotion_scatter_plot(surprised_list, 'y')
print("scatter 2 done")
emotion_scatter_plot(sad_list, 'b')
print("scatter 3 done")
emotion_scatter_plot(angry_list, 'r')
print("scatter 4 done")
plt.show()