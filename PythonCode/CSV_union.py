import pandas as pd
import numpy as np
import glob
import os

name = "Nakabayashi_Double_Demo"

# パスで指定したファイルの一覧をリスト形式で取得. （ここでは一階層下のtestファイル以下）
csv_files = glob.glob('C:/Users/yukin/AffectiveHMD with Head Pose/CaptureData_csv/{}/*.csv'.format(name))

#読み込むファイルのリストを表示
csv_files.sort(key=os.path.getmtime)
for a in csv_files:
    print(a)

data_list = []
for file in csv_files:
    data_list.append(pd.read_csv(file).to_numpy())
data_list = np.concatenate(data_list)
np.savetxt('C:/Users/yukin/AffectiveHMD with Head Pose/CaptureData_csv/PreRelabelDataSet/{}_DataSet.csv'.format(name), data_list, fmt="%s", delimiter=',')
"""
#csvファイルの中身を追加していくリストを用意
First_data_list = []
Second_data_list = []
Third_data_list = []
#読み込むファイルのリストを走査
i=0
for file in csv_files:
    if i<5:
        print("First",file)
        First_data_list.append(pd.read_csv(file).to_numpy())
    elif i < 10:
        print("Second",file)
        Second_data_list.append(pd.read_csv(file).to_numpy())
    else:
        print("Third", file)
        Third_data_list.append(pd.read_csv(file).to_numpy())
    i+=1

First_dataset = np.concatenate(First_data_list)
Second_dataset = np.concatenate(Second_data_list)
Third_dataset = np.concatenate(Third_data_list)

print("First_shape",First_dataset.shape)
print("Second_shape",Second_dataset.shape)
print("Third_shape",Third_dataset.shape)
np.savetxt('C:/Users/yukin/AffectiveHMD with Head Pose/CaptureData_csv/PreRelabelDataSet/{}_DataSet_First.csv'.format(name), First_dataset, fmt="%s", delimiter=',')
np.savetxt('C:/Users/yukin/AffectiveHMD with Head Pose/CaptureData_csv/PreRelabelDataSet/{}_DataSet_Second.csv'.format(name), Second_dataset, fmt="%s", delimiter=',')
np.savetxt('C:/Users/yukin/AffectiveHMD with Head Pose/CaptureData_csv/PreRelabelDataSet/{}_DataSet_Third.csv'.format(name), Third_dataset, fmt="%s", delimiter=',')

"""