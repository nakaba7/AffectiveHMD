import pandas as pd
import numpy as np


def save_123csv(name, order1, order2):
    first_file = "C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_{0}_DataSet_{1}.csv".format(name, order1)
    second_file = "C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_{0}_DataSet_{1}.csv".format(name, order2)
    print("concat", first_file, "and", second_file)
    #csvファイルの中身を追加していくリストを用意
    newdatalist = []
    #読み込むファイルのリストを走査
    
    newdatalist.append(pd.read_csv(first_file).to_numpy())
    newdatalist.append(pd.read_csv(second_file).to_numpy())
           
    Newdataset = np.concatenate(newdatalist)
    print("shape",Newdataset.shape)
    np.savetxt("C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_{0}_DataSet_{1}_{2}.csv".format(name, order1, order2), Newdataset, fmt="%s", delimiter=',')
    print("save!","C:\\Users\\yukin\\AffectiveHMD with Head Pose\\CaptureData_csv\\RelabeledDataSet\\Relabeled_{0}_DataSet_{1}_{2}.csv".format(name, order1, order2))


name_list = ["Arai", "Ozaki", "Nakagawa", "Nitta", "Kawamura", "Mashiyama", "Haba", "Hamano", "Takechi"]

for name in name_list:
    save_123csv(name, "First", "Second")
    save_123csv(name, "Second", "Third")
    save_123csv(name, "Third", "First")
    
