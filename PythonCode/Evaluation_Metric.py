"""
クラス分類にて使用するPrecision, Recall, F-measureのマクロ平均を返す関数

入力: 
cm: confusion matrix(sklearn.metrics の confusion_matrix関数)
出力: 各評価指標のマクロ平均
"""
#再現率(Recall)
def macro_recall_score(cm):
    recall_list=[]
    for i in range(len(cm[0])):
        row_list=[]
        for j in range(len(cm[0])):
            row_list.append(cm[i][j])
        #print("sum", sum(row_list), "分母", cm[i][i])
        tmp_recall = cm[i][i] / sum(row_list) 
        #print("i",i,"recall", tmp_recall)
        recall_list.append(tmp_recall)
    return sum(recall_list)/len(recall_list)

#適合率(Precision)
def macro_precision_score(cm):
    precision_list=[]
    for i in range(len(cm[0])):
        column_list=[]
        for j in range(len(cm[0])):
            column_list.append(cm[j][i])
        #print("sum", sum(row_list), "分母", cm[i][i])
        tmp_precision = cm[i][i] / sum(column_list) 
        #print("i",i,"recall", tmp_recall)
        precision_list.append(tmp_precision)
    return sum(precision_list)/len(precision_list)

#F値(F1-measure)
def macro_f1_score(cm):
    precision = macro_precision_score(cm)
    recall = macro_recall_score(cm)
    return 2 / (1/precision + 1/recall)