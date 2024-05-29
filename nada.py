import numpy as np
"""cm_f = np.array([[25560351, 0, 0, 0, 3, 38, 452],
        [11, 16559007, 2999758, 36586, 2546, 392, 53],
        [9, 579167, 10972312, 833805, 59927, 18376, 367],
        [24, 20005, 372377, 2540684, 526205, 138326, 4805],
        [42, 14174, 45266, 74661, 174088, 36103, 963],
        [84, 21, 8137, 78887, 165813, 1784378, 38197],
        [147, 26, 358, 1692, 4508, 160733, 660696]])

print(cm_f)

len_categories = len(cm_f)

TP = np.zeros(len_categories, np.uint32)
TN = np.zeros(len_categories, np.uint32)
FP = np.zeros(len_categories, np.uint32)
FN = np.zeros(len_categories, np.uint32)
print(TP)
print(TN)
print(FP)
print(FN)

for i in range(len_categories):
    for j in range (len_categories):
        if i == j:
            TP[i] = cm_f[i,j]
            for k in range(len_categories):
                if k != j:
                    FP[i] += cm_f[k, j]
            for k in range(i):
                for l in range(j):
                    TN[i] += cm_f[k, l]
            for k in range(i+1, len_categories):
                for l in range(j+1, len_categories):
                    TN[i] += cm_f[k, l]
            for k in range(i):
                for l in range(j+1, len_categories):
                    TN[i] += cm_f[k, l]
            for k in range(i+1, len_categories):
                for l in range(j):
                    TN[i] += cm_f[k, l]
        else:
            FN[i] += cm_f[i,j]

print(TP)
print(TN)
print(FP)
print(FN)
precision = np.zeros(len_categories, np.float32)
recall = np.zeros(len_categories, np.float32)
f1_score = np.zeros(len_categories, np.float32)
accuracy = np.zeros(len_categories, np.float32)

for i in range(len_categories):
    precision[i] = TP[i] / (TP[i] + FP[i])
    recall[i] = TP[i] / (TP[i] + FN[i])
    f1_score[i] = 2*((precision[i] * recall[i]) /(precision[i] + recall[i]))
    accuracy[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])

print(precision)
print(recall)
print(f1_score)
print(accuracy)

macro_f1 = np.sum(f1_score)/len_categories

print(macro_f1)"""

import os
import glob
import csv
import pandas

path = "SPIReescale/"

os.chdir(path)
lista = glob.glob("*")
lista.sort()
print(lista)

df = pandas.DataFrame(data={"names":lista})
df.to_csv("NamesSPIDataset.csv", sep=',', index=False)


#with open("NamesSPIDataset", "w", newline='') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(list)