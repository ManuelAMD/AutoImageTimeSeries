import os

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from app.data_treatment.load_imgs import *
import matplotlib.pyplot as plt
from app.common.color_tools import *
from sklearn.metrics import multilabel_confusion_matrix

def get_Positions(data, rows= 122, cols=360):
    elements = []
    for i in data:
        ix = int(i/cols)
        iy = int(np.round(((i/cols)-ix)*cols))
        elements.append((ix,iy))
    #print(index/cols)
    #print((index/cols)-ix)
    #print(((index/cols)-ix)*cols)
    print("Posiciones!!! {} , {}".format(ix, iy))
    return elements

#Crea cubos con su propia información de tamaño h
def get_cubes(data, h):
    new_data = []
    for i in range(0, len(data)-h):
        new_data.append(data[i:i+h])
    new_data = np.array(new_data)
    print(new_data.shape)
    return new_data

def recolor(args):
    data, pallete = args
    #print("Empezando a colorear")
    aux = []
    #for i in data:
    #    aux2 = []
    for i in data:
        res = gray_quantized(i, pallete)
        res = recolor_greys_image(res, pallete)
        aux.append(res)
    #    aux.append(np.array(aux2))
    #print("He terminado de colorear")
    return np.array(aux)

def evaluation(args):
    test, pred, naive, l_clas = args
    f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    n = np.zeros((l_clas, l_clas), dtype=np.uint64)
    for k in range(test.shape[0]):
        for i in range(test.shape[1]):
            for j in range(test.shape[2]):
                pos1 = np.where(classes == test[k, i, j])[0][0]
                pos2 = np.where(classes == pred[k, i, j])[0][0]
                pos3 = np.where(classes == naive[k, i, j])[0][0]
                #print(np.where(classes == y_test[e, k, i, j]))
                #print(pos1, pos2, pos3)
                f[pos1, pos2] += 1
                n[pos1, pos3] += 1
    return f, n

def multi_process_recolor(data, pallete):
    args = [(d, pallete) for d in data]
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores-1) as pool:
        with tqdm(total = len(data)) as progress:
            futures = []

            for img in args:
                future = pool.submit(recolor, img)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result)
            #executor.map(recolor, data, pallete)

    return np.array(results)

def multi_process_evaluation(test, prediction, naive, l_clas):
    cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    cm_n = np.zeros((l_clas, l_clas), dtype=np.uint64)
    print(cm_f)
    res_lists = zip(test, prediction, naive)
    args = [(t, p, n, l_clas) for (t, p, n) in res_lists]
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores-1) as pool:
        with tqdm(total= len(test)) as progress:
            futures = []

            for eval in args:
                future = pool.submit(evaluation, eval)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result)
    print(len(result))
    print(result[0])
    print(result[1])
    for result in results:
        cm_f = np.add(cm_f, result[0])
        cm_n = np.add(cm_n, result[1])
    return cm_f, cm_n

classes = np.array([0, 255, 220, 177, 119, 70, 35]) # 255, 220, 177, 119, 70, 35  0
classes_rgb = np.array([[0,0,0], [35,35,35], [70,70,70], [119,119,119], [177,177,177], [220,220,220], [255,255,255]])
rows = 260
cols = 640
h = 4

data = np.load("Models/PredictionsConvolutionLSTM_forecast_1.npy")
x_test = np.load("Models/x_test_convlstm_greys_forecast.npy") 
y_test = np.load("Models/y_test_convlstm_greys_forecast.npy")

print(data.shape)
print(x_test.shape)
print(y_test.shape)

y_test = get_cubes(y_test, h)

colors = get_colors(x_test[-10,0])
print("COLORSS", colors)
print("COLORS", colors.shape)

colorss = get_colors(data[-10,0])
print("COLORSS", colorss)

naive = x_test[:-4]
data = data[1:-3]

#y_real = y_test[:, -h:]*255
new_data = data[:, -h:]
n_real = naive[:, -h:]*255

#y_test = y_test[:, -h:]
naive = naive[:, -h:]

print("XX")
print(y_test.shape)
print(new_data.shape)
print(n_real.shape)

print(min(new_data[0,0,60]))
print(max(new_data[0,0,60]))

new_data = new_data * 255
new_data = new_data.astype(np.uint8)

print("HEY", new_data.shape)
print(colorss.shape)
print(min(new_data[0,0,60]))
print(max(new_data[0,0,60]))

new_data = new_data.reshape(new_data.shape[:-1])
print("HoY", new_data.shape)


""" Original
aux = []
print(type(new_data))
for i in new_data:
    aux2 = []
    for j in i:
        #res = cv2.cvtColor(j, cv2.COLOR_GRAY2RGB)
        #res = recolor_greys_image(j, classes)
        #rgb_quantized(res, classes_rgb)
        #res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        res = gray_quantized(j, classes)
        res = recolor_greys_image(res, classes)
        aux2.append(res)
    aux.append(np.array(aux2))
new_data = np.array(aux)"""

new_data = multi_process_recolor(new_data, classes)


print("SHAPEE", new_data.shape)
color_data = get_colors(new_data[-10,0])
print("DCOLORS", color_data)
new_data = new_data.reshape(new_data.shape[0],new_data.shape[1],new_data.shape[2],new_data.shape[3],1)

#y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2]))*255
#naive = naive.reshape((naive.shape[0], naive.shape[1], naive.shape[2])) * 255

plt.imshow(y_test[0,0], cmap="gray")
#plt.show()


plt.imshow(new_data[0,0], cmap="gray")
#plt.show()


plt.imshow(naive[0,0], cmap="gray")
#plt.show()

y_test = y_test * 255
naive = naive * 255

print("YCOLORS", get_colors(y_test[-10,0]))
print("NCOLORS", get_colors(naive[-10,0]))

print("XS")
print(new_data.shape)
print(y_test.shape)
print(naive.shape)

l_clas = len(classes)

fig = plt.figure(figsize=(10,7))
r = 3
c = 4
ac = 1
pos = 100
for i in range(h):
    fig.add_subplot(r, c, ac)
    ac += 1
    plt.imshow(y_test[pos,i], cmap='gray')
    plt.axis('off')
    plt.title('Original_t+{}'.format(i))
for i in range(h):
    fig.add_subplot(r, c, ac)
    ac += 1
    plt.imshow(new_data[pos,i], cmap='gray')
    plt.axis('off')
    plt.title('Pronóstico_t+{}'.format(i))
for i in range(h):
    fig.add_subplot(r, c, ac)
    ac += 1
    plt.imshow(naive[pos,i], cmap='gray')
    plt.axis('off')
    plt.title('Naive_t+{}'.format(i))

#plt.show()
"""
cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
cm_n = np.zeros((l_clas, l_clas), dtype=np.uint64)
print(cm_f)

for e in range(y_test.shape[0]):
    for k in range(h):
        for i in range(rows):
            for j in range(cols):
                pos1 = np.where(classes == y_test[e, k, i, j])[0][0]
                pos2 = np.where(classes == new_data[e, k, i, j])[0][0]
                pos3 = np.where(classes == naive[e, k, i, j])[0][0]
                #print(np.where(classes == y_test[e, k, i, j]))
                #print(pos1, pos2, pos3)
                cm_f[pos1, pos2] += 1
                cm_n[pos1, pos3] += 1"""
""" Original funcional
cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
cm_n = np.zeros((l_clas, l_clas), dtype=np.uint64)
print(cm_f)

for e in range(y_test.shape[0]):
    for k in range(h):
        for i in range(rows):
            for j in range(cols):
                pos1 = np.where(classes == y_test[e, k, i, j])[0][0]
                pos2 = np.where(classes == new_data[e, k, i, j])[0][0]
                pos3 = np.where(classes == naive[e, k, i, j])[0][0]
                cm_f[pos1, pos2] += 1
                cm_n[pos1, pos3] += 1
"""
cm_f, cm_n = multi_process_evaluation(y_test, new_data, naive, l_clas)
print("Matriz de confusión de pronóstico")
print(cm_f)

len_categories = len(cm_f)

TP = np.zeros(len_categories, np.uint32)
TN = np.zeros(len_categories, np.uint32)
FP = np.zeros(len_categories, np.uint32)
FN = np.zeros(len_categories, np.uint32)

for i in range(len_categories):
    for j in range (len_categories):
        if i == j:
            TP[i] += cm_f[i,j]
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
    if (TP[i] + FP[i]) == 0:
        precision[i] = 0
    else:
        precision[i] = TP[i] / (TP[i] + FP[i])
    if (TP[i] + FN[i]) == 0:
        recall[i] = 0
    else:
        recall[i] = TP[i] / (TP[i] + FN[i])
    if (precision[i] + recall[i]) == 0:
        f1_score[i] = 0
    else:
        f1_score[i] = 2*((precision[i] * recall[i]) /(precision[i] + recall[i]))
    if (TP[i] + TN[i] + FP[i] + FN[i]) == 0:
        accuracy[i] = 0
    else:
        accuracy[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])

print(precision)
print(recall)
print(f1_score)
print(accuracy)

macro_f1 = np.sum(f1_score)/len_categories

print(macro_f1)

print("NAIVEEE!!!!!!!!!!!!!!")

print("Matriz de confusión de naive")
print(cm_n)

len_categories = len(cm_n)

TP = np.zeros(len_categories, np.uint32)
TN = np.zeros(len_categories, np.uint32)
FP = np.zeros(len_categories, np.uint32)
FN = np.zeros(len_categories, np.uint32)

for i in range(len_categories):
    for j in range (len_categories):
        if i == j:
            TP[i] += cm_n[i,j]
            for k in range(len_categories):
                if k != j:
                    FP[i] += cm_n[k, j]
            for k in range(i):
                for l in range(j):
                    TN[i] += cm_n[k, l]
            for k in range(i+1, len_categories):
                for l in range(j+1, len_categories):
                    TN[i] += cm_n[k, l]
            for k in range(i):
                for l in range(j+1, len_categories):
                    TN[i] += cm_n[k, l]
            for k in range(i+1, len_categories):
                for l in range(j):
                    TN[i] += cm_n[k, l]
        else:
            FN[i] += cm_n[i,j]
print(TP)
print(TN)
print(FP)
print(FN)
precision = np.zeros(len_categories, np.float32)
recall = np.zeros(len_categories, np.float32)
f1_score = np.zeros(len_categories, np.float32)
accuracy = np.zeros(len_categories, np.float32)

for i in range(len_categories):
    if (TP[i] + FP[i]) == 0:
        precision[i] = 0
    else:
        precision[i] = TP[i] / (TP[i] + FP[i])
    if (TP[i] + FN[i]) == 0:
        recall[i] = 0
    else:
        recall[i] = TP[i] / (TP[i] + FN[i])
    if (precision[i] + recall[i]) == 0:
        f1_score[i] = 0
    else:
        f1_score[i] = 2*((precision[i] * recall[i]) /(precision[i] + recall[i]))
    if (TP[i] + TN[i] + FP[i] + FN[i]) == 0:
        accuracy[i] = 0
    else:
        accuracy[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])

print(precision)
print(recall)
print(f1_score)
print(accuracy)

macro_f1 = np.sum(f1_score)/len_categories

print(macro_f1)