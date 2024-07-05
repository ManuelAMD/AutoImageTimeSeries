import numpy as np
from app.common.color_tools import *
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_cubes(data, h):
    new_data = []
    for i in range(0, len(data)-h):
        new_data.append(data[i:i+h])
    return np.array(new_data)

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
    test, pred, naive, classes, l_clas = args
    f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    n = np.zeros((l_clas, l_clas), dtype=np.uint64)
    for k in range(4):
        for i in range(test.shape[1]):
            for j in range(test.shape[2]):
                pos1 = np.where(classes == test[k, i, j])[0][0]
                pos2 = np.where(classes == pred[k, i, j])[0][0]
                pos3 = np.where(classes == naive[k, i, j])[0][0]
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

def multi_process_evaluation(test, prediction, naive, classes, l_clas):
    cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    cm_n = np.zeros((l_clas, l_clas), dtype=np.uint64)
    print(cm_f)
    res_lists = zip(test, prediction, naive)
    args = [(t, p, n, classes, l_clas) for (t, p, n) in res_lists]
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

def main():
    classes = np.array([0,255,220,177,119,70,35])
    rows = 122
    cols = 360
    h = 4

    data = np.load("Models/PredictionMultiCNN_forecast.npy")
    x_test = np.load("Models/x_test_multicnn_greys.npy")
    y_test = np.load("Models/y_test_multicnn_greys.npy")

    print(data.shape)
    print(x_test.shape)
    print(y_test.shape)

    y_test = get_cubes(y_test, h)

    print(y_test.shape)
    colors = get_colors(x_test[-10,0])
    colorss = get_colors(data[-10,0])

    print("Test colors", colors)
    print("Prediction colors", colorss)
    print("colors Shapes, test: {}, prediction: {}".format(colors.shape, colorss.shape))

    #Eliminar para horizontes más de 1
    #data = data.reshape((data.shape[0],1,*data.shape[1:]))

    naive = x_test[:-4]
    data = data[1:-3]

    prediction = data[:, -h:]

    naive = naive[:, -h:]

    print("XXXXXXX")
    print(y_test.shape)
    print(prediction.shape)
    print(naive.shape)

    print(min(prediction[0,0,60]))
    print(max(prediction[0,0,60]))

    prediction = prediction * 255
    prediction = prediction.astype(np.uint8)

    print("HEYY", prediction.shape)
    print(colorss.shape)
    print(min(prediction[0,0,60]))
    print(max(prediction[0,0,60]))

    prediction = prediction.reshape(prediction.shape[:-1])
    print("HOYYY", prediction.shape)


    prediction = multi_process_recolor(prediction, classes)

    print("SHAPEEE", prediction.shape)
    color_data = get_colors(prediction[-10,0])
    print("PCOLORS", color_data)
    prediction = prediction.reshape(*prediction.shape[:], 1)
    print("SHAPEEE2", prediction.shape)

    plt.imshow(y_test[0,0], cmap="gray")
    plt.show()
    plt.imshow(prediction[0,0], cmap="gray")
    plt.show()
    plt.imshow(naive[0,0], cmap="gray")
    plt.show()

    y_test = y_test * 255
    naive = naive * 255

    print("YCOLORS", get_colors(y_test[-10,0]))
    print("NCOLORS", get_colors(naive[-10,0]))

    print("XS")
    print(prediction.shape)
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
        plt.imshow(prediction[pos,i], cmap='gray')
        plt.axis('off')
        plt.title('Pronóstico_t+{}'.format(i))
    for i in range(h):
        fig.add_subplot(r, c, ac)
        ac += 1
        plt.imshow(naive[pos,i], cmap='gray')
        plt.axis('off')
        plt.title('Naive_t+{}'.format(i))
        
    plt.show()

    cm_f, cm_n = multi_process_evaluation(y_test, prediction, naive, classes, l_clas)
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


if __name__ == "__main__":
    main()