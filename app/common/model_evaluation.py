import numpy as np
from app.common.color_tools import *
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_cubes(data, h):
    new_data = []
    for i in range (0, len(data) - h):
        new_data.append(data[i:i+h])
    new_data = np.array(new_data)
    print(new_data.shape)
    return new_data

def recolor(args):
    data, pallete = args
    aux = []
    for i in data:
        res = gray_quantized(i, pallete)
        res = recolor_greys_image(res, pallete)
        aux.append(res)
    return np.array(aux)

def evaluation(args):
    test, pred, l_clas = args
    f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    for k in range(test.shape[0]):
        for i in range(test.shape[1]):
            for j in range(test.shape[2]):
                pos1 = np.where(classes == test[k, i, j])[0][0]
                pos2 = np.where(classes == pred[k, i, j])[0][0]
                f[pos1, pos2] += 1
    return f

def multi_process_recolor(data, pallete):
    args = [(d, pallete) for d in data]
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores-4) as pool:
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

    return np.array(results)

def multi_process_evaluation(test, prediction, l_clas):
    cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    print(cm_f)
    res_lists = zip(test, prediction)
    args = [(t, p, l_clas) for (t, p) in res_lists]
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores-4) as pool:
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
    for result in results:
        cm_f = np.add(cm_f, result)
    return cm_f

classes = np.array([0, 255, 220, 177, 119, 70, 35]) # 255, 220, 177, 119, 70, 35  0

def image_confussion_matrix(prediction: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, h: int):
    print(prediction.shape, x_test.shape, y_test.shape)
    y_test = get_cubes(y_test, h)
    colors = get_colors(x_test[-10, 0])
    print("COLORS: ", colors)
    print("Colors shape: ", colors.shape)

    colorss = get_colors(prediction[-10, 0])
    print("COLORSS P: ", colorss)

    naive = x_test[:-h]
    if h > 1:
        prediction = prediction[1:-(h-1)]
    else:
        prediction = prediction[1:]

    new_prediction = prediction[:, -h:]
    print("PRED shape:", prediction.shape)
    #n_real = naive[:, -h:] * 255

    prediction = None

    naive = naive[:, -h:]

    print("XX")
    #print(y_test.shape, new_prediction.shape, n_real.shape)

    print(min(new_prediction[0, 0, 60]))
    print(max(new_prediction[0, 0, 60]))

    new_prediction = new_prediction * 255
    new_prediction = new_prediction.astype(np.uint8)

    print("HEYY", new_prediction.shape)
    print(colorss.shape)
    print(min(new_prediction[0, 0, 60]))
    print(max(new_prediction[0, 0, 60]))

    #new_prediction = new_prediction.reshape(new_prediction.shape[:])
    print("HOOYY", new_prediction.shape)

    """aux = []
    for i in new_prediction:
        aux2 = []
        for j in i:
            res = gray_quantized(j, classes)
            res = recolor_greys_image(res, classes)
            aux2.append(res)
        aux.append(np.array(aux2))
    new_prediction = np.array(aux)"""

    new_prediction = multi_process_recolor(new_prediction, classes)

    print("SHAPEEEE", new_prediction.shape)
    color_data = get_colors(new_prediction[-10, 0])
    print("DCOLORS", color_data)
    #shape = new_prediction.shape
    #new_prediction = new_prediction.reshape(shape[0], shape[1], shape[2], shape[3], 1)

    y_test = y_test * 255
    #naive = naive * 255
    print("YCOLORS", get_colors(y_test[-10, 0]))
    print("XCOLORS", get_colors(naive[-10, 0]))
    print("XSSSS")
    print(new_prediction.shape, y_test.shape)

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
        plt.imshow(new_prediction[pos,i], cmap='gray')
        plt.axis('off')
        plt.title('Pronóstico_t+{}'.format(i))
    for i in range(h):
        fig.add_subplot(r, c, ac)
        ac += 1
        plt.imshow(naive[pos,i], cmap='gray')
        plt.axis('off')
        plt.title('Naive_t+{}'.format(i))

    #plt.show()

    #cm_f = np.zeros((l_clas, l_clas), dtype= np.uint64)
    #cm_n = np.zeros(())
    #print(cm_f)

    #for e in range(y_test.shape[0]):
    #    for k in range(h):
    #        for i in range(y_test.shape[2]):
    #            for j in range(y_test.shape[3]):
    #                pos1 = np.where(classes == y_test[e, k, i, j]) [0][0]
    #                pos2 = np.where(classes == new_prediction[e, k, i, j]) [0][0]
    #                cm_f[pos1, pos2] += 1

    cm_f = multi_process_evaluation(y_test, new_prediction, l_clas)
    
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

    suma = np.sum(f1_score)

    macro_f1 = suma/len_categories

    print(macro_f1)
    return macro_f1