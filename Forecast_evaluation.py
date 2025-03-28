import os
import json
import pandas as pd

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from app.data_treatment.load_imgs import *
import matplotlib.pyplot as plt
from app.common.color_tools import *
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

def read_json_file(filename):
    f = open('configurations/{}'.format(filename), "r")
    parameters = json.load(f)
    print(type(parameters))
    return parameters

def get_positions(data, rows: int, cols: int):
    elements = []
    for i in data:
        ix = int(i/cols)
        iy = int(np.round(((i/cols) - ix) * cols))
        elements.append((ix, iy))
    print("Prosiciones!!! {}, {}".format(ix, iy))

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
    aux = []
    for i in data:
        res = gray_quantized(i, pallete)
        res = recolor_greys_image(res, pallete)
        aux.append(res)
    return np.array(aux)

def evaluation(args):
    test, pred, naive, l_clas, classes = args
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

    return np.array(results)

def multi_process_evaluation(test, prediction, naive, cm_f, cm_n, l_clas, classes):
    #cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
    #cm_n = np.zeros((l_clas, l_clas), dtype=np.uint64)
    print(cm_f)
    res_lists = zip(test, prediction, naive)
    args = [(t, p, n, l_clas, classes) for (t, p, n) in res_lists]
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

def mae_image(y_true, y_pred):
    err = np.sum((y_true.astype('float') - y_pred.astype('float')) ** 2)
    err /= float(y_true.shape[0] * y_true.shape[1])
    return err

def calculate_errors(y_true, y_pred):
    MAE = MSE = RMSE = 0
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            MAE += mean_absolute_error(y_true[i,j], y_pred[i,j])
            MSE += mean_squared_error(y_true[i,j], y_pred[i,j])
            RMSE += root_mean_squared_error(y_true[i,j], y_pred[i,j])

    cant = y_true.shape[0] * y_true.shape[1]
    if MAE == 0:
        MAE = 0
    else:
        MAE /= cant
    if MSE == 0:
        MSE = 0
    else:
        MSE /= cant
    if RMSE == 0:
        RMSE = 0
    else:
        RMSE /= cant
    return MAE, MSE, RMSE

def confusion_matrix_evaluation(confusion_matrix):
    len_categories = len(confusion_matrix)
    TP = np.zeros(len_categories, np.uint32)
    TN = np.zeros(len_categories, np.uint32)
    FP = np.zeros(len_categories, np.uint32)
    FN = np.zeros(len_categories, np.uint32)

    for i in range(len_categories):
        for j in range (len_categories):
            if i == j:
                TP[i] += confusion_matrix[i,j]
                for k in range(len_categories):
                    if k != j:
                        FP[i] += confusion_matrix[k, j]
                for k in range(i):
                    for l in range(j):
                        TN[i] += confusion_matrix[k, l]
                for k in range(i+1, len_categories):
                    for l in range(j+1, len_categories):
                        TN[i] += confusion_matrix[k, l]
                for k in range(i):
                    for l in range(j+1, len_categories):
                        TN[i] += confusion_matrix[k, l]
                for k in range(i+1, len_categories):
                    for l in range(j):
                        TN[i] += confusion_matrix[k, l]
            else:
                FN[i] += confusion_matrix[i,j]
    
    print("Verdaderos positivos", TP)
    print("Verdaderos negativos", TN)
    print("Falsos positivos", FP)
    print("Falsos negativos", FN)

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

    print("Precisión", precision)
    print("Recuperación", recall)
    print("Micro f1_score", f1_score)
    print("Exactitud", accuracy)

    macro_f1 = np.sum(f1_score)/len_categories

    print("Macro f1-score", macro_f1)

    return np.array([TP, TN, FP, FN]), np.array([precision, recall, f1_score, accuracy])

def main(forecast_path, config_file, h: int, display= False):
    config_json = read_json_file(config_file)
    classes = np.array(config_json['classes'])
    rows = config_json['rows']
    cols = config_json['cols']
    horizon = h
    window = config_json['window_size'] -1

    forecasts = np.load("Models/{}".format(forecast_path))
    x_test = np.load("Models/x_test_data.npy")
    y_test = np.load("Models/y_test_data.npy")

    print(forecasts.shape)
    print(x_test.shape)
    print(y_test.shape)

    #Generar cubos de y dependiendo del horizonte para evaluar
    y_test = get_cubes(y_test, horizon)

    #Para el modelo naive se toman todos los elementos de test hasta el tamaño del horizonte, 
    # para tratarlos correctamente
    naive = x_test[: -horizon]
    #Debido a la forma del funcionamiento del naive, las evaluaciones del mismo tendrian h
    # valores menos que los datos, para corregir esto, se toma desde el primer elmento hasta 
    # los últimos h-1
    #Por qué estas posiciones?
    data = forecasts[: -horizon]
    #data = forecasts [1: -(horizon-1)]
    if horizon == 1:
        data = forecasts[1:]
    else: 
        data = forecasts [1: -(horizon-1)]
    #data = data [1: -(horizon-1)]
    #data = data[horizon :]
    #De todos los datos solo se toman los datos pronosticados, los últimos h
    #data = data[:, -horizon:]
    data = data[:, :horizon]

    #naive = naive[:, -horizon:]
    aux_naive = []
    for ele in naive:
        aux = [ele[-1] for _ in range(horizon)]
        aux_naive.append(np.array(aux))
    naive = np.array(aux_naive)

    #Formato de los datos, valores enteros
    data = data * 255
    data = data.astype(np.uint8)

    #Quitamos el canal de colores para categorizar correctamente
    data = data.reshape(data.shape[: -1])

    print("Datos procesados para evaluar")
    print(data.shape)
    print(naive.shape)
    print(y_test.shape)

    #Mascara
    print("Aplicando mascara")
    mascara = np.load("Models/mascara.npy")
    print(mascara)
    aux = np.array([])
    for i in data:
        aux2 = np.array([])
        for j in i:
            img_new = cv2.bitwise_and(j, j, mask= mascara)
            aux2 = np.append(aux2, img_new)
        aux = np.append(aux, aux2)
    data = aux.reshape(data.shape[:]).astype(np.uint8)
    print(data.shape)

    if display:
        plt.imshow(y_test[0,0], cmap="gray")
        plt.imshow(data[0,0], cmap="gray")
        plt.imshow(naive[0,0], cmap="gray")
        plt.show()

    #Debido al funcionamiento de las redes, el resultado no esta categorizado, 
    # por lo que se realiza este proceso en los pronósticos
    colors = get_colors(data[-10,0])
    print("Colores de pronóstico: {}".format(colors))

    print("CATEGORIZANDO PRONÓSTICOS...")
    data = multi_process_recolor(data, classes)
    colors = get_colors(data[-10,0])
    print("Colores de pronóstico categorizados: {}".format(colors))
    s = data.shape[:]
    data = data.reshape(s[0], s[1], s[2], s[3], 1)
    
    if display:
        plt.imshow(y_test[0,0], cmap="gray")
        plt.imshow(data[0,0], cmap="gray")
        plt.imshow(naive[0,0], cmap="gray")
        plt.show()
        
    
    y_test *= 255
    naive *= 255
    y_test = y_test.astype(np.uint8)
    naive = naive.astype(np.uint8)
    data = data.astype(np.uint8)

    np.save("Models/DiferencesOriginal"+ str(window) +".npy", y_test[10])
    np.save("Models/DiferencesNaive"+ str(window) +".npy", naive[10])
    np.save("Models/DiferencesForecast"+ str(window) + forecast_path[:-4] +".npy", data[10])
    

    l_class = len(classes)

    if display:
        fig = plt.figure(figsize=(10,7))
        r = 3
        c = horizon
        ac = 1
        h = horizon
        pos = 100
        for i in range(h):
            fig.add_subplot(r, c, ac)
            ac += 1
            plt.imshow(y_test[pos,i], cmap='gray')
            plt.axis('off')
            plt.title('Original_t+{}'.format(i))
            im = Image.fromarray(y_test[pos, i].reshape(rows, cols))
            im.save("GeneratedImageComparation/Original_t+{}.png".format(i))
        for i in range(h):
            fig.add_subplot(r, c, ac)
            ac += 1
            plt.imshow(data[pos,i], cmap='gray')
            plt.axis('off')
            plt.title('Pronóstico_t+{}'.format(i))
            im = Image.fromarray(data[pos, i].reshape(rows, cols))
            im.save("GeneratedImageComparation/Pronostico_t+{}.png".format(i))
        for i in range(h):
            fig.add_subplot(r, c, ac)
            ac += 1
            plt.imshow(naive[pos,i], cmap='gray')
            plt.axis('off')
            plt.title('Naive_t+{}'.format(i))
            im = Image.fromarray(naive[pos, i].reshape(rows, cols))
            im.save("GeneratedImageComparation/Naive_t+{}.png".format(i))
        plt.show()

    cm_f = np.zeros((l_class, l_class), dtype=np.uint64)
    cm_n = np.zeros((l_class, l_class), dtype=np.uint64)

    for i in range(horizon):
        print(i)
        actual_y = y_test[:, :(i+1)]
        actual_f = data[:, :(i+1)]
        actual_n = naive[:, :(i+1)]

        print("Resultados de los errores de pronóstico")
        MAE, MSE, RMSE = calculate_errors(actual_y.reshape(actual_y.shape[:-1])/255, actual_f.reshape(actual_f.shape[:-1])/255)
        print("MAE", MAE)
        print("MSE", MSE)
        print("RMSE", RMSE)
        res_err_pred = [MAE, MSE, RMSE]

        print("Resultados de los errores de Naive")
        MAE, MSE, RMSE = calculate_errors(actual_y.reshape(actual_y.shape[:-1])/255, actual_n.reshape(actual_n.shape[:-1])/255)
        print("MAE", MAE)
        print("MSE", MSE)
        print("RMSE", RMSE)

        res_err_naive = [MAE, MSE, RMSE]

        cm_f, cm_n = multi_process_evaluation(actual_y[:,-1], actual_f[:,-1], actual_n[:,-1], cm_f, cm_n, l_class, classes)
        print("Matriz de confusión de pronóstico")
        print(cm_f)

        cm_res_f, cm_conclusion_f = confusion_matrix_evaluation(cm_f)

        print("NAIVEEE!!!!!!!!!!!!!!")

        print("Matriz de confusión de naive")
        print(cm_n)

        cm_res_n, cm_conclusion_n = confusion_matrix_evaluation(cm_n)

        #Guardar datos
        df_err_pronostico = pd.DataFrame(res_err_pred, columns=['Forecast errs'])
        df_err_naive = pd.DataFrame(res_err_naive, columns=['Naive errs'])

        df_cm_f = pd.DataFrame(cm_f, columns=[f'cat_{i}' for i in range(cm_f.shape[1])])
        df_cm_res_f = pd.DataFrame(cm_res_f, columns=[f'cat_res_{i}' for i in range(cm_res_f.shape[1])])
        df_cm_conclusion_f = pd.DataFrame(cm_conclusion_f, columns=[f'cat__con_{i}' for i in range(cm_res_f.shape[1])])
        df_cm_n = pd.DataFrame(cm_n, columns=[f'cat_{i}' for i in range(cm_n.shape[1])])
        df_cm_res_n = pd.DataFrame(cm_res_n, columns=[f'cat_res_{i}' for i in range(cm_res_n.shape[1])])
        df_cm_conclusion_n = pd.DataFrame(cm_conclusion_n, columns=[f'cat_con_{i}' for i in range(cm_res_f.shape[1])])


        df_combinado = pd.concat([df_err_pronostico, df_err_naive, df_cm_f, df_cm_res_f, df_cm_conclusion_f, df_cm_n, df_cm_res_n, df_cm_conclusion_n], axis= 1)
        df_combinado.to_excel('Res_ConvLSTM/'+forecast_path[:-4]+'_h_'+str(i+1)+'.xlsx', index= False)



if __name__ == "__main__":
    #Multi-CNN
    #W=7
    #main('Model_MultiCNN_testing_1740654153.npy','Conv-LSTM_1.json', 12, True)
    #W=9
    #main('Model_MultiCNN_testing_1740655076.npy','Conv-LSTM_1.json', 12, True)
    #ViViTs ViVitLayers=4 Heads=4
    #W=6
    #main('DroughtDataset_model_testing_1740659737.npy','Conv-LSTM_1.json', 12, True)
    #w=7
    #main('DroughtDataset_model_testing_1740657903.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=2 Heads=4
    #main('DroughtDataset_model_testing_1740666525.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=2 Heads=2
    #main('DroughtDataset_model_testing_1740666407.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=1 Heads=2
    #main('DroughtDataset_model_testing_1740666277.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=1 Heads=4
    #main('DroughtDataset_model_testing_1740666131.npy','Conv-LSTM_1.json', 12, True)

    main('DroughtDataset_model_testing_1743006558.npy','Conv-LSTM_1.json', 12, True)

    #W=8
    #main('DroughtDataset_model_testing_1740657199.npy','Conv-LSTM_1.json', 12, True)
    #W=9
    #main('DroughtDataset_model_testing_1740659226.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=2 Heads=4
    #main('DroughtDataset_model_testing_1740664011.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=2 Heads=2
    #main('DroughtDataset_model_testing_1740664506.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=1 Heads=2
    #main('DroughtDataset_model_testing_1740664761.npy','Conv-LSTM_1.json', 12, True)
      #ViVitLayers=1 Heads=4
    #main('DroughtDataset_model_testing_1740664957.npy','Conv-LSTM_1.json', 12, True)

    #ViViTs 2 ViVitLayers=4 Heads=4
    #W=6
    #main('DroughtDataset_model_testing_1740660179.npy','Conv-LSTM_1.json', 12, True)
    #w=7
    #main('DroughtDataset_model_testing_1740660626.npy','Conv-LSTM_1.json', 12, True)
    #W=8
    #main('DroughtDataset_model_testing_1740661453.npy','Conv-LSTM_1.json', 12, True)
    #W=9
    #main('DroughtDataset_model_testing_1740662779.npy','Conv-LSTM_1.json', 12, True)
    

    #Nan = 1737572691
    #main('PredictionsTransformers.npy', 'Conv-LSTM_1.json', 4, True)
    # W = 2
    #main('DroughtDataset_model_testing_1737402833.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737486356.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1737576879.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737486356.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737581087.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1738926050.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1738941557.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739281653.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737402833.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739971663.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 3
    #main('DroughtDataset_model_testing_1737403771.npy', 'Conv-LSTM_1.json', 12, False)
    
    #Model_2 Normal
    #main('DroughtDataset_model_testing_1737575462.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_2 No batch
    #main('DroughtDataset_model_testing_1737579251.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1737491510.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737647586.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1738922834.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1738959571.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739273055.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737403771.npy', 'Conv-LSTM_1.json', 12, False) 
    #main('DroughtDataset_model_testing_1740159109.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 4
    #main('DroughtDataset_model_testing_1737404801.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737572691.npy', 'Conv-LSTM_1.json', 12, True)

    #main('DroughtDataset_model_testing_1737577664.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737495165.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_3 Normal
    #main('DroughtDataset_model_testing_1737651027.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_3 No batch
    #main('DroughtDataset_model_testing_1737653624.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_4 Normal
    #main('DroughtDataset_model_testing_1738880067.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_4 No batch
    #main('DroughtDataset_model_testing_1738918316.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_5 Normal
    #main('DroughtDataset_model_testing_1739003797.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_5 No batch
    #main('DroughtDataset_model_testing_1739012848.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_6 Normal
    #main('DroughtDataset_model_testing_1739214878.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_6 No batch
    #main('DroughtDataset_model_testing_1739221320.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737404801.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740218609.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 5
    #main('DroughtDataset_model_testing_1737402108.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1737571625.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737509868.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737660505.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1738686721.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739017484.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739202635.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737402108.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740387465.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 6
    #main('DroughtDataset_model_testing_1737484251.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1731542803.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731556679.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731702460.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731770579.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731781237.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1732204124.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1737484251.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740399564.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 7
    #main('DroughtDataset_model_testing_1737483278.npy', 'Conv-LSTM_1.json', 12, False)
    
    #main('DroughtDataset_model_testing_1731371092.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731374986.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731386833.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731423857.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731458809.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731467453.npy', 'Conv-LSTM_1.json', 12, False)
    
    #main('DroughtDataset_model_testing_1737483278.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740404989.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 8
    #main('DroughtDataset_model_testing_1737481537.npy', 'Conv-LSTM_1.json', 12, False)
    
    #main('DroughtDataset_model_testing_1737144261.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737390342.npy', 'Conv-LSTM_1.json', 12, False)
    
    #main('DroughtDataset_model_testing_1730753475.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1729969972.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1730245559.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1730674369.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1730578222.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1730477291.npy', 'Conv-LSTM_1.json', 12, False)
    
    #main('DroughtDataset_model_testing_1737481537.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740415036.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 9
    #main('DroughtDataset_model_testing_1737480194.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1730956825.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1730906187.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1730964452.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1730996203.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731001789.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731010169.npy', 'Conv-LSTM_1.json', 12, False)
    
    #main('DroughtDataset_model_testing_1737480194.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740423831.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 10
    #main('DroughtDataset_model_testing_1737478852.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1731173349.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731163779.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731127046.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731114524.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731094863.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1731078521.npy', 'Conv-LSTM_1.json', 12, False)
    
    #main('DroughtDataset_model_testing_1737478852.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740480296.npy', 'Conv-LSTM_1.json', 12, False)

    # W = 11
    #main('DroughtDataset_model_testing_1737476158.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1737570085.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_2
    #main('DroughtDataset_model_testing_1737517576.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_2
    #main('DroughtDataset_model_testing_1740394316.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_3 Normal
    #main('DroughtDataset_model_testing_1737695688.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_3 No batch
    #main('DroughtDataset_model_testing_1737743383.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1738092523.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739104601.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_6 Normal
    #main('DroughtDataset_model_testing_1739183381.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_6 No batch
    #main('DroughtDataset_model_testing_1739195585.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737476158.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740485474.npy', 'Conv-LSTM_1.json', 12, False)
    

    # W = 12
    #main('DroughtDataset_model_testing_1737477102.npy', 'Conv-LSTM_1.json', 12, False)

    #main('DroughtDataset_model_testing_1737566740.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737527554.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_3 normal
    #main('DroughtDataset_model_testing_1737752104.npy', 'Conv-LSTM_1.json', 12, False)
    #Model_3 No batch
    #main('DroughtDataset_model_testing_1737735406.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737780655.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739120074.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1739132424.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1737477102.npy', 'Conv-LSTM_1.json', 12, False)
    #main('DroughtDataset_model_testing_1740490926.npy', 'Conv-LSTM_1.json', 12, False)

