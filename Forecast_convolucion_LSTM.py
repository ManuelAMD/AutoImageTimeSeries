import pandas as pd
from pandas import DataFrame
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.callbacks import TensorBoard, Callback
import tensorflow as tf
from tensorflow import keras
from mpl_toolkits.mplot3d import axes3d
from keras import backend as K
import gc
import time
from app.common.color_tools import *
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class CustomCallback(Callback):
    def __init__(self, model, x_test):
        self.model = model
        self.x_test = x_test
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test[:1], batch_size= 2)
        plt.figure(figsize=(10,10))
        plt.imshow(y_pred[0], cmap='gray')
        plt.show()

def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y

def create_shifted_frames_2(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, data.shape[1]-1, :, :]
    return x, y

def agroup_window(data, window):
    new_data = [data[i:window+i] for i in range(len(data)-window+1)]
    return np.array(new_data)

def to_monochromatic(img_data, min_val= 10, max_val= 255):
    x_mono = []
    for i in img_data:
        (thresh, monoImg) = cv2.threshold(i, min_val, max_val, cv2.THRESH_BINARY)
        x_mono.append(monoImg)
    x_mono = np.array(x_mono)
    return x_mono

def add_last(data, new_vals):
    print(data.shape)
    x_test_new = data[:,1:]
    print(x_test_new.shape)
    print(new_vals.shape)
    l = []
    for i in range(len(x_test_new)):
        l.append(np.append(x_test_new[i], new_vals[i]))
    x_test_new = np.array(l).reshape(data.shape[:])
    print("CX", x_test_new.shape)
    return x_test_new

def recolor(args):
    data, pallete = args
    res = gray_quantized(data, pallete)
    res = recolor_greys_image(res, pallete)
    return np.array(res)


def limit_memory():
    """ Release unused memory resources. Force garbage collection """
    K.clear_session()
    gc.collect()

def main():
    window = 10
    channels = 1
    rows = 122
    cols = 360
    categorical = False
    categories = np.array([0, 35, 70, 119, 177, 220, 255]) #[0,51,102,153,204,255]
    horizon = 4
    name = 'Model_autoML_testing_{}'.format(int(time.time()))

    if categorical:
        x = np.load("Models/Data_full_select_color.npy") * 255
        x = x.astype(np.uint8)
        #Obtención de la paleta de colores, se toma una imagen muestra
        aux = x[1168]
        res = n_colors_img(aux, 6)
        colors = get_colors(res).reshape(6,1,3)
        print(len(colors))
        #Se utiliza una función de cuantificación en las imágenes para que
        # todas las imágenes manejen una paleta de colores.
        aux_data = np.array([rgb_quantized(i, colors) for i in x])
        print(aux_data.shape)
        #Se comprueban los colores obtenidos
        c1 = get_colors(aux_data[0])
        c2 = get_colors(aux_data[1167])
        print(c1)
        print(c2)
        #Se transforma el dataset de colores a escala de grieses, cv2 para mejor calidad.
        x_greys = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in aux_data])
        #np.save('Models/Data_full_select_greys.npy', x_greys)
        colors_greys = get_colors(x_greys[1168])
        cg1 = get_colors(x_greys[0])
        cg2 = get_colors(x_greys[1167])
        print(cg1)
        print(cg2)
        a1 = x_greys[0]
        a2 = x_greys[1167]
        x_greys = np.array([balance_img_categories(img, colors_greys, categories) for img in x_greys])
        ig1 = x_greys[0]
        ig2 = x_greys[1167]
        print(get_colors(ig1))
        print(get_colors(ig2))
        x = x_greys.astype('float32') / 255
    else:
        x = np.load("Models/DroughtDatasetMask.npy").astype(np.uint8)
        #x = np.load('Models/SPIDatasetMask.npy').astype(np.uint8) #/255

        #Mostrar imágenes
        fig, axes = plt.subplots(2, 3, figsize= (10,8))

        data_choise = np.random.choice(range(len(x)), size= 1)[0]
        for idx, ax in enumerate(axes.flat):
            ax.imshow(np.squeeze(x[data_choise+idx]), cmap='gray')
            ax.set_title(f"Frame {idx + 1}")
            ax.axis("off")

        #print("Displaying frames for example {}".format(data_choise))
        plt.show()
        
        args = [(d, categories) for d in x]

        num_cores = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cores-4) as pool:
            with tqdm(total = len(x)) as progress:
                futures = []

                for img in args:
                    future = pool.submit(recolor, img)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                
                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
        x_greys = np.array(results)
        #x = np.array([gray_quantized(i, np.array(categories)) for i in x])
        #colors_greys = get_colors(x[1168])
        #print(colors_greys)
        #x_greys = np.array([recolor_greys_image(img, categories) for img in x])
        x = x_greys.astype('float32') / 255
        #print(get_colors(x[1168]))
    #x = np.load("Models/Data_full_select_greys.npy")
    print(x.shape)

    #Mostrar imágenes
    fig, axes = plt.subplots(2, 3, figsize= (10,8))

    data_choise = np.random.choice(range(len(x)), size= 1)[0]
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(x[data_choise+idx]), cmap='gray')
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    #print("Displaying frames for example {}".format(data_choise))
    plt.show()

    x_2 = agroup_window(x, window)
    print(x_2.shape)
    x_train = x_2[:int(len(x_2)*.7)]
    x_test = x_2[int(len(x_2)*.7):]
    x_validation = x_train[int(len(x_train)*.8):]
    x_train = x_train[:int(len(x_train)*.8)]

    #x_trian = x_train.astype('float32') / 255
    #x_validation = x_validation.astype('float32') / 255
    #x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape(len(x_train), window, rows, cols, channels)
    x_validation = x_validation.reshape(len(x_validation), window, rows, cols, channels)
    x_test = x_test.reshape(len(x_test), window, rows, cols, channels)

    print("Forma de datos de entrenamiento: {}".format(x_train.shape))
    print("Forma de datos de validación: {}".format(x_validation.shape))
    print("Forma de datos de pruebas: {}".format(x_test.shape))

    x_train, y_train = create_shifted_frames_2(x_train)
    x_validation, y_validation = create_shifted_frames_2(x_validation)
    x_test, y_test = create_shifted_frames_2(x_test)

    print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
    print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
    print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

    np.save("Models/x_test_convlstm_greys_forecast.npy", x_test)
    np.save("Models/y_test_convlstm_greys_forecast.npy", y_test)

    #Mostrar imágenes
    #fig, axes = plt.subplots(2, 3, figsize= (10,8))

    #data_choise = np.random.choice(range(len(x_2)), size= 1)[0]
    #for idx, ax in enumerate(axes.flat):
    #    ax.imshow(np.squeeze(x_2[data_choise][idx]), cmap='gray')
    #    ax.set_title(f"Frame {idx + 1}")
    #    ax.axis("off")

    #print("Displaying frames for example {}".format(data_choise))
    #plt.show()

    #strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
    with strategy.scope():
        #Construction of Convolutional LSTM network
        print(*x_train.shape[2:])
        print(*x_train.shape[1:])
        inp = keras.layers.Input(shape=(None, *x_train.shape[2:]))

        #It will be constructed a 3 ConvLSTM2D layers with batch normalization,
        #Followed by a Conv3D layer for the spatiotemporal outputs.

        #m = keras.layers.ConvLSTM2D(8, (5,5), padding= "same", activation= "relu")(inp)
        #m = keras.layers.BatchNormalization()(m)
        #m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
        #m = keras.layers.BatchNormalization()(m)
        #m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
        #m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)

        m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
        m = keras.layers.BatchNormalization()(m)
        #m = keras.layers.ConvLSTM2D(12, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
        m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
        #m = keras.layers.BatchNormalization()(m)
        #m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
        #m = keras.layers.BatchNormalization()(m)
        #m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
        m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)

        model = keras.models.Model(inp, m)
        model.compile(loss= "binary_crossentropy", optimizer= "Adam")

        print(model.summary())

        #Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor= "val_loss", patience= 6, restore_best_weights= True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 4)

        board = TensorBoard(log_dir='logs/{}'.format(name))

        #Define moifiable training hyperparameters
        epochs = 150
        batch_size = 4

        #Model training
        model.fit(
            x_train, y_train,
            batch_size= batch_size,
            epochs= epochs,
            validation_data= (x_validation, y_validation),
            #callbacks= [early_stopping, reduce_lr, board, CustomCallback(model, x_test)]
            callbacks= [reduce_lr, early_stopping]
        )

        example = x_test[np.random.choice(range(len(x_test)), size= 1)[0]]

        #frames = example[:4, ...]
        #original_frames = example[4:, ...]
        print(example.shape)
        #print(frames.shape)
        #print(original_frames.shape)

        for _ in range(horizon):
            print(example.shape)
            new_prediction = model.predict(example.reshape(1,*example.shape[0:]))
            example = np.concatenate((example[1:], new_prediction), axis=0)
            print(example.shape)

        predictions = example[:-4]
        print(predictions.shape)
        #fig, axes = plt.subplots(2,4, figsize= (20,4))
        #for idx, ax in enumerate(axes[0]):
        #    ax.imshow((predictions[idx]), cmap='gray')
        #    ax.set_title("Frame {}".format(idx+3))
        #    ax.axis("off")
        #plt.show()
        err = model.evaluate(x_test, y_test, batch_size= 2)
        print("El error del modelo es: {}".format(err))
        preds = model.predict(x_test, batch_size= 2)
        print(preds.shape)
        x_test_new = add_last(x_test, preds[:])
        preds2 = model.predict(x_test_new, batch_size= 2)
        #print(preds2.shape)
        x_test_new = add_last(x_test_new, preds2[:])
        preds3 = model.predict(x_test_new, batch_size= 2)
        x_test_new = add_last(x_test_new, preds3[:])
        preds4 = model.predict(x_test_new, batch_size= 2)
        res_forecast = add_last(x_test_new, preds4[:])
        print("PREDSS",res_forecast.shape)
        if categorical:
            np.save("Models/PredictionsConvolutionLSTM_greys_forecast_1.npy", res_forecast)
        else:
            np.save("Models/PredictionsConvolutionLSTM_forecast_1.npy", res_forecast)
        raise
        try:
            for _ in range(horizon):
                new_prediction = model.predict(example)
                example = np.concatenate((example[0:], new_prediction), axis=0)
                print(example.shape)

                #new_prediction = model.predict(np.expand_dims(frames, axis= 0),batch_size= 2)
                #new_prediction = np.squeeze(new_prediction, axis= 0)
                #predicted_frame = np.expand_dims(new_prediction[-1, ...], axis= 0)

                #frames = np.concatenate((frames, predicted_frame), axis= 0)
            raise
            fig, axes = plt.subplots(2,4, figsize= (20,4))
            for idx, ax in enumerate(axes[0]):
                ax.imshow(np.squeeze(original_frames[idx]), cmap='gray')
                ax.set_title("Frame {}".format(idx+3))
                ax.axis("off")

            new_frames = frames[4:, ...]
            for idx, ax in enumerate(axes[1]):
                ax.imshow(np.squeeze(new_frames[idx]), cmap='gray')
                ax.set_title("F Frame {}".format(idx+3))
                ax.axis("off")

            plt.show()
        except:
            print("Cannot display forecast, continuing for evaluation")

        err = model.evaluate(x_test, y_test)
        print("El error del modelo es: {}".format(err))

        preds = model.predict(x_test)
        print(preds.shape)

        #aux = preds[:,-1]
        #aux = aux.reshape(aux.shape[0], 1, aux.shape[1], aux.shape[2], aux.shape[3])
        #print(aux.shape)
        #x_test_new = x_test[:,1:]
        #print(x_test_new.shape)
        #l = []
        #for i in range(len(x_test_new)):
        #    l.append(np.append(x_test_new[i], aux[i]))
        #x_test_new = np.concatenate((x_test_new, aux), axis=1)
        #x_test_new = np.array(l).reshape(x_test.shape[:])
        #print(x_test_new.shape)
        x_test_new = add_last(x_test, preds[:,-1])
        preds2 = model.predict(x_test_new)
        #print(preds2.shape)
        x_test_new = add_last(x_test_new, preds2[:,-1])
        preds3 = model.predict(x_test_new)
        x_test_new = add_last(x_test_new, preds3[:,-1])
        preds4 = model.predict(x_test_new)
        print(preds4.shape)
        
        try:
            pos = 100
            fig, axes = plt.subplots(5, window-1, figsize= (20,30))
            for idx, ax in enumerate(axes[0]):
                if idx == window-1:
                    break
                ax.imshow(np.squeeze(x_test_new[pos][idx]), cmap= 'gray')
                ax.set_title(f"Original t_{pos+idx}")
                #ax.set_title(f"Original")
                ax.axis("off")

            for idx, ax in enumerate(axes[1]):
                if idx == window-1:
                    break
                ax.imshow(np.squeeze(preds[pos][idx]), cmap= 'gray')
                ax.set_title(f"preds t_{pos+idx}")
                #ax.set_title(f"Original")
                ax.axis("off")
            
            for idx, ax in enumerate(axes[2]):
                if idx == window-1:
                    break
                ax.imshow(np.squeeze(preds2[pos][idx]), cmap= 'gray')
                ax.set_title(f"Preds2 t_{pos+idx}")
                #ax.set_title(f"Original")
                ax.axis("off")
            
            for idx, ax in enumerate(axes[3]):
                if idx == window-1:
                    break
                ax.imshow(np.squeeze(preds3[pos][idx]), cmap= 'gray')
                ax.set_title(f"preds3 t_{pos+idx}")
                #ax.set_title(f"Original")
                ax.axis("off")
            
            for idx, ax in enumerate(axes[4]):
                if idx == window-1:
                    break
                ax.imshow(np.squeeze(preds4[pos][idx]), cmap= 'gray')
                ax.set_title(f"Preds4 t_{pos+idx}")
                #ax.set_title(f"Original")
                ax.axis("off")
        except:
            print("Cannot display")

        plt.show()

        print(preds4.shape)

        np.save("Models/PredictionsConvolutionLSTM_greys_forecast.npy", preds4)

if __name__ == "__main__":
    main()