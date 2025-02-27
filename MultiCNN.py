from app.common.color_tools import *
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard, Callback
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from mapPreprocessing import Preprocessing
import json

def recolor(args):
    data, pallete = args
    res = gray_quantized(data, pallete)
    res = recolor_greys_image(res, pallete)
    return np.array(res)

def agroup_window(data, window):
    new_data = [data[i:window+i] for i in range(len(data)-window+1)]
    return np.array(new_data)

def read_json_file(filename):
    f = open('configurations/{}'.format(filename), "r")
    parameters = json.load(f)
    print(type(parameters))
    return parameters

def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, data.shape[1] - 1, :, :]
    return x, y

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

def map_forecast_recursive(model: keras.Model, x_test: np.array, horizonte: int):
    x_aux = x_test
    total_preds = []
    for i in range(horizonte):
        predictions = model.predict((x_aux[:,0],x_aux[:,1], x_aux[:,2], x_aux[:,3], x_aux[:,4], x_aux[:,5], x_aux[:,6], x_aux[:,7], x_aux[:,8]), batch_size= 2)
        #predictions = model.predict((x_aux[:,0],x_aux[:,1], x_aux[:,2], x_aux[:,3], x_aux[:,4], x_aux[:,5], x_aux[:,6]), batch_size= 2)
        total_preds.append(predictions)
        x_aux = add_last(x_aux, predictions[:])
    total_preds = np.array(total_preds)
    print(total_preds.shape)
    total_preds = np.transpose(total_preds, (1,0,2,3,4))
    print(total_preds.shape)
    return total_preds

def main(config_file, load_and_forecast=False, model_name='', display= False):
    config_json = read_json_file(config_file)
    window = config_json['window_size']
    rows = config_json['rows']
    cols = config_json['cols']
    channels = config_json['channels']
    horizon = config_json['horizon']
    name = config_json['name'] + '_model_testing_{}'.format(int(time.time()))
    optimizer = config_json['optimizer']
    data_name = '{}/{}.npy'.format(config_json['folder_models_save'], config_json['folder'])
    early_stopping_value = config_json['deep_training_early_stopping_patience']
    #epochs = 150
    #batch_size = 4
    #window = 9
    #channels = 1
    #rows = 260
    #cols = 640
    categories = np.array([0,35,70,119,177,220,255])
    #horizon = 12
    name = 'Model_MultiCNN_testing_{}'.format(int(time.time()))

    preprocess = Preprocessing()
    preprocess.load_from_numpy_array(data_name, rows, cols, channels)
    x_train, y_train, x_validation, y_validation, x_test, y_test = preprocess.create_STI_dataset(window)

    """x = np.load('Models/SPIDatasetMask.npy').astype(np.uint8)

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
    x = x_greys.astype('float32') / 255
    print(x.shape)

    x = agroup_window(x, window+1)

    x_train = x[:int(len(x)*.7)]
    x_test = x[int(len(x)*.7):]
    x_validation = x_train[int(len(x_train)*.8):]
    x_train = x_train[:int(len(x_train)*.8)]

    x_train = x_train.reshape(*x_train.shape[:], channels)
    x_test = x_test.reshape(*x_test.shape[:], channels)
    x_validation = x_validation.reshape(*x_validation.shape[:], channels)
    
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)

    x_train, y_train = create_shifted_frames(x_train)
    x_validation, y_validation = create_shifted_frames(x_validation)
    x_test, y_test = create_shifted_frames(x_test)

    print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
    print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
    print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

    np.save("Models/x_test_multicnn_greys.npy", x_test)
    np.save("Models/y_test_multicnn_greys.npy", y_test)"""

    strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
    with strategy.scope():
        print(x_train.shape)
        print(x_train[:,0].shape, x_train[:,1].shape)
        d_shape = x_train.shape[2:]
        inp = keras.layers.Input(shape=(x_train.shape[2:]))
        inp2 = keras.layers.Input(shape=(x_train.shape[2:]))
        inp3 = keras.layers.Input(shape=(x_train.shape[2:]))
        inp4 = keras.layers.Input(shape=(x_train.shape[2:]))
        inp5 = keras.layers.Input(shape=(x_train.shape[2:]))
        inp6 = keras.layers.Input(shape=(x_train.shape[2:]))
        inp7 = keras.layers.Input(shape=(x_train.shape[2:]))
        inp8 = keras.layers.Input(shape=(x_train.shape[2:]))
        inp9 = keras.layers.Input(shape=(x_train.shape[2:]))

        m1 = keras.layers.Conv2D(64, (7,7), padding= "same", activation= "relu")(inp)
        #m1 = keras.layers.Conv2D(32, (5,5), padding="same", activation= "relu")(m1)

        m2 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp2)
        #m2 = keras.layers.Conv2D(32, (5,5), padding="same", activation= "relu")(m2)

        m3 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp3)
        #m3 = keras.layers.Conv2D(32, (5,5), padding="same", activation= "relu")(m3)

        m4 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp4)
        #m4 = keras.layers.Conv2D(32, (5,5), padding="same", activation= "relu")(m4)

        m5 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp5)

        m6 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp6)
        
        m7 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp7)
        
        m8 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp8)

        m9 = keras.layers.Conv2D(64, (7,7), padding="same", activation= "relu")(inp9)

        m = keras.layers.concatenate([m1, m2, m3, m4, m5, m6, m7, m8, m9])
        #m = keras.layers.concatenate([m1, m2, m3, m4, m5, m6, m7])
        #m = keras.layers.ConvLSTM2D(8, (7,7), padding= "same", activation= "relu")(m)
        m = keras.layers.Conv2D(32, (3,3), padding="same", activation= "relu")(m)
        m = keras.layers.Conv2D(16, (3,3), padding="same", activation= "relu")(m)
        m = keras.layers.Conv2D(16, (3,3), padding="same", activation= "relu")(m)
        m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)

        model = keras.models.Model([inp, inp2, inp3, inp4, inp5, inp6, inp7, inp8, inp9], m)
        #model = keras.models.Model([inp, inp2, inp3, inp4, inp5, inp6, inp7], m)
        model.compile(loss= 'mae', optimizer= 'Adam')

        print(model.summary())

        early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= early_stopping_value, restore_best_weights= True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= 'val_loss', patience= 4)

        board = TensorBoard(log_dir= 'logs/{}'.format(name))

        t = (x_train[:,0], x_train[:,1], x_train[:,2], x_train[:,3], x_train[:,4], x_train[:,5], x_train[:,6], x_train[:,7], x_train[:,8])
        v = (x_validation[:,0], x_validation[:,1], x_validation[:,2], x_validation[:,3], x_validation[:,4], x_validation[:,5], x_validation[:,6], x_validation[:,7], x_validation[:,8])
        #t = (x_train[:,0], x_train[:,1], x_train[:,2], x_train[:,3], x_train[:,4], x_train[:,5], x_train[:,6])
        #v = (x_validation[:,0], x_validation[:,1], x_validation[:,2], x_validation[:,3], x_validation[:,4], x_validation[:,5], x_validation[:,6])

        model.fit(
            t, 
            y_train,
            batch_size = 4,
            epochs = 150,
            validation_data= (v, y_validation),
            callbacks= [early_stopping, reduce_lr]
        )

        example = x_test[np.random.choice(range(len(x_test)), size= 1)[0]]
        print(example.shape)
        
        for _ in range(horizon):
            example = example.reshape(1, *example.shape[:])
            print(example.shape)
            new_prediction = model.predict((example[:,0], example[:,1], example[:,2], example[:,3], example[:,4], example[:,5], example[:,6], example[:,7], example[:,8]))
            #new_prediction = model.predict((example[:,0], example[:,1], example[:,2], example[:,3], example[:,4], example[:,5], example[:,6]))
            example = example.reshape(example.shape[1:])
            example = np.concatenate((example[1:], new_prediction), axis=0)
            print(example.shape)

        #predictions = example[:-window]
        #print(example.shape)
        #fig, axes = plt.subplots(2,window, figsize= (20,4))
        #for idx, ax in enumerate(axes[0]):
        #    ax.imshow((example[idx]), cmap='gray')
        #    ax.set_title("Frame {}".format(idx+3))
        #    ax.axis("off")
        #plt.show()

        #prediction = model.predict((example[:,0], example[:,1]))
        #print(prediction.shape)

        #img = plt.imshow(prediction[0])
        #plt.show()

        print(x_test.shape)

        err = model.evaluate((x_test[:,0],x_test[:,1], x_test[:,2], x_test[:,3], x_test[:,4], x_test[:,5], x_test[:,6], x_test[:,7], x_test[:,8]), y_test, batch_size= 2)
        #err = model.evaluate((x_test[:,0],x_test[:,1], x_test[:,2], x_test[:,3], x_test[:,4], x_test[:,5], x_test[:,6]), y_test, batch_size= 2)
        print("El error del modelo es: {}".format(err))
        """preds = model.predict((x_test[:,0],x_test[:,1], x_test[:,2], x_test[:,3], x_test[:,4], x_test[:,5], x_test[:,6], x_test[:,7], x_test[:,8]), batch_size= 2)
        print(preds.shape)
        x_test_new = add_last(x_test, preds[:])
        preds2 = model.predict((x_test_new[:,0],x_test_new[:,1], x_test_new[:,2], x_test_new[:,3], x_test_new[:,4], x_test_new[:,5], x_test_new[:,6], x_test_new[:,7], x_test_new[:,8]), batch_size= 2)
        #print(preds2.shape)
        x_test_new = add_last(x_test_new, preds2[:])
        preds3 = model.predict((x_test_new[:,0],x_test_new[:,1], x_test_new[:,2], x_test_new[:,3], x_test_new[:,4], x_test_new[:,5], x_test_new[:,6], x_test_new[:,7], x_test_new[:,8]), batch_size= 2)
        x_test_new = add_last(x_test_new, preds3[:])
        preds4 = model.predict((x_test_new[:,0],x_test_new[:,1], x_test_new[:,2], x_test_new[:,3], x_test_new[:,4], x_test_new[:,5], x_test_new[:,6], x_test_new[:,7], x_test_new[:,8]), batch_size= 2)
        res_forecast = add_last(x_test_new, preds4[:])
        print("PREDSS",res_forecast.shape)

        #predictions = model.predict((x_test[:,0], x_test[:,1]))
        np.save('Models/PredictionMultiCNN_forecast.npy', res_forecast)"""
        forecast = map_forecast_recursive(model, x_test, horizon)
        forecast_name = "Models/{}".format(name)
        model.save(forecast_name+'.keras')
        np.save(forecast_name+'.npy', forecast)
        print("Pronósticos almacenados en: {}".format(forecast_name))

if __name__ == '__main__':
    main('Conv-LSTM_1.json')