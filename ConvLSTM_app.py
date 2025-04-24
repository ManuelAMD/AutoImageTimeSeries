import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, Callback
from keras import backend as K
import keras
import tensorflow as tf
import gc
import json
import time
from mapPreprocessing import Preprocessing

class CustomCallback(Callback):
    def __init__(self, model, x_test):
        self.model = model
        self.x_test = x_test
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test[:1], batch_size= 2)
        plt.figure(figsize=(10,10))
        plt.imshow(y_pred[0], cmap='gray')
        plt.show()

def clean_memory():
    """ Release unused memory resources. Force garbage collection """
    K.clear_session()
    gc.collect()

def read_json_file(filename):
    f = open('configurations/{}'.format(filename), "r")
    parameters = json.load(f)
    print(type(parameters))
    return parameters

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
        predictions = model.predict(x_aux, batch_size= 2)
        total_preds.append(predictions)
        x_aux = add_last(x_aux, predictions[:])
    total_preds = np.array(total_preds)
    print(total_preds.shape)
    total_preds = np.transpose(total_preds, (1,0,2,3,4))
    print(total_preds.shape)
    return total_preds

def model1(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(32, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(16, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model2(inp, channels):
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    #m = keras.layers.Dropout(0.25)(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model3(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.ConvLSTM2D(channels, (3,3), padding= "same", activation= "relu")(m)
    return m

def model4(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model5(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(64, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model6(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(64, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(32, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model7(inp, channels):
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    #m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    #m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model0(inp, channels):
    m = keras.layers.ConvLSTM2D(1, (5,5), padding= "same", activation= "sigmoid")(inp)
    #m = keras.layers.BatchNormalization()(m)
    #m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    #m = keras.layers.BatchNormalization()(m)
    #m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    #m = keras.layers.Dropout(0.25)(m)
    #m = keras.layers.Conv2D(channels, (3,3), activation= "s9.igmoid", padding= "same")(m)
    return m

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

    preprocess = Preprocessing()
    preprocess.load_from_numpy_array(data_name, rows, cols, channels)
    x_train, y_train, x_validation, y_validation, x_test, y_test = preprocess.create_STI_dataset(window)

    strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
    with strategy.scope():
        if load_and_forecast:
            model = keras.models.load_model(model_name)
            err = model.evaluate(x_test, y_test, batch_size= 2)
            print("El error del modelo es: {}".format(err))

            forecast = map_forecast_recursive(model, x_test, horizon)
            forecast_name = "Models/{}".format(model_name)
            np.save(forecast_name+'.npy', forecast)
            print("Pronósticos almacenados en: {}".format(forecast_name))
            return

        inp = keras.layers.Input(shape= (None, *x_train.shape[2:]))
        m = model2(inp, channels)

        model = keras.models.Model(inp, m)
        model.compile(loss = 'mae', optimizer= optimizer)
        #model.compile(loss = 'binary_crossentropy', optimizer= optimizer)

        print(model.summary())
        #Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= early_stopping_value, restore_best_weights= True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 3)

        board = TensorBoard(log_dir='logs/{}'.format(name))

        epochs = config_json['epochs']
        batch_size = config_json['batch_size']

        model.fit(
            x_train, y_train,
            batch_size= batch_size,
            epochs = epochs,
            validation_data= (x_validation, y_validation),
            callbacks = [reduce_lr, early_stopping]
        )
        if display:
            example = x_test[np.random.choice(range(len(x_test)), size= 1)[0]]
            print(example.shape)

            for _ in range(horizon):
                print(example.shape)
                new_prediction = model.predict(example.reshape(1,*example.shape[0:]))
                example = np.concatenate((example[1:], new_prediction), axis=0)
                print(example.shape)
            predictions = example[:-4]
            print(predictions.shape)
    
        err = model.evaluate(x_test, y_test, batch_size= 2)
        print("El error del modelo es: {}".format(err))

        forecast = map_forecast_recursive(model, x_test, horizon)
        forecast_name = "Models/{}".format(name)
        model.save(forecast_name+'.keras')
        np.save(forecast_name+'.npy', forecast)
        print("Pronósticos almacenados en: {}".format(forecast_name))

if __name__ == '__main__':
    main('Conv-LSTM_1.json')