import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, Callback
from keras import backend as K
import keras
import tensorflow as tf
import gc
import json
import time
import math
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from mapPreprocessing import Preprocessing
from app.common.color_tools import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("{} GPUs detectadas y configuradas".format(len(gpus)))

class CustomCallback(Callback):
    def __init__(self, model, x_test):
        self.model = model
        self.x_test = x_test
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test[:1], batch_size= 2)
        plt.figure(figsize=(10,10))
        plt.imshow(y_pred[0], cmap='gray')
        plt.show()
    
    def get_memory_usage():
        info = tf.config.experimental.get_memory_info()

class MemoryMonitor(Callback):
    def __init__(self):
        self.static_memory = {}
        self.peak_memory = {}
        self.dynamic_memory = {}

    def _get_memory_info(self):
        usage = {}
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            device_name = gpu.name.split(':')[-2] + ':' + gpu.name.split(':')[-1]
            try:
                memory_info = tf.config.experimental.get_memory_info(device_name)
                usage[device_name] = {
                    'current': memory_info['current'] / (1024 ** 3),
                    'peak': memory_info['peak'] / (1024 ** 3)
                }
            except ValueError:
                pass
        return usage
    
    def on_train_begin(self, logs= None):
        current_usage = self._get_memory_info()
        print("-- Línea Base de memoria (memoria estática) --")
        for gpu, info in current_usage.items():
            self.static_memory[gpu] = info['current']
            print(f"{gpu}: {self.static_memory[gpu]:.4f} GB (ocupado por pesos + framework)")
        
        #Reset el contador pico para medir lo que ocurre en el entrenamiento
        for gpu in tf.config.list_physical_devices('CPU'):
            device_name = gpu.name.split(':')[-2] + ':' + gpu.name.split(':')[-1]
            try:
                tf.config.experimental.reset_memory_stats(device_name)
            except:
                pass
    
    def on_train_batch_end(self, batch, logs=None):
        #Se monitorea cada cierto tiempo
        if batch % 3 == 0:
            current_usage = self._get_memory_info()

            for gpu, info in current_usage.items():
                self.peak_memory[gpu] = info['peak']
                #Calculo de memoria dinámica (pico máximo - estática inicial)
                static = self.static_memory.get(gpu, 0)
                dynamic = info['peak'] - static
                self.dynamic_memory[gpu] = dynamic
    
    def on_train_end(self, logs=None):
        print("-- USO completo de memoria --")
        for gpu in self.static_memory.keys():
            static = self.static_memory[gpu]
            peak = self.peak_memory.get(gpu, 0)
            dynamic = self.dynamic_memory.get(gpu, 0)
            
            print(f"Dispositivo: {gpu}")
            print(f"1.- Memoria estática (Modelo):          {static:.4f} GB")
            print(f"2.- Memoria dinámica (Activaciones):    {dynamic:4f} GB")
            print(f"3.- Pico total alcanzado:               {peak:.4f}   GB")
            print("-"*30)

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

def recolor(args):
    data, pallete = args
    aux = []
    for i in data:
        res = gray_quantized(i, pallete)
        res = recolor_greys_image(res, pallete)
        aux.append(res)
    return np.array(aux)

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

def recolorize_predictions(data, classes):
    #Mascara
    prediction_aux = data * 255
    prediction_aux = prediction_aux.astype(np.uint8)
    prediction_aux = prediction_aux.reshape(prediction_aux.shape[: -1])
    mascara = np.load("Models/mascara.npy")
    aux = np.array([])
    for i in prediction_aux:
        img_new = cv2.bitwise_and(i, i, mask= mascara)
        aux = np.append(aux, img_new)
    predictions = aux.reshape(prediction_aux.shape[:]).astype(np.uint8)
    s = predictions.shape[:]
    predictions = predictions.reshape(s[0], 1, s[1], s[2])

    #Recolorización
    predictions = multi_process_recolor(predictions, classes)
    s = predictions.shape[:]
    predictions = predictions.reshape(s[0], s[2], s[3], 1)
    predictions = predictions.astype(np.float32)
    predictions /= 255
    return predictions


def map_forecast_recursive(model: keras.Model, x_test: np.array, horizonte: int, classes):
    x_aux = x_test
    total_preds = []
    for i in range(horizonte):
        predictions = model.predict(x_aux, batch_size= 2)

        predictions = recolorize_predictions(predictions, classes)

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

def model_multi_step(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model_tesis_1(inp, channels):
    m = keras.layers.ConvLSTM2D(channels, (3,3), padding= "same", activation= "sigmoid", dtype="float32")(inp)
    return m

def model_tesis_2(inp, channels):
    m = keras.layers.ConvLSTM2D(channels, (5,5), padding= "same", activation= "sigmoid", dtype="float32")(inp)
    return m

def model_tesis_3(inp, channels):
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.Conv2D(channels, (3,3), padding= "same", activation= "sigmoid", dtype="float32")(m)
    return m

def model_tesis_4(inp, channels):
    m = keras.layers.ConvLSTM2D(32, (3,3), padding= "same", activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.Conv2D(channels, (3,3), padding= "same", activation= "sigmoid", dtype="float32")(m)
    return m

def model_tesis_5(inp, channels):
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.Conv2D(channels, (3,3), padding= "same", activation= "sigmoid", dtype="float32")(m)
    return m

class CheckpointedConvLSTM2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__()
        self.inner = keras.layers.ConvLSTM2D(filters, kernel_size, **kwargs)

    def call(self, inputs, training=None):
        def forward(x):
            return self.inner(x, training=training)
        return tf.recompute_grad(forward)(inputs)

def model_tesis_6(inp, channels):
    #m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = CheckpointedConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    #m = keras.layers.ConvLSTM2D(32, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = CheckpointedConvLSTM2D(32, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(channels, (3,3), padding= "same", activation= "sigmoid", dtype="float32")(m)
    return m

def model_tesis_7(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    #m = keras.layers.ConvLSTM2D(channels, (3,3), padding= "same", activation= "sigmoid", dtype="float32")(m)
    m = keras.layers.ConvLSTM2D(channels, (3,3), padding= "same", activation= "sigmoid")(m)
    return m

def model_tesis_8(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    #m = CheckpointedConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    #m = CheckpointedConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(32, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(16, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m


def recursive_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value):
    #keras.mixed_precision.set_global_policy("mixed_float16")
    inp = keras.layers.Input(shape= (None, *x_train.shape[2:]))
    m = model_tesis_7(inp, channels)
    model = keras.models.Model(inp, m)
    #opt = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-4, clipnorm= 1.0)
    #opt = keras.mixed_precision.LossScaleOptimizer(opt)
    model.compile(loss = 'mae', optimizer= optimizer)
    #model.compile(loss = 'binary_crossentropy', optimizer= optimizer)
    print(model.summary())
    #Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= early_stopping_value, restore_best_weights= True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 3)
    memory_monitor = MemoryMonitor()
    
    board = TensorBoard(log_dir='logs/{}'.format(name))
    epochs = config_json['epochs']
    batch_size = config_json['batch_size']
    print("Tipo de datos para entrenamiento", x_train.dtype)
    model.fit(
        x_train, y_train,
        batch_size= batch_size,
        epochs = epochs,
        validation_data= (x_validation, y_validation),
        callbacks = [reduce_lr, early_stopping, memory_monitor]
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
    err = model.evaluate(x_test, y_test, batch_size= batch_size)
    print("El error del modelo es: {}".format(err))
    classes = np.array(config_json['classes'])
    forecast = map_forecast_recursive(model, x_test, horizon, classes)

    forecast_name = "Models/{}".format(name)
    model.save(forecast_name+'.keras')
    np.save(forecast_name+'.npy', forecast)
    print("Pronósticos almacenados en: {}".format(forecast_name))

def model_multi_step_1(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model_multi_step_2(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(8, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(16, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model_multi_step_3(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(8, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(4, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(32, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(16, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model_multi_step_4(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(8, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(4, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(2, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(64, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(32, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(16, (3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m


def direct_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value, continue_at= 0):
    total_preds = []
    forecast_name = "Models/{}".format(name)
    board = TensorBoard(log_dir='logs/{}'.format(name))
    epochs = config_json['epochs']
    batch_size = config_json['batch_size']
    if continue_at != 0:
        for i in range(continue_at):
            model = keras.saving.load_model(forecast_name+'_horizon_{}.keras'.format(i))
            forecast = model.predict(x_test, batch_size= batch_size)
            total_preds.append(forecast)

    for h in range(continue_at, horizon):
        print("** EVALUANDO MODELO PARA EL HORIZONTE {} **".format(h+1))
        y_train_actual = y_train[:,h]
        y_validation_actual = y_validation[:,h]
        y_test_actual = y_test[:,h]
        inp = keras.layers.Input(shape= (None, *x_train.shape[2:]))
        
        #if h%4 == 0:
        #    m = model_multi_step_1(inp, channels)
        #elif h%4 == 1:
        #    m = model_multi_step_2(inp, channels)
        #elif h%4 == 2:
        #    m = model_multi_step_3(inp, channels)
        #else:
        #    m = model_multi_step_4(inp, channels)
        m = model_tesis_8(inp, channels)
        
        model = keras.models.Model(inp, m)
        model.compile(loss = 'mae', optimizer= optimizer)
        model.summary()

        #Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= early_stopping_value, restore_best_weights= True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 3)
        
        model.fit(
            x_train, y_train_actual,
            batch_size= batch_size,
            epochs = epochs,
            validation_data= (x_validation, y_validation_actual),
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
        
        err = model.evaluate(x_test, y_test_actual, batch_size= batch_size)
        print("El error del modelo es: {}".format(err))
        #forecast = map_forecast_recursive(model, x_test, horizon)
        forecast = model.predict(x_test, batch_size= batch_size)
        total_preds.append(forecast)
        new_name = forecast_name+'_horizon_{}'.format(h)
        model.save(new_name+'.keras')
        print("Modelo directo almacenado en: {}".format(new_name))

    total_preds = np.array(total_preds)
    print(total_preds.shape)
    total_preds = np.transpose(total_preds, (1,0,2,3,4))
    print(total_preds.shape)
    np.save(forecast_name+'.npy', total_preds)
    print("Pronósticos almacenados en: {}".format(forecast_name))

def model_1_MIMO(inp, Total_output):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.Conv3D(1, (3,3,3), activation= "sigmoid", padding= "same")(m)
    return m

def model_2_MIMO(inp, Total_output):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.Conv3D(1, (3,3,3), activation= "sigmoid", padding= "same")(m)
    return m

def testing_MIMO(inp, Total_output):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", return_sequences= False, activation= "relu")(m)
    def repeat_output(tensor):
        tensor = tf.expand_dims(tensor, axis= 1)
        return tf.repeat(tensor, repeats= Total_output, axis= 1)
    m = keras.layers.Lambda(repeat_output)(m)
    m = keras.layers.Conv3D(1, (3,3,3), activation= "sigmoid", padding= "same")(m)
    return m

def testing_MIMO2(inp, Total_output):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (3,3), padding= "same", return_sequences= False, activation= "relu")(m)
    def repeat_output(tensor):
        tensor = tf.expand_dims(tensor, axis= 1)
        return tf.repeat(tensor, repeats= Total_output, axis= 1)
    m = keras.layers.Lambda(repeat_output)(m)
    m = keras.layers.Conv3D(1, (3,3,3), activation= "sigmoid", padding= "same")(m)
    return m

def testing_MIMO3(inp, Total_output):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (3,3), padding= "same", return_sequences= False, activation= "relu")(m)
    def repeat_output(tensor):
        tensor = tf.expand_dims(tensor, axis= 1)
        return tf.repeat(tensor, repeats= Total_output, axis= 1)
    m = keras.layers.Lambda(repeat_output)(m)
    m = keras.layers.Conv3D(16, (3,3,3), activation= "sigmoid", padding= "same")(m)
    m = keras.layers.Conv3D(1, (3,3,3), activation= "sigmoid", padding= "same")(m)
    return m

def testing_MIMO5(inp, Total_output):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (3,3), padding= "same", return_sequences= False, activation= "relu")(m)
    def repeat_output(tensor):
        tensor = tf.expand_dims(tensor, axis= 1)
        return tf.repeat(tensor, repeats= Total_output, axis= 1)
    m = keras.layers.Lambda(repeat_output)(m)
    m = keras.layers.Conv3D(32, (5,5,5), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv3D(16, (3,3,3), activation= "relu", padding= "same")(m)
    m = keras.layers.Conv3D(1, (3,3,3), activation= "sigmoid", padding= "same")(m)
    return m

def testing_MIMO4(inp, Total_output):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(32, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", return_sequences= False, activation= "relu")(m)
    def repeat_output(tensor):
        tensor = tf.expand_dims(tensor, axis= 1)
        return tf.repeat(tensor, repeats= Total_output, axis= 1)
    m = keras.layers.Lambda(repeat_output)(m)
    m = keras.layers.Conv3D(16, (3,3,3), activation= "sigmoid", padding= "same")(m)
    m = keras.layers.Conv3D(1, (3,3,3), activation= "sigmoid", padding= "same")(m)
    return m

def MIMO_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value):
    inp = keras.layers.Input(shape= (x_train.shape[1:]))
    print(inp)
    #output_shape = keras.layers.Input(shape= (y_train.shape[1], *x_train.shape[2:]))
    #inp_seq_length = keras.layers.Input(x_train.shape[1:])
    #print(inp_seq_length)
    #inp = keras.layers.Input(shape= (None, x_train.shape[4], x_train.shape[2], x_train.shape[3]))
    m = testing_MIMO5(inp, horizon)
    model = keras.models.Model(inp, m)
    model.compile(loss = 'mae', optimizer= optimizer)
    #model.compile(loss = 'binary_crossentropy', optimizer= optimizer)
    print(model.summary())
    #Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= early_stopping_value, restore_best_weights= True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 3)
    memory_monitor = MemoryMonitor()

    board = TensorBoard(log_dir='logs/{}'.format(name))
    epochs = config_json['epochs']
    #epochs = 10
    batch_size = config_json['batch_size']
    print(model.output_shape)
    model.fit(
        x_train, y_train,
        batch_size= batch_size,
        epochs = epochs,
        validation_data= (x_validation, y_validation),
        callbacks = [reduce_lr, early_stopping, memory_monitor]
    )
    #K.clear_session()

    err = model.evaluate(x_test, y_test, batch_size= batch_size)
    print("El error del modelo es: {}".format(err))
    forecast = model.predict(x_test)
    forecast_name = "Models/{}".format(name)
    model.save(forecast_name+'.keras')
    np.save(forecast_name+'.npy', forecast)
    print("Pronósticos almacenados en: {}".format(forecast_name))

def DirRec_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value, continue_at= 0):
    total_preds = []
    forecast_name = "Models/{}".format(name)
    classes = np.array(config_json['classes'])
    #Taking all data and store in aux variables for when the recursive part sustitute the last part of the set.
    x_train_actual = x_train[:]
    x_validation_actual = x_validation[:]
    x_test_actual = x_test[:]
    board = TensorBoard(log_dir='logs/{}'.format(name))
    epochs = config_json['epochs']
    batch_size = config_json['batch_size']
    if continue_at != 0:
        for i in range(continue_at):
            model = keras.saving.load_model(forecast_name+'_horizon_{}.keras'.format(i))
            #Adding the prediction in the last part
            preds = model.predict(x_train_actual, batch_size= batch_size)
            preds = recolorize_predictions(preds, classes)
            x_train_actual = add_last(x_train_actual, preds[:])
            preds = model.predict(x_validation_actual, batch_size= batch_size)
            preds = recolorize_predictions(preds, classes)
            x_validation_actual = add_last(x_validation_actual, preds[:])
            #The test predictions will be saved, the others are only for DirRec strategy flow
            predictions = model.predict(x_test_actual, batch_size= batch_size)
            predictions = recolorize_predictions(predictions, classes)
            x_test_actual = add_last(x_test_actual, predictions[:])
            total_preds.append(predictions)

    for h in range(continue_at, horizon):
        print("** EVALUANDO MODELO PARA EL HORIZONTE {} **".format(h+1))
        y_train_actual = y_train[:,h]
        y_validation_actual = y_validation[:,h]
        y_test_actual = y_test[:,h]
        inp = keras.layers.Input(shape= (None, *x_train_actual.shape[2:]))
        m = model_multi_step_2(inp, channels)
        model = keras.models.Model(inp, m)
        model.compile(loss = 'mae', optimizer= optimizer)
        model.summary()

        #Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= early_stopping_value, restore_best_weights= True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 3)
        
        model.fit(
            x_train_actual, y_train_actual,
            batch_size= batch_size,
            epochs = epochs,
            validation_data= (x_validation_actual, y_validation_actual),
            callbacks = [reduce_lr, early_stopping]
        )

        if display:
            example = x_test_actual[np.random.choice(range(len(x_test_actual)), size= 1)[0]]
            print(example.shape)
            for _ in range(horizon):
                print(example.shape)
                new_prediction = model.predict(example.reshape(1,*example.shape[0:]))
                example = np.concatenate((example[1:], new_prediction), axis=0)
                print(example.shape)
            predictions = example[:-4]
            print(predictions.shape)
        
        err = model.evaluate(x_test_actual, y_test_actual, batch_size= batch_size)
        print("El error del modelo es: {}".format(err))
        #forecast = map_forecast_recursive(model, x_test, horizon)
        #forecast = model.predict(x_test, batch_size= 2)

        #Adding the prediction in the last part
        preds = model.predict(x_train_actual, batch_size= batch_size)
        preds = recolorize_predictions(preds, classes)
        x_train_actual = add_last(x_train_actual, preds[:])

        preds = model.predict(x_validation_actual, batch_size= batch_size)
        preds = recolorize_predictions(preds, classes)
        x_validation_actual = add_last(x_validation_actual, preds[:])

        #The test predictions will be saved, the others are only for DirRec strategy flow
        predictions = model.predict(x_test_actual, batch_size= batch_size)
        predictions = recolorize_predictions(predictions, classes)
        x_test_actual = add_last(x_test_actual, predictions[:])
        
        total_preds.append(predictions)
        new_name = forecast_name+'_horizon_{}'.format(h)
        model.save(new_name+'.keras')
        print("Modelo directo recursivo almacenado en: {}".format(new_name))

    total_preds = np.array(total_preds)
    print(total_preds.shape)
    total_preds = np.transpose(total_preds, (1,0,2,3,4))
    print(total_preds.shape)
    np.save(forecast_name+'.npy', total_preds)
    print("Pronósticos almacenados en: {}".format(forecast_name))

def DIRMO_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value, prediction_batch=4, continue_at=0):
    total_preds = None
    forecast_name = "Models/{}".format(name)
    inp = keras.layers.Input(shape= (x_train.shape[1:]))
    print(inp)
    #Number of times for the loop to process
    if prediction_batch > horizon:
        steps = 1
    else:
        steps = math.ceil(horizon / prediction_batch)

    if continue_at != 0:
        for i in range(continue_at+1):
            model = keras.saving.load_model(forecast_name+'_horizon_{}_{}.keras'.format(i, prediction_batch))
            forecast = model.predict(x_test, batch_size= batch_size)
            total_preds.append(forecast)

    for step in range(continue_at, steps):
        print("** EVALUANDO MODELO PARA EL PASO {} **".format(step+1))
        print("PASO:{}, {}".format(step*prediction_batch, (step*prediction_batch)+prediction_batch))
        if step*prediction_batch > horizon:
            y_train_actual = y_train[:, step*prediction_batch : horizon]
            y_validation_actual = y_validation[:, step*prediction_batch : horizon]
            y_test_actual = y_test[:, step*prediction_batch : horizon]
        else:
            y_train_actual = y_train[:, step*prediction_batch : (step*prediction_batch)+prediction_batch]
            y_validation_actual = y_validation[:, step*prediction_batch : (step*prediction_batch)+prediction_batch]
            y_test_actual = y_test[:, step*prediction_batch : (step*prediction_batch)+prediction_batch]
        
        m = testing_MIMO5(inp, prediction_batch)
        model = keras.models.Model(inp, m)
        model.compile(loss = 'mae', optimizer= optimizer)
        model.summary()
        #Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= early_stopping_value, restore_best_weights= True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 3)
        board = TensorBoard(log_dir='logs/{}'.format(name))
        epochs = config_json['epochs']
        batch_size = config_json['batch_size']
        print(model.output_shape)
        model.fit(
            x_train, y_train_actual,
            batch_size= batch_size,
            epochs = epochs,
            validation_data= (x_validation, y_validation_actual),
            callbacks = [reduce_lr, early_stopping]
        )

        err = model.evaluate(x_test, y_test_actual, batch_size= batch_size)
        print("El error del modelo es: {}".format(err))
        #forecast = map_forecast_recursive(model, x_test, horizon)
        forecast = model.predict(x_test, batch_size= batch_size)
        if total_preds is None:
            total_preds = forecast
        else:
            total_preds = np.concatenate((total_preds, forecast), axis=1)
        new_name = forecast_name+'_horizon_{}_{}'.format(step*prediction_batch, (step*prediction_batch)+prediction_batch)
        model.save(new_name+'.keras')
        print("Modelo directo almacenado en: {}".format(new_name))

    print(total_preds.shape)
    np.save(forecast_name+'.npy', total_preds)
    print("Pronósticos almacenados en: {}".format(forecast_name))


def main(config_file, load_and_forecast=False, model_name='', display= False):
    config_json = read_json_file(config_file)
    window = config_json['window_size']
    rows = config_json['rows']
    cols = config_json['cols']
    channels = config_json['channels']
    horizon = config_json['horizon']
    name = config_json['name'] + '_model_testing_{}'.format(int(time.time()))
    #name = config_json['name'] + '_model_testing_{}'.format(1754031464)
    #name = config_json['name'] + '_model_testing_{}'.format(1755604147)
    
    optimizer = config_json['optimizer']
    data_name = '{}/{}.npy'.format(config_json['folder_models_save'], config_json['folder'])
    early_stopping_value = config_json['deep_training_early_stopping_patience']

    preprocess = Preprocessing()
    preprocess.load_from_numpy_array(data_name, rows, cols, channels)
    #For recursive strategy
    x_train, y_train, x_validation, y_validation, x_test, y_test = preprocess.create_STI_dataset(window)

    #For direct, MIMO, DirRec, DIRMO
    #x_train, y_train, x_validation, y_validation, x_test, y_test = preprocess.create_STI_multi_output(window, horizon)

    start_time = time.time()
    strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
    with strategy.scope():
        if load_and_forecast:
            #Cuidar las entradas por si son simples o múltiples
            model = keras.models.load_model(model_name)
            err = model.evaluate(x_test, y_test, batch_size= 2)
            print("El error del modelo es: {}".format(err))

            forecast = map_forecast_recursive(model, x_test, horizon)
            forecast_name = "Models/{}".format(model_name)
            np.save(forecast_name+'.npy', forecast)
            print("Pronósticos almacenados en: {}".format(forecast_name))
            return
        
        #Recursive strategy
        recursive_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value)

        #Direct strategy
        #direct_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value)

        #MIMO strategy
        #MIMO_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value)

        #DirRec strategy
        #DirRec_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value)

        #DIRMO strategy
        #DIRMO_strategy(x_train, y_train, x_validation, y_validation, x_test, y_test, name, display, horizon, channels, optimizer, config_json, early_stopping_value, prediction_batch=4)

    processing_time = time.time() - start_time
    print(f"Elapsed Time: {processing_time:.4f}")


if __name__ == '__main__':
    main('Conv-LSTM_1.json', display=True)