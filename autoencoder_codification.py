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

def model_1(inp, channels):
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(64, (3,3), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.ConvLSTM2D(channels, (3,3), padding= "same", activation= "relu")(m)
    return m

def model_2(inp, channels):
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    #m = keras.layers.Dropout(0.25)(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    return m


def get_model_shape(model, layer_i= -1):
    layer = model.layers[layer_i]
    layer_shape = layer.output.shape[1:]
    return layer_shape
    
def get_encoder(model, input_dim):
    encoder = input_dim
    cant = int(len(model.layers)/2) - 1
    for i in range(cant):
        encoder = model.layers[i+1](encoder)
    encoder = keras.Model(input_dim, encoder)
    return encoder

def get_decoder(model):
    cant = int(len(model.layers)/2)
    decoder_input = get_model_shape(model, cant)
    print("Actual model shape: {} con {} y {}".format(decoder_input, model, cant))
    input_shape = keras.layers.Input(shape= decoder_input)
    decoder = input_shape
    for i in reversed(range(cant)):
        decoder = model.layers[-(i+1)](decoder)
    decoder = keras.Model(input_shape, decoder)
    return decoder, decoder_input

def construct_autoencoder(CMPBlocks, input_shape: tuple):
    input = keras.layers.Input(input_shape)
    layers = input
    #Encoder layers
    for CMPBlock in CMPBlocks:
        layers = keras.layers.Conv2D(CMPBlock[0], CMPBlock[1], activation= CMPBlock[2], padding= 'same')(layers)
        layers = keras.layers.MaxPooling2D(CMPBlock[3])(layers)
    #Decoder layers
    for CMPBlock in reversed(CMPBlocks):
        layers = keras.layers.Conv2D(CMPBlock[0], CMPBlock[1], activation= CMPBlock[2], padding= 'same')(layers)
        layers = keras.layers.UpSampling2D(CMPBlock[3])(layers)
    layers = keras.layers.Conv2D(input_shape[2], (3,3), activation= 'relu', padding= 'same')(layers)
    autoencoder = keras.Model(input, layers)
    #print(autoencoder.summary())
    encoder = get_encoder(autoencoder, input)
    #print(tf.keras.Model(self.input, encoder).summary())
    decoder, _ = get_decoder(autoencoder)
    return autoencoder, encoder, decoder

def conf_1 ():
    #32,400  25% reduction
    return [[3, (3,3), "relu", (2,2)]]

def conf_2 ():
    #21,600 50% reduction
    return [[2, (3,3), "relu", (2,2)]]

def conf_3 ():
    #10,800 75% reduction
    return [[1, (3,3), "relu", (2,2)]]

def conf_4 ():
    #21,600 50% reduction
    return [[16, (3,3), "relu", (2,2)],
            [8, (3,3), "relu", (2,2)]]

def conf_5 ():
    #16,200 62.5% reduction
    return [[12, (3,3), "relu", (2,2)],
            [6, (3,3), "relu", (2,2)]]

def conf_6 ():
    #10,800 75% reduction
    return [[8, (3,3), "relu", (2,2)],
            [4, (3,3), "relu", (2,2)]]
def conf_7 ():
    #5,400 87.5% reduction
    return [[4, (3,3), "relu", (2,2)],
            [2, (3,3), "relu", (2,2)]]


def autoencoder_reduction(data, validation_part= 0.3):
    print(data.shape)
    x_train = data[int(len(data) * validation_part):]
    x_val = data[: int(len(data) * validation_part)]
    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    #x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    #CMPBlocks = [[8, (3,3), "relu", (2,2)],
    #             [4, (3,3), "relu", (2,2)]]
    CMPBlocks = conf_4()
    autoencoder, encoder, decoder = construct_autoencoder(CMPBlocks, (data.shape[1], data.shape[2], 1))
    es = keras.callbacks.EarlyStopping(monitor= 'val_loss', mode= 'min', patience= 10, restore_best_weights= True)
    autoencoder.compile(optimizer= "adam", loss="mae")
    autoencoder.summary()
    print(es)
    history = autoencoder.fit(x_train, x_train, epochs= 100, validation_data= (x_val, x_val), shuffle=True, verbose= 1, callbacks= [es])
    print(history)
    encoded_data = encoder.predict(data)


    ##
    # CAMBIAR EL NOMBRE DE LOS DECODER, ALMACENAR POR CONFIGURACIÓN PARA PODER REPETIR EXPERIMENTOS O EVALUACIÓN
    ##

    decoder.save("Models/actual_decoder.h5")
    return encoded_data


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

    #Autoencoder process.
    preprocess = Preprocessing()
    preprocess.load_from_numpy_array(data_name, rows, cols, channels)
    strategy = tf.distribute.OneDeviceStrategy(device= '/gpu:0')
    data = autoencoder_reduction(preprocess.get_full_data())
    #preprocess.autoencoder_codification(data)
    print(data.shape)
    x_train_cod, y_train_cod, x_validation_cod, y_validation_cod, x_test_cod, y_test_cod = preprocess.autoencoder_codification(data, window)

    #print(len(x_train_frags))
    #print(x_train_frags[0].shape)
    #print(len(y_train_frags))
    #print(y_train_frags[0].shape)
    count = 0
    i=0
    while i < len(x_train_cod):

    #for i in range(len(x_train_cod)):
        print("Processing codification part {}".format(i))
        x_train = x_train_cod[i]
        y_train = y_train_cod[i]
        x_validation = x_validation_cod[i]
        y_validation = y_validation_cod[i]
        x_test = x_test_cod[i]
        y_test = y_test_cod[i]
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
            m = model_2(inp, channels)

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
            if err > 0.1 and count < 3:
                print("Previous values i: {} count:{}".format(i, count))
                count += 1
                i -= 1
                print("Not accomplish enough loss value part {} times {}".format(i, count))
                continue
            print("El error del modelo es: {}".format(err))
            count = 0

            forecast = map_forecast_recursive(model, x_test, horizon)
            forecast_name = "Models/{}".format(name)
            model.save(forecast_name+'_'+str(i)+'.keras')
            np.save(forecast_name+'_'+str(i)+'.npy', forecast)
            print("Pronósticos almacenados en: {}".format(forecast_name))
            i += 1

if __name__ == '__main__':
    main('Conv-LSTM_1.json')