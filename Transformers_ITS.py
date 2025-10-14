import numpy as np
import cv2
import os
import json
import time
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pandas as pd
from app.common.color_tools import *
from keras import layers, ops
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from mapPreprocessing import Preprocessing

#Características de un transformer
class TubeletEmbedding(layers.Layer):
  def __init__(self, embed_dim, patch_size, **kwargs):
    super().__init__(**kwargs)
    self.projection = layers.Conv3D(
        filters = embed_dim,
        kernel_size = patch_size,
        strides = patch_size,
        padding = "VALID"
    )
    self.flatten = layers.Reshape(target_shape= (-1, embed_dim))

  def call(self, videos):
    projected_patches = self.projection(videos)
    flattened_patches = self.flatten(projected_patches)
    return flattened_patches
  
class PositionalEncoder(layers.Layer):
    #embed_dim= Dimensiones del vector de características resultante.
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = ops.arange(0, num_tokens, 1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, data.shape[1] - 1, :, :]
    return x, y

def read_json_file(filename):
    f = open('configurations/{}'.format(filename), "r")
    parameters = json.load(f)
    print(type(parameters))
    return parameters

def agroup_window(data, window):
    new_data = [data[i : window + i] for i in range(len(data) - window + 1)]
    return np.array(new_data)

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

config_json = read_json_file('Conv-LSTM_1.json')
window = config_json['window_size']
rows = config_json['rows']
cols = config_json['cols']
channels = config_json['channels']
horizon = config_json['horizon']
name = config_json['name'] + '_model_testing_{}'.format(int(time.time()))
optimizer = config_json['optimizer']
data_name = '{}/{}.npy'.format(config_json['folder_models_save'], config_json['folder'])
early_stopping_value = config_json['deep_training_early_stopping_patience']

#window = 8
#channels = 1
#rows = 120
#cols = 360
#categories = np.array([0, 51, 102, 153, 204, 255])
#horizon = 4

#MAX_SEQ_LENGTH = 20
#NUM_FEATURES = 1024
#IMG_SIZE = 128
#EPOCHS = 5

#DATA
DATASET_NAME = "usdrought"
BATCH_SIZE = 2
WINDOW_SIZE = window - 1
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (WINDOW_SIZE, 120, 360, 1)

#OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

#TRAINING
EPOCHS = 150

def vivit_params_tesis_1():
   return (4,8,8), 16, 4, 4

def vivit_params_tesis_2():
   return (4,10,10), 32, 4, 4

def vivit_params_tesis_3():
   return (4,12,12), 64, 8, 6

def vivit_params_tesis_4():
   return (4,16,16), 128, 8, 6

def vivit_params_tesis_5():
   return (2,14,14), 16, 8, 8

def vivit_params_tesis_6():
   return (2,14,14), 32, 8, 10

def vivit_params_tesis_7():
   return (2,16,16), 48, 10, 10

def vivit_params_tesis_8():
   return (2,16,16), 64, 12, 10

PATCH_SIZE, PROJECTION_DIM, NUM_HEADS, NUM_LAYERS = vivit_params_tesis_8()

#TUBELET EMBEDDING
#PATCH_SIZE = (4, 4, 12)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

#ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
#PROJECTION_DIM = 64
#NUM_HEADS = 16
#NUM_LAYERS = 4



data = np.load("Models/ProcessedDroughtDataset.npy")
print(data.shape)

#Mostrar imágenes
fig, axes = plt.subplots(2, 3, figsize= (10,8))

data_choise = np.random.choice(range(len(data)), size= 1)[0]
for idx, ax in enumerate(axes.flat):
    ax.imshow(np.squeeze(data[data_choise+idx]), cmap='gray')
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")

plt.show()

preprocess = Preprocessing()
preprocess.load_from_numpy_array(data_name, rows, cols, channels)
x_train, y_train, x_validation, y_validation, x_test, y_test = preprocess.create_STI_dataset(window)

def resize_dims(tensor):
    tensor = tf.image.resize(tensor, (120,360))
    return tensor

def vivit_cnn_tesis_1(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(16, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def vivit_cnn_tesis_2(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(32, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def vivit_cnn_tesis_3(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(64, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def vivit_cnn_tesis_4(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(32, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(16, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def vivit_cnn_tesis_5(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(64, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(32, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def vivit_cnn_tesis_6(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(64, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(32, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(16, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def vivit_cnn_tesis_7(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(32, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(16, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(8, kernel_size = (3,3), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def vivit_cnn_tesis_8(prev):
    m = layers.Conv2DTranspose(1, (3,3), strides= (3,3), padding='same', activation='relu')(prev)
    m = keras.layers.Lambda(resize_dims)(m)
    m = layers.Conv2D(64, kernel_size = (5,5), padding='same', activation='relu')(m)
    m = layers.Conv2D(64, kernel_size = (5,5), padding='same', activation='relu')(m)
    m = layers.Conv2D(64, kernel_size = (5,5), padding='same', activation='relu')(m)
    m = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(m)
    return m

def video_transformer(
    tubelet_embedder,
    positional_encoder,
    input_shape = INPUT_SHAPE,
    transformer_layers = NUM_LAYERS,
    num_heads = NUM_HEADS,
    embed_dim = PROJECTION_DIM,
    layer_norm_eps = LAYER_NORM_EPS
  ):
  #Create input layer
  inputs = layers.Input(shape= input_shape)
  #create patches
  patches = tubelet_embedder(inputs)
  #Encode patches
  encoded_patches = positional_encoder(patches)

  #Create multiple layers of the transformer block
  for _ in range(transformer_layers):
    #Layer normalization and MHSA
    #x1 = layers.LayerNormalization(epsilon= 1e-6)(patches)
    x1 = layers.LayerNormalization(epsilon= 1e-6)(encoded_patches)
    attention_output = layers.MultiHeadAttention(
        num_heads = num_heads, key_dim = embed_dim // num_heads, dropout = 0.1
    )(x1, x1)

    #Skip connection
    x2 = layers.Add()([attention_output, encoded_patches])
    #x2 = layers.Add()([attention_output, patches])

    #Layer Normalization and MLP
    x3 = layers.LayerNormalization(epsilon= 1e-6)(x2)
    x3 = keras.Sequential(
        [
            layers.Dense(units= embed_dim * 4, activation = ops.gelu),
            layers.Dense(units= embed_dim, activation = ops.gelu)
        ]
    )(x3)

    #skip connection
    encoded_patches = layers.Add()([x3, x2])
    #patches = layers.Add()([x3, x2])
    #patches = layers.Reshape((688, 128, 1))(patches)

    #patches = layers.Conv2D(1, kernel_size = (2,1))(patches)

  representation = layers.LayerNormalization(epsilon= layer_norm_eps)(encoded_patches)

  

  #representation = layers.Reshape((616, 128, 1))(representation)
  #----
  #representation = layers.Reshape((462, 128, 1))(representation)
  #representation = layers.Reshape((900, 64, 1))(representation)

  #ADAPTAR A LAS CONFIGURACIONES!
  #Model_testing_1
  #w=4, 5, 6, 7
  #representation = layers.Reshape((675, 16, 1))(representation)
  #w=8, 9, 10
  #representation = layers.Reshape((1350, 16, 1))(representation)
  #Model_testing_2
  #w=4, 5, 6, 7
  #representation = layers.Reshape((432, 32, 1))(representation)
  #w=8, 9, 10
  #representation = layers.Reshape((864, 32, 1))(representation)
  #Model_testing_3
  #w=4, 5, 6, 7
  #representation = layers.Reshape((300, 64, 1))(representation)
  #w=8, 9, 10
  #representation = layers.Reshape((600, 64, 1))(representation)
  #Model_testing_4
  #w=4, 5, 6, 7
  #representation = layers.Reshape((154, 128, 1))(representation)
  #w=8, 9, 10
  #representation = layers.Reshape((308, 128, 1))(representation)

  #Model_testing_5
  #w=4, 5
  #representation = layers.Reshape((400, 16, 1))(representation)
  #w=6, 7 
  #representation = layers.Reshape((600, 16, 1))(representation)
  #w=8, 9
  #representation = layers.Reshape((800, 16, 1))(representation)
  #w=10 -
  #representation = layers.Reshape((1000, 16, 1))(representation)
  #Model_testing_6
  #w=4, 5
  #representation = layers.Reshape((400, 32, 1))(representation)
  #w=6, 7 
  #representation = layers.Reshape((600, 32, 1))(representation)
  #w=8, 9
  #representation = layers.Reshape((800, 32, 1))(representation)
  #w=10 -200
  #representation = layers.Reshape((1000, 32, 1))(representation)
  #Model_testing_7
  #w=4, 5
  #representation = layers.Reshape((308, 48, 1))(representation)
  #w=6, 7 
  #representation = layers.Reshape((462, 48, 1))(representation)
  #w=8, 9
  #representation = layers.Reshape((616, 48, 1))(representation)
  #w=10 -177
  #representation = layers.Reshape((770, 48, 1))(representation)
  #Model_testing_8
  #w=4, 5
  #representation = layers.Reshape((308, 64, 1))(representation)
  #w=6, 7 
  #representation = layers.Reshape((462, 64, 1))(representation)
  #w=8, 9
  representation = layers.Reshape((616, 64, 1))(representation)
  #w=10 -154
  #representation = layers.Reshape((770, 64, 1))(representation)

  #representation = layers.Conv2D(16, kernel_size = (3,3), strides=(2,1), padding='same', activation='relu')(representation)
  #representation = layers.Conv2D(8, kernel_size = (3,3), strides=(2,1), padding='same', activation='relu')(representation)
  #----
  #representation = layers.Conv2D(1, kernel_size = (3,3), strides=(2,1), padding='same', activation='relu')(representation)


  m = vivit_cnn_tesis_8(representation)

  
  #representation = layers.Conv2D(64, kernel_size = (5,5), padding='same', activation='relu')(representation)
  #representation = layers.Reshape((338, 128))(representation)
  #representation = layers.GlobalAvgPool1D()(representation)
  #----
  #representation = layers.Flatten()(representation)
  #representation = layers.Conv2D(64, (3,3), activation= "relu", padding= "same")(representation)
  #cnn = keras.layers.MaxPooling2D((2,2), padding="same")(cnn)
  #representation = layers.Conv2D(32, (5,5), activation= "relu", padding= "same")(representation)
  #cnn = keras.layers.MaxPooling2D((2,2), padding="same")(cnn)
  #representation = layers.Conv2D(16, (3,3), activation= "relu", padding= "same")(representation)

  #representation = layers.Conv2D(8, (3, 3), strides=(4, 1), padding='same', activation='relu')(representation)

  #representation = layers.Conv2DTranspose(1, (3,3), strides= (1,3), padding='same', activation='relu')(representation)
  #def resize_dims(tensor):
  #  tensor = tf.image.resize(tensor, (120,360))
  #  return tensor
  #representation = (lambda x: tf.image.resize(x, (120, 360)))(representation)
  #representation = keras.layers.Lambda(resize_dims)(representation)


  #x = layers.Dense(10800, activation= 'relu')(representation)
  #----
  #x = layers.Dense(2700, activation= 'relu')(representation)
  #----
  #x = keras.layers.BatchNormalization()(x)
  #outputs = layers.Reshape((input_shape[1],input_shape[2],1))(x)
  #----
  #cnn = layers.Reshape((30,90,1))(x)
  #cnn = layers.Reshape((60,180,1))(x)
  #----
  #cnn = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(cnn)
  #----
  #cnn = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(cnn)
  #cnn = keras.layers.BatchNormalization()(cnn)
  #cnn = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(cnn)
  #----
  
  #outputs = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(representation)


  #embeddings = layers.TimeDistributed(patches)(inputs)
  model = keras.models.Model(inputs, m)
  return model

def run_experiment():
  #Initializing model
  model = video_transformer(
      tubelet_embedder = TubeletEmbedding(
          embed_dim= PROJECTION_DIM,
          patch_size= PATCH_SIZE
      ),
      positional_encoder= PositionalEncoder(embed_dim= PROJECTION_DIM)
  )
  #Compile the model with the optimizer, loss function and the metrics
  #optimizer = keras.optimizers.Adam(learning_rate= LEARNING_RATE)
  model.compile(
      optimizer= 'Adam',
      loss= "mae",
  )
  model.summary()
  early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 10, restore_best_weights= True)
  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 3)
  history = model.fit(x_train, y_train, epochs= EPOCHS, validation_data= (x_validation, y_validation),callbacks = [reduce_lr, early_stopping])
  return model

strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
with strategy.scope():
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = run_experiment()

    preds = model.predict(x_test)
    preds.shape

    example = x_test[np.random.choice(range(len(x_test)), size= 1)[0]]

    print(example.shape)

    for _ in range(horizon):
        print(example.shape)
        new_prediction = model.predict(example.reshape(1,*example.shape[0:]))
        example = np.concatenate((example[1:], new_prediction), axis=0)
        print(example.shape)
        

    predictions = example[:]
    print(predictions.shape)

    fig, axes = plt.subplots(2,3, figsize= (20,4))
    for idx, ax in enumerate(axes[0]):
        ax.imshow((predictions[idx]), cmap='gray')
        ax.set_title("Frame {}".format(idx+3))
        ax.axis("off")
    plt.show()

    err = model.evaluate(x_test, y_test, batch_size= 2)
    print("El error del modelo es: {}".format(err))
    """preds = model.predict(x_test, batch_size= 2)
    print(preds.shape)
    x_test_new = add_last(x_test, preds[:])
    preds2 = model.predict(x_test_new, batch_size= 2)
    x_test_new = add_last(x_test_new, preds2[:])
    preds3 = model.predict(x_test_new, batch_size= 2)
    x_test_new = add_last(x_test_new, preds3[:])
    preds4 = model.predict(x_test_new, batch_size= 2)
    res_forecast = add_last(x_test_new, preds4[:])
    print("PREDSS",res_forecast.shape)

    np.save("Models/PredictionsTransformers.npy", res_forecast)"""
    forecast = map_forecast_recursive(model, x_test, horizon)
    forecast_name = "Models/{}".format(name)
    model.save(forecast_name+'.keras')
    np.save(forecast_name+'.npy', forecast)
    print("Pronósticos almacenados en: {}".format(forecast_name))