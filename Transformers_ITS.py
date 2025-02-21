import numpy as np
import cv2
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pandas as pd
from app.common.color_tools import *
from keras import layers, ops
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


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

window = 8
channels = 1
rows = 120
cols = 360
categories = np.array([0, 51, 102, 153, 204, 255])
horizon = 4

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
EPOCHS = 50

#TUBELET EMBEDDING
PATCH_SIZE = (2, 16, 16)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

#ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

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
"""
args = [(d, categories) for d in data]

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
x_greys = np.array(results)
x = x_greys.astype('float32') / 255"""

x_2 = agroup_window(data, window)
print(x_2.shape)
x_train = x_2[: int(len(x_2) * .7)]
x_test = x_2[int(len(x_2) * .7) :]
x_validation = x_train[int(len(x_train) * .8) :]
x_train = x_train[: int(len(x_train) * .8)]

x_train = x_train.reshape(len(x_train), window, rows, cols, channels)
x_validation = x_validation.reshape(len(x_validation), window, rows, cols, channels)
x_test = x_test.reshape(len(x_test), window, rows, cols, channels)

print("Forma de datos de entrenamiento: {}".format(x_train.shape))
print("Forma de datos de validación: {}".format(x_validation.shape))
print("Forma de datos de pruebas: {}".format(x_test.shape))

x_train, y_train = create_shifted_frames(x_train)
x_validation, y_validation = create_shifted_frames(x_validation)
x_test, y_test = create_shifted_frames(x_test)

print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

np.save("Models/x_test_transformer_greys_forecast.npy", x_test)
np.save("Models/y_test_transformer_greys_forecast.npy", y_test)

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
  representation = layers.Reshape((462, 128, 1))(representation)
  #representation = layers.Conv2D(16, kernel_size = (3,3), strides=(2,1), padding='same', activation='relu')(representation)
  #representation = layers.Conv2D(8, kernel_size = (3,3), strides=(2,1), padding='same', activation='relu')(representation)
  representation = layers.Conv2D(1, kernel_size = (3,3), strides=(2,1), padding='same', activation='relu')(representation)
  #representation = layers.Reshape((338, 128))(representation)
  #representation = layers.GlobalAvgPool1D()(representation)
  representation = layers.Flatten()(representation)

  x = layers.Dense(2700, activation= 'relu')(representation)
  x = keras.layers.BatchNormalization()(x)
  #outputs = layers.Reshape((input_shape[1],input_shape[2],1))(x)
  cnn = layers.Reshape((30,90,1))(x)
  cnn = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(cnn)
  cnn = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(cnn)
  #cnn = keras.layers.BatchNormalization()(cnn)
  #cnn = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(cnn)
  outputs = layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(cnn)


  #embeddings = layers.TimeDistributed(patches)(inputs)
  model = keras.Model(inputs= inputs, outputs= outputs)
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
  history = model.fit(x_train, y_train, epochs= EPOCHS, validation_data= (x_validation, y_validation))
  return model

strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
with strategy.scope():
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
    preds = model.predict(x_test, batch_size= 2)
    print(preds.shape)
    x_test_new = add_last(x_test, preds[:])
    preds2 = model.predict(x_test_new, batch_size= 2)
    x_test_new = add_last(x_test_new, preds2[:])
    preds3 = model.predict(x_test_new, batch_size= 2)
    x_test_new = add_last(x_test_new, preds3[:])
    preds4 = model.predict(x_test_new, batch_size= 2)
    res_forecast = add_last(x_test_new, preds4[:])
    print("PREDSS",res_forecast.shape)

    np.save("Models/PredictionsTransformers.npy", res_forecast)