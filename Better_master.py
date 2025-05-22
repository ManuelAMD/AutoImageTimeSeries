from distutils.log import error
from gc import callbacks
from json import decoder
from pickle import NONE
from random import sample
from symbol import try_stmt
import tempfile
import shutil
import datetime
import traceback
from typing_extensions import dataclass_transform
from keras import backend as K
import numpy as np
import os
import json
import pika
from load_imgs import load_imgs, to_monochromatic
#from ThreadRabbitMQ import ThreadRabbitMQ
import tensorflow as tf
import asyncio
from RabbitMQ import RabbitMQ
import matplotlib as mat
import matplotlib.image as img
from clear_queues import clear
import gc
import pandas as pd
from pathlib import Path

class BetterMaster():

    def __init__(self, json_file= 'autoencoderConfig.json'):
        #asyncio.set_event_loop(asyncio.new_event_loop)
        self.loop = asyncio.get_event_loop()
        f = open(json_file, "r")
        self.parameters_data = json.loads(f.read())
        print(self.parameters_data)
        self.rows = self.parameters_data['rows']
        self.cols = self.parameters_data['cols']
        self.channels = self.parameters_data['channels']
        print("({}, {}, {})".format(self.rows, self.cols, self.channels))
        self.color_mode = self.parameters_data['color_mode']
        self.dataset_name = self.parameters_data['name']
        self.img_type = self.parameters_data['img_type']
        self.data_folder = self.parameters_data['folder']
        self.names_file = self.parameters_data['names_file']
        self.train_size = self.parameters_data['train_size']
        self.validation_size = self.parameters_data['validation_size']
        self.epochs = self.parameters_data['epochs']
        self.batch_size = self.parameters_data['batch_size']
        self.predictions_size = self.parameters_data['predictions_size']
        self.window_size = self.parameters_data['window_size'] 
        print("{}, {}, {}, {}, {}".format(self.dataset_name, self.img_type, self.data_folder, self.names_file, self.color_mode))
        print("{}, {}, {}, {}, {}".format(self.train_size, self.epochs, self.batch_size, self.predictions_size, self.window_size))
        
        self.with_autoencoder = self.parameters_data['with_autoencoder']
        self.with_feature_selection = self.parameters_data['with_feature_selection']
        self.optimizer = self.parameters_data['optimizer']
        self.early_stopping_patience = self.parameters_data['early_stopping_patience']
        self.CMPBlocks = self.parameters_data['autoencoder_CMPBlocks']
        for cmpblock in self.CMPBlocks:
            cmpblock[1] = tuple(cmpblock[1])
            cmpblock[3] = tuple(cmpblock[3])
        print("{}, {}, {}".format(self.optimizer, self.early_stopping_patience, self.CMPBlocks))
        self.host = self.parameters_data['host']
        self.user = self.parameters_data['user']
        self.password = self.parameters_data['password']
        self.port = self.parameters_data['port']
        self.queue_publish = self.parameters_data['queue_publish']
        self.queue_results = self.parameters_data['queue_results']
        self.temporal = tempfile.mkdtemp()
        self.save_path = self.parameters_data['folder_models_save']
        self.band = False
        self.info_res = []

    def tensorflow_definition(self):
        config = tf.compat.v1.ConfigProto(gpu_options= tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction= 0.8))
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config = config)
        tf.compat.v1.keras.backend.set_session(session)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        return config, session, options

    def tensorflow_dataset_definition(self):
        return

    def start_master(self):
        try:
            init_time = datetime.datetime.now()
            print("Initital time of the process: {}".format(init_time.strftime("%Y%m%d-%H%M%S")))
            print("Getting the data...")
            self.x = self.load_and_prepare_all_data()
            self.x_train, self.x_test = self.load_and_prepare_data()
            #np.save('Models/Data_select.npy', self.x)
            pred_size = len(self.x_test) - self.window_size
            print(pred_size)
            self.x_train_part, self.x_validation = self.get_validation_data(self.x_train, self.validation_size)
            print(self.x.shape)
            if self.with_feature_selection:
                if Path("Models/points.npy").is_file():
                    c = np.load("Models/points.npy")
                else:
                    c = self.checkAllChanges(self.x)
                    np.save("Models/points.npy", c)
                print(c.shape)
                print(c[:10])
                if Path("Models/x_train_s.npy").is_file():
                    self.x_train = np.load("Models/x_train_s.npy")
                else:
                    self.x_train = self.get_images_info(self.x_train, c)
                    np.save("Models/x_train_s.npy", self.x_train)
                print(self.x_train.shape)
                if Path("Models/x_test_s.npy").is_file():
                    self.x_test = np.load("Models/x_test_s.npy")
                else:
                    self.x_test = self.get_images_info(self.x_test, c)
                    np.save("Models/x_test_s.npy", self.x_test)
                print(self.x_test.shape)
            #print(self.x_train[50][400][500:520])
            print("Finishing getting the data!")
        except:
            print('Cannot load or prepare data.')
            shutil.rmtree(self.temporal)
            raise
        try:
            if self.with_autoencoder:
                print('Constructing Autoencoder...')
                config, session, options = self.tensorflow_definition()
                dataset = tf.data.Dataset.from_tensor_slices((self.x_train_part, self.x_train_part))
                dataset = dataset.with_options(options)
                dataset_val = tf.data.Dataset.from_tensor_slices((self.x_validation, self.x_validation))
                dataset_val = dataset_val.with_options(options)
                strategy = tf.distribute.MirroredStrategy()
                self.gpu_cant = strategy.num_replicas_in_sync
                self.batch_size = self.batch_size * self.gpu_cant
                dataset = dataset.batch(self.batch_size)
                dataset_val = dataset_val.batch(self.batch_size)
                with strategy.scope():
                    self.autoencoder, self.encoder, self.decoder = self.construct_autoencoder((self.rows, self.cols, self.channels))
                    print(self.autoencoder.summary())
                    #print(self.encoder.summary())
                    #print(self.decoder.summary())      
                    es = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', mode= 'min', patience= self.early_stopping_patience, restore_best_weights= True)
                    self.autoencoder.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy')
                    print("Initializing autoencoder training...")
                    history = self.autoencoder.fit(dataset, epochs= self.epochs, validation_data=dataset_val, shuffle= True, verbose= 1, callbacks= [es])
                    print(history)
                    loss = self.autoencoder.evaluate(self.x_test, self.x_test, verbose= 0)
                    print("Autoencoder loss: {}".format(loss))
                print("Successfully construct autoencoder!")
                img.imsave('ResDrought/original.png', self.x_test[-1].reshape(self.rows, self.cols), cmap= 'Greys')
                encoded_img = self.encoder.predict(self.x_test[-1].reshape(1,self.rows, self.cols))
                decoded_img = self.decoder.predict(encoded_img.reshape(1, self.decoder_input[0], self.decoder_input[1], self.decoder_input[2]))
                autoencoder_img = self.autoencoder.predict(self.x_test[-1].reshape(1,self.rows, self.cols))
                img.imsave('ResDrought/prediction.png', decoded_img.reshape(self.rows, self.cols), cmap= 'Greys')
                img.imsave('ResDrought/autoencoder.png', autoencoder_img.reshape(self.rows, self.cols), cmap= 'Greys')
            else:
                self.decoder_input = (self.rows, self.cols, self.channels)
                loss=0.0
        except:
            print("Error trying to construct the autoencoder")
            shutil.rmtree(self.temporal)
            raise
        try:
            print("Initializing models prediction process...")
            init_train_time = datetime.datetime.now()
            if self.with_autoencoder:
                #self.x = self.transform_all_info(self.encoder, self.decoder_input)
                #np.save('latent_space.npy', self.x)
                self.train_matrix, self.test_matrix = self.transform_info(self.encoder, self.decoder_input)
                self.save_info_in_disk(self.autoencoder, self.encoder, self.decoder, self.train_matrix, self.test_matrix)
            else:
                if self.with_feature_selection:
                    tam_imgs = c.shape[0]
                else:
                    tam_imgs = self.rows * self.cols * self.channels
                #self.x = self.create_matrices(self.x, tam_imgs)
                print("HEEEYYYY {}".format(self.x_train.shape))
                #np.save('Data_select.npy', self.x)
                self.train_matrix = self.create_matrices(self.x_train, tam_imgs)
                #self.validation_matrix = self.create_matrices(self.x_validation,tam_imgs)
                self.test_matrix = self.create_matrices(self.x_test, tam_imgs)
                print("EOOOOOOO {}".format(self.train_matrix.shape))
                self.save_info_in_disk_no_autoencoder(self.train_matrix, self.test_matrix)
            print(self.train_matrix[10])
            print("Processing data, quantity: {}".format(len(self.train_matrix)))
            self.trianing_size = len(self.train_matrix)
            if Path("Models/predictions.txt").is_file():
                with open("Models/predictions.txt", 'r') as f:
                    aux = []
                    for line in f:
                        if line.rstrip('\n')== 'None':
                            aux.append(None)
                        else:
                            aux.append(np.array(line.rstrip('\n').strip('][').split(', '),dtype=np.float32).tolist())
                    #aux = [if : None else:  for line in f]
                #self.predictions = np.load("Models/predictions.npy")
                #print(aux)
                self.predictions = np.array(aux, dtype=object)
                #print(self.predictions.shape)
                #print(self.predictions)
                #print(type(self.predictions[-1]))
                print(np.where(self.predictions==None))
                initial = np.where(self.predictions==None)[0][0]
                self.predictions = self.predictions.tolist()
                self.recieved_size = int(initial)
                self.actual_train = int(initial)
                print("RESUMING TRAINING, POSITION {}".format(self.actual_train))
            else:
                self.predictions = [None] * self.trianing_size
                #self.trianing_size = 5
                self.recieved_size = 0
                self.actual_train = 0
            self.count_save = 10
            #self.x_train, self.y_train, self.x_val, self.y_val = np.array([]), np.array([]), np.array([]), np.array([])
            #self.x_train = self.x_train.reshape(0,0,0,0)
            #self.generate_dataset_async()
            self.x_test = self.test_matrix[:,-self.window_size:]
            #print(self.x_train[0])
            #print("Y[0]:",self.x_train[0])
            #print(self.x_test[0])
            print(self.predictions[0])
            try:
                connection = self.loop.run_until_complete(self.start_rabbitmq())
                #self.predictions = self.test_matrix[:, -10:]
                #print(self.predictions.shape)
            except Exception as e:
                print("Hubo un problema.")
                traceback.print_exc()
            print("Initializing models forecast!")
            self.predictions = np.array(self.predictions, dtype=np.float32)
            print(self.predictions[0])
            #self.predictions = np.array(self.predictions)
            self.predictions = self.predictions.transpose()
            print(self.predictions.shape)
            if self.with_feature_selection:
                predictions_aux = []
                for p in self.predictions:
                    predictions_aux.append(self.returnImage(p, c))
                self.predictions = np.array(predictions_aux, dtype=np.float32)
            print(self.predictions.shape)
            print(self.predictions[0])
            try:
                ##AQUIIIIIIIIIIIIIII
                print(self.x_test.shape)
                self.predictions = self.predictions.reshape((self.x_test.shape[0], self.decoder_input[0], self.decoder_input[1], self.decoder_input[2]))
            except:
                print("WEPPPPP")
                self.predictions = self.predictions.reshape((self.predictions.shape[0], self.decoder_input[0], self.decoder_input[1], self.decoder_input[2]))
            print(self.predictions)
            print(self.predictions.shape)
            if self.with_autoencoder:
                pred = self.decoder.predict(self.predictions)
            else:
                pred = self.predictions
            print(pred[0,1])
            pred = pred * 255
            print(pred[0,1])
            #pred = to_monochromatic(pred, 127)
            #print(pred[0])
            finish_train_time = datetime.datetime.now()
            train_time = finish_train_time - init_train_time
            folder_results = 'ResDrought/train{}_{}_{}_{}'.format(init_train_time.strftime("%Y%m%d-%H%M%S"), self.decoder_input, self.strfdelta(train_time, "{d}d-{h}h-{m}m-{s}s"), loss)
            #print("Pixels Color", pred[-1][69,36])
            print("Model prediction process finished!")
            try:
                os.mkdir(folder_results)
            except OSError:
                print('Creation of the directory ResDrought/{} failed'.format(folder_results))
                print('Saving in the root directory.')
                try:
                    folder_results = 'train{}_{}_{}_{}'.format(init_train_time.strftime("%Y%m%d-%H%M%S"), self.decoder_input, self.strfdelta(train_time, "{d}d-{h}h-{m}m-{s}s"), loss)
                except:
                    folder_results = "Default_saves"
                    os.mkdir(folder_results)
            else:
                print('Successfully created the directory {}'.format(folder_results))
            df = pd.DataFrame(self.info_res, columns= ["ID", "Error"])
            df.to_csv(folder_results+"/Errores.csv")
            self.save_imgs(pred, folder_results, self.rows, self.cols, self.channels, begin= 0)
        except:
            print("The models generation went wrong.")
            shutil.rmtree(self.temporal)
            raise
        shutil.rmtree(self.temporal)

    def save_imgs(self, data, folder, rows, cols, channels= 1, color_map='Greys', begin= -10, end= -1):
        if end == -1:
            save = data[begin:]
        else:
            save = data[begin:end]
        for i in range(len(save)):
            if channels == 1:
                img.imsave('{}/{}.png'.format(folder, i), save[i].reshape(rows, cols), cmap= color_map)
            else:
                img.imsave('{}/{}.png'.format(folder, i), save[i].reshape(rows, cols, channels), cmap= color_map)

    async def start_rabbitmq(self):
        self.rabbitmq = RabbitMQ(self.host, self.user, self.password, self.port)
        await self.queues_startup()
        return await self.rabbitmq.listen(self.queue_results, self._on_results)
    
    async def queues_startup(self): 
        consumers = await self.rabbitmq.get_consumers(self.queue_publish)
        #consumers = await rabbitmq_client.get_consumers('models')
        print("Consumers: {}".format(consumers))
        for i in range(0, consumers + 1):
            self.generate_request()
            self.actual_train += 1

    def generate_request(self):
        #print("VOY!!", y_train)
        x_train, y_train, x_val, y_val = self.generate_dataset_index(self.actual_train)
        #x_val, y_val = self.generate_validation_dataset_index(self.actual_train)
        #x_train, y_train, x_val, y_val = self.x_train[self.actual_train], self.y_train[self.actual_train], self.x_val[self.actual_train], self.y_val[self.actual_train]
        x_test, y_test = self.generate_test_dataset_index(self.actual_train)
        print(x_train.shape)
        print(x_val.shape)
        print(x_test.shape)
        print(y_test.shape)
        #print(x_test)
        #print(y_test)
        #print(self.x_train[0])
        #print(self.y_train[0])
        dictionary = {
            'train_part': self.actual_train,
            'x_train_data': x_train.tolist(),
            'y_train_data': y_train.tolist(),
            'x_val_data': x_val.tolist(),
            'y_val_data': y_val.tolist(),
            'x_test_data': x_test.tolist(),
            'y_test_data': y_test.tolist(),
            'temp_file': self.temporal,
            'window_size': self.window_size,
            'prediction_size': self.predictions_size
        }
        self.rabbitmq.publish(self.queue_publish, dictionary)
        #print("Sended: {}".format(dictionary))

    def generate_dataset_index(self, i):
        x_train_part = np.array([])
        y_train_part = np.array([])
        tam = len(self.train_matrix[i]) - self.window_size
        for j in range(tam):
            x_train_part = np.append(x_train_part, self.train_matrix[i][j : j + self.window_size])
            y_train_part = np.append(y_train_part, self.train_matrix[i][j + self.window_size])
        x_train_part = x_train_part.reshape(tam, 1, self.window_size)
        y_train_part = y_train_part.reshape(tam, 1, 1)
        x_validation = x_train_part[int(x_train_part.shape[0] * self.validation_size) :]
        y_validation = y_train_part[int(y_train_part.shape[0] * self.validation_size) :]
        x_train = x_train_part[: int(x_train_part.shape[0] * self.validation_size)]
        y_train = y_train_part[: int(y_train_part.shape[0] * self.validation_size)]
        return x_train, y_train, x_validation, y_validation

    def generate_validation_dataset_index(self, i):
        x_val_part = np.array([])
        y_val_part = np.array([])
        tam = len(self.validation_matrix[i]) - self.window_size
        for j in range(tam):
            x_val_part = np.append(x_val_part, self.validation_matrix[i][j : j + self.window_size])
            y_val_part = np.append(y_val_part, self.validation_matrix[i][j + self.window_size])
        x_validation = x_val_part.reshape(tam, 1, self.window_size)
        y_validation = y_val_part.reshape(tam, 1, 1)
        return x_validation, y_validation


    def generate_test_dataset_index(self, i):
        x_test_part = np.array([])
        y_test_part = np.array([])
        tam = len(self.test_matrix[i]) - self.window_size
        for j in range(tam):
            x_test_part = np.append(x_test_part, self.test_matrix[i][j : j + self.window_size])
            y_test_part = np.append(y_test_part, self.test_matrix[i][j + self.window_size])
        x_test = x_test_part.reshape(tam, 1, self.window_size)
        y_test = y_test_part.reshape(tam, 1, 1)
        return x_test, y_test

    #optimizar función
    def generate_dataset(self):
        x_train = np.array([])
        y_train = np.array([])
        x_validation = np.array([])
        y_validation = np.array([])
        for i in range(len(self.train_matrix)):
            x_train_part = np.array([])
            y_train_part = np.array([])
            for j in range(len(self.train_matrix[i]) - self.window_size):
                x_train_part = np.append(x_train_part, self.train_matrix[i][j : j + self.window_size])
                y_train_part = np.append(y_train_part, self.train_matrix[i][j + self.window_size])
            x_train_part = x_train_part.reshape((len(self.train_matrix[i]) - self.window_size, 1, self.window_size))
            y_train_part = y_train_part.reshape((len(y_train_part), 1, 1))
            x_train = np.append(x_train, x_train_part)
            y_train = np.append(y_train, y_train_part)
        x_train = x_train.reshape((len(self.train_matrix), len(self.train_matrix[0]) - self.window_size, 1, self.window_size))
        y_train = y_train.reshape((len(self.train_matrix), len(self.train_matrix[0]) - self.window_size, 1, 1))
        x_validation = x_train[:, int(x_train.shape[1] * self.validation_size) :]
        y_validation = y_train[:, int(y_train.shape[1] * self.validation_size) :]
        x_train = x_train[:, : int(x_train.shape[1] * self.validation_size)]
        y_train = y_train[:, : int(y_train.shape[1] * self.validation_size)]
        return x_train, y_train, x_validation, y_validation

    def load_all_dataset(self):
        x = load_imgs(self.data_folder, self.names_file, self.rows, self.cols, self.channels)
        return x

    def load_dataset(self):
        x = load_imgs(self.data_folder, self.names_file, self.rows, self.cols, self.channels)
        #print("Pixels Color", x[-1][69,36])
        #img.imsave('ResDrought/org_grays.png', x[-1].reshape(self.rows, self.cols), cmap='Greys')
        #x = to_monochromatic(x,15)
        train_len = int(len(x) * self.train_size)
        train = x[:train_len]
        test = x[train_len:]
        return train, test

    def get_validation_data(self, train, train_percent= 0.8):
        split_cant = int(len(train) * train_percent)
        train_data = train[: split_cant]
        val_data = train[split_cant :]
        print('original_size: {}. split_cant: {}, train_size: {}, val_size: {}'.format(len(train), split_cant, len(train_data), len(val_data)))
        return train_data, val_data

    def load_and_prepare_all_data(self):
        x = self.load_all_dataset()
        x = x.astype('float32')/255
        x = x.reshape(len(x), self.rows, self.cols, self.channels)
        print('Data shape: {}'.format(x.shape))
        return x
    
    def load_and_prepare_data(self):
        x_train, x_test = self.load_dataset()
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255
        x_train = x_train.reshape(len(x_train), self.rows, self.cols, self.channels)
        x_test = x_test.reshape(len(x_test), self.rows, self.cols, self.channels)
        print('Train data shape: {}'.format(x_train.shape))
        print('Test data shape: {}'.format(x_test.shape))
        return x_train, x_test
    
    def get_model_shape(self, model, layer_i= -1):
        layer = model.layers[layer_i]
        layer_shape = layer.output_shape[1:]
        return layer_shape

    def get_encoder(self, model, input_dim):
        encoder = input_dim
        cant = int(len(model.layers)/2) - 1
        for i in range(cant):
            encoder = model.layers[i+1](encoder)
        encoder = tf.keras.Model(input_dim, encoder)
        return encoder
    
    def get_decoder(self, model):
        cant = int(len(model.layers)/2)
        self.decoder_input = self.get_model_shape(model, cant)
        print("Actual model shape: {} con {} y {}".format(self.decoder_input, model, cant))
        input_shape = tf.keras.layers.Input(shape= self.decoder_input)
        decoder = input_shape
        for i in reversed(range(cant)):
            decoder = model.layers[-(i+1)](decoder)
        decoder = tf.keras.Model(input_shape, decoder)
        return decoder

    def construct_autoencoder(self, input_shape: tuple):
        self.input = tf.keras.layers.Input(input_shape)
        layers = self.input
        #Encoder layers
        for CMPBlock in self.CMPBlocks:
            layers = tf.keras.layers.Conv2D(CMPBlock[0], CMPBlock[1], activation= CMPBlock[2], padding= 'same')(layers)
            layers = tf.keras.layers.MaxPooling2D(CMPBlock[3])(layers)
        #Decoder layers
        for CMPBlock in reversed(self.CMPBlocks):
            layers = tf.keras.layers.Conv2D(CMPBlock[0], CMPBlock[1], activation= CMPBlock[2], padding= 'same')(layers)
            layers = tf.keras.layers.UpSampling2D(CMPBlock[3])(layers)
        layers = tf.keras.layers.Conv2D(input_shape[2], (3,3), activation= 'relu', padding= 'same')(layers)
        autoencoder = tf.keras.Model(self.input, layers)
        #print(autoencoder.summary())
        encoder = self.get_encoder(autoencoder, self.input)
        #print(tf.keras.Model(self.input, encoder).summary())
        decoder = self.get_decoder(autoencoder)
        return autoencoder, encoder, decoder

    def transform_info(self, encoder, decoder_input):
        try:
            train_encoded_imgs = encoder.predict(self.x_train)
            #validation_encoded_imgs = encoder.predict(self.x_validation)
            test_encoded_imgs = encoder.predict(self.x_test)
            print("Train encoded shape: {}".format(train_encoded_imgs.shape))
            tam_imgs = decoder_input[0] * decoder_input[1] * decoder_input[2]
            train_matrix = self.create_matrices(train_encoded_imgs, tam_imgs)
            #validation_matrix = self.create_matrices(validation_encoded_imgs, tam_imgs)
            test_matrix = self.create_matrices(test_encoded_imgs, tam_imgs)
            print("Train matrix shape: {}".format(train_matrix.shape))
            #print("Validation matrix shape: {}".format(validation_matrix.shape))
            print("Test matrix shape: {}".format(test_matrix.shape))
        except:
            print("Something went wrong trying to process the data info.")
            raise
        #return train_matrix, validation_matrix, test_matrix
        return train_matrix, test_matrix

    def transform_all_info(self, encoder, decoder_input):
        try:
            encoded_images= encoder.predict(self.x)
            print("Train encoded shape: {}".format(encoded_images.shape))
            tam_imgs = decoder_input[0] * decoder_input[2] * decoder_input[2]
            matrix = self.create_matrices(encoded_images, tam_imgs)
            print("Matrix shape: {}".format(encoded_images.shape))
        except:
            print("Something went wrong trying to process the data info.")
            raise
        return matrix

    def create_matrices(self, data, tam):
        matrix = np.array([])
        for i in data:
            matrix = np.append(matrix, i.reshape(1, tam))
        matrix = matrix.reshape(len(data), tam)
        matrix = matrix.transpose()
        return matrix

    def save_info_in_disk(self, autoencoder, encoder, decoder, train, test):
        decoder.save('decoder.h5'.format(self.save_path))
        encoder.save('encoder.h5'.format(self.save_path))
        autoencoder.save('autoencoder.h5'.format(self.save_path))
        np.save('{}/train_data.npy'.format(self.save_path), train)
        np.save('{}/test_data.npy'.format(self.save_path), test)

    def save_info_in_disk_no_autoencoder(self, train, test):
        np.save('{}/train_data.npy'.format(self.save_path), train)
        np.save('{}/test_data.npy'.format(self.save_path), test)

    def checkChanges(self, img, number = 0):
        changes = []
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] != number:
                    changes.append((i,j))
        return changes

    def checkAllChanges(self, imgs, number = 0):
        changes = []
        for img in imgs:
            changes.extend(self.checkChanges(img, number))
        changes = set(changes)
        changes = list(changes)
        changes.sort()
        return np.array(changes)

    def get_images_info(self, imgs, points):
        data = [np.array([])]* len(imgs)
        print(len(data))
        for point in points:
            for i in range(len(data)):
                val = imgs[i][point[0]][point[1]]
                data[i] = np.append(data[i],val)
                #print(len(data[0]))
        return np.array(data)
    
    def returnImage(self, data, points):
        forecast = [np.array([0])]*(self.rows*self.cols)
        forecast = np.array(forecast, dtype=np.float32)
        forecast = forecast.reshape(self.rows, self.cols)
        for i in range(len(points)):
            forecast[points[i][0]][points[i][1]] = data[i]
        return forecast

    def _on_results(self, response: dict):
        self.recieved_size += 1
        print("HEY")
        res = json.loads(response)
        print('Receiving results: {}'.format(res))
        err = res['error']
        if err > 0.06:
            print('Model number: {} does not complete the enough score.'.format(res['res']))
            print('val loss: {}'.format(err))
            print('saving state')
            self.info_res.append([res['res'], err]) 
            with open("Models/bad_trainings.txt", 'w') as f:
                for s in self.info_res:
                    f.write(str(s) + '\n')
        self.predictions[res['res']] = res['forecast'] 
        #self.predictions.append(res['forecast'])
        if self.recieved_size >= self.trianing_size:
            print("Breaking the loop")
            self.loop.close()
            raise
        #if self.actual_train >= self.trianing_size:
        if self.actual_train >= self.trianing_size:
            print("Waiting for the models...")
        else:
            self.generate_request()
            self.actual_train += 1
        self.count_save -= 1
        if self.count_save <= 0:
            print("SAVING STATE, ACTUAL {}".format(self.recieved_size))
            self.count_save = 10
            with open("Models/predictions.txt", 'w') as f:
                for s in self.predictions:
                    f.write(str(s) + '\n')
            #np.save("Models/predictions.npy", np.array(self.predictions))
        gc.collect()

    def strfdelta(self, tdelta, fmt):
        d = {'d': tdelta.days}
        d['h'], rem = divmod(tdelta.seconds, 3600)
        d['m'], d['s'] = divmod(rem, 60)
        return fmt.format(**d)
