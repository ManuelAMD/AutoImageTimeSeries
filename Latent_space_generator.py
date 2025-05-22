import json
from load_imgs import load_imgs, to_monochromatic
import tensorflow as tf
import matplotlib as mat
import matplotlib.pyplot as plt

class LatentSpaceGenerator():
    def __init__(self, json_file= 'autoencoderConfig.json'):
        f = open(json_file, "r")
        self.parameters_data = json.loads(f.read())
        self.rows = self.parameters_data['rows']
        self.cols = self.parameters_data['cols']
        self.channels = self.parameters_data['channels']
        print("({}, {}, {})".format(self.rows, self.cols, self.channels))
        self.train_size = self.parameters_data['train_size']
        self.validation_size = self.parameters_data['validation_size']
        self.epochs = self.parameters_data['epochs']
        self.batch_size = self.parameters_data['batch_size']
        self.window_size = self.parameters_data['window_size'] 
        self.optimizer = self.parameters_data['optimizer']
        self.early_stopping_patience = self.parameters_data['early_stopping_patience']
        self.CMPBlocks = self.parameters_data['autoencoder_CMPBlocks']
        for cmpblock in self.CMPBlocks:
            cmpblock[1] = tuple(cmpblock[1])
            cmpblock[3] = tuple(cmpblock[3])
        print("{}, {}, {}".format(self.optimizer, self.early_stopping_patience, self.CMPBlocks))
        self.save_path = self.parameters_data['folder_models_save']

    def tensorflow_definition(self):
        config = tf.compat.v1.ConfigProto(gpu_options= tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction= 0.8))
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config = config)
        tf.compat.v1.keras.backend.set_session(session)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        return config, session, options
    
    def part_data(self, data):
        x_train = data[:int(len(data)*self.train_size)]
        x_test = data[int(len(data)*self.train_size):]
        x_validation = x_train[int(len(x_train)*self.validation_size):]
        x_train = x_train[:int(len(x_train)*self.validation_size)]
        return x_train, x_test, x_validation
    
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

    def get_latent_space(self, data):
        x_train, x_test, x_validation = self.part_data(data)
        config, session, options = self.tensorflow_definition()
        dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
        dataset = dataset.with_options(options)
        dataset_val = tf.data.Dataset.from_tensor_slices((x_validation, x_validation))
        dataset_val = dataset_val.with_options(options)
        strategy = tf.distribute.MirroredStrategy()
        self.gpu_cant = strategy.num_replicas_in_sync
        self.batch_size = self.batch_size * self.gpu_cant
        dataset = dataset.batch(self.batch_size)
        dataset_val = dataset_val.batch(self.batch_size)
        with strategy.scope():
            autoencoder, encoder, decoder = self.construct_autoencoder((self.rows, self.cols, self.channels))
            print(autoencoder.summary())
            es = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', mode= 'min', patience= self.early_stopping_patience, restore_best_weights= True)
            autoencoder.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy')
            print("Initializing autoencoder training...")
            history = autoencoder.fit(dataset, epochs= self.epochs, validation_data=dataset_val, shuffle= True, verbose= 1, callbacks= [es])
            print(history)
            loss = autoencoder.evaluate(x_test, x_test, verbose= 0)
            print("Autoencoder loss: {}".format(loss))
            print("Successfully construct autoencoder!")
            encoded_img = encoder.predict(x_test[-1].reshape(1,self.rows, self.cols))
            decoded_img = decoder.predict(encoded_img.reshape(1, self.decoder_input[0], self.decoder_input[1], self.decoder_input[2]))
            autoencoder_img = autoencoder.predict(x_test[-1].reshape(1,self.rows, self.cols))
            new_data = encoder.predict(data)

        print(encoded_img.shape)
        print(decoded_img.shape)
        print(autoencoder_img.shape)
        encoded_img = encoded_img[0,:,:,0].reshape(encoded_img.shape[1],encoded_img.shape[2])
        decoded_img = decoded_img.reshape(decoded_img.shape[1],decoded_img.shape[2])
        autoencoder_img = autoencoder_img.reshape(autoencoder_img.shape[1],autoencoder_img.shape[2])

        fig = plt.figure(figsize=(10,7))
        r = 1
        c = 3
        fig.add_subplot(r, c, 1)
        plt.imshow(encoded_img, cmap='gray')
        plt.axis('off')
        plt.title('Original')
        fig.add_subplot(r, c, 2)
        plt.imshow(decoded_img, cmap='gray')
        plt.axis('off')
        plt.title('Pronóstico')
        fig.add_subplot(r, c, 3)
        plt.imshow(autoencoder_img, cmap='gray')
        plt.axis('off')
        plt.title('Naive')

        plt.show()


        return new_data
