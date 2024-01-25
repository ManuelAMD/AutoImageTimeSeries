import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
from keras import layers
from app.common.model_communication import *
from app.dataset_files.dataset import *
from app.common.search_space import *
import time
import logging

physical_devices = tf.config.list_physical_devices('GPU')
print("_______ {} _______".format(physical_devices))

class Model:
    def __init__(self, model_training_request: ModelTrainingRequest, dataset: Dataset):
        self.id = model_training_request.id
        self.experiment_id = model_training_request.experiment_id
        self.training_type = model_training_request.training_type
        self.search_space_type = model_training_request.search_space_type
        self.model_params = model_training_request.architecture
        self.epochs = model_training_request.epochs
        self.early_stopping_patience = model_training_request.early_stopping_patience
        self.is_partial_training = model_training_request.is_partial_training
        self.model: tf.Keras.Model
        self.dataset:Dataset = dataset

    def build_model(self, input_shape: tuple, class_count: int):
        if self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES.value:
            return self.build_image_time_series_model(self.model_params, input_shape, class_count)
        
    def is_model_valid(self) -> bool:
        is_valid = True
        try:
            strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
            with strategy.scope():
                input_shape = self.dataset.get_input_shape()
                class_count = self.dataset.get_classes_count()
                if self.build_model(input_shape, class_count) == None:
                    is_valid = False
        except ValueError as e:
            logging.warning(e)
            is_valid = False
        tf.keras.backend.clear_session()
        return is_valid
    
    def build_and_train_cpu(self):
        print("Training with CPU")
        try:
            strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
            with strategy.scope():
                self.build_and_train()
        except ValueError as e:
            logging.warning(e)

    def build_and_train(self) -> float:
        input_shape = self.dataset.get_input_shape()
        class_count = self.dataset.get_classes_count()
        model = self.build_model(input_shape, class_count)
        x_train, y_train = self.dataset.get_train_data()
        #training_steps = self.dataset.get_training_steps()
        x_validation, y_validation = self.dataset.get_validation_data()
        #validation_steps = self.dataset.get_validation_steps()
        x_test, y_validation = self.dataset.get_test_data()
        def scheduler(epoch):
            if epoch < 10:
                return 0.001
            else:
                return 0.001 * tf.math.exp(0.01 * (10 - epoch))
        scheduler_callback = keras.callbacks.LearningRateScheduler(scheduler)
        early_stopping: keras.callbacks.EarlyStopping = None
        if self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES.value:
            monitor_exploration_training = 'val_loss'
            monitor_deep_training = 'val_loss'
        
        if self.is_partial_training:
            early_stopping = keras.callbacks.EarlyStopping(monitor= monitor_exploration_training, patience= self.early_stopping_patience, verbose= 1, restore_best_weights= True)
        else:
            early_stopping = keras.callbacks.EarlyStopping(monitor= monitor_deep_training, patience= self.early_stopping_patience, verbose= 1, restore_best_weights= True)
        model_stage = "exp" if self.is_partial_training else 'hof'
        log_dir = "logs/{}/{}-{}".format(self.experiment_id, model_stage, self.id)
        tensorboard = keras.callbacks.TensorBoard(log_dir= log_dir, histogram_freq= 1)
        callbacks = [early_stopping, tensorboard, scheduler_callback]
        total_weights = np.sum([np.prod(v.get_shape().as_list()) for v in model.variables])
        print("Total model weights {}".format(total_weights))
        print("****** Beggining the training ******")
        history = model.fit(
            x_train, y_train,
            epochs = self.epochs,
            #steps_per_epoch = training_steps,
            validation_data = (x_validation, y_validation),
            #validation_steps = validation_steps
            verbose = 1
        )
        print("****** Training end ******")
        did_finish_epochs = self._did_finish_epochs(history, self.epochs)
        if self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES.value:
            training_val = model.evaluate(x_test, y_validation, batch_size = 2, verbose = 0)
            print("Model loss test: {}".format(training_val))
        tf.keras.backend.clear_session()
        return training_val, did_finish_epochs
    
    @staticmethod
    def _did_finish_epochs(history, requested_epochs: int) -> bool:
        h = history.history
        trained_epochs = len(h['loss'])
        return requested_epochs == trained_epochs
    
    
    def build_image_time_series_model(self, model_parameters: ImageTimeSeriesArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
        start_time = int(round(time.time() * 1000))
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Compute dtype: {}".format(policy.compute_dtype))
        print("Variable dtype: {}".format(policy.variable_dtype))
        model = keras.Sequential()
        model.add(layers.Input(input_shape))
        if model_parameters.base_architecture == 'conv_lstm_2d':
            self._add_conv_lstm_architecture(model, model_parameters)
        model.add(layers.Conv2D(input_shape[-1], (3,3), padding= 'same', activation= 'sigmoid'))
        model.compile(loss= 'binary_crossentropy', optimizer= 'Adam')
        elapsep_seconds = int(round(time.time() * 1000)) - start_time
        print("** Model building took {} (miliseconds) **".format(elapsep_seconds))
        print(model.summary())
        return model

    def _add_conv_lstm_architecture(self, model: keras.Model, model_parameters = ImageTimeSeriesArchitectureParameters, activation= 'relu', padding= 'same'):
        max_layers = model_parameters.conv_lstm_layers_n
        for n in range(0, max_layers-1):
            filters = model_parameters.conv_lstm_filters[n]
            conv_size = model_parameters.conv_lstm_filters_sizes[n]
            model.add(layers.ConvLSTM2D(filters, conv_size, padding= padding, return_sequences= True, activation= activation))
            norm_layer = model_parameters.normalization_layers[n]
            if norm_layer:
                model.add(layers.BatchNormalization())
        filters = model_parameters.conv_lstm_filters[max_layers-1]
        conv_size = model_parameters.conv_lstm_filters_sizes[max_layers-1]
        model.add(layers.ConvLSTM2D(filters, conv_size, padding= padding, activation= activation))
        if model_parameters.normalization_layers[max_layers-1]:
            model.add(layers.BatchNormalization())
        
        for n in range(0, model_parameters.cnn_layers_n):
            filters = model_parameters.cnn_filters[n]
            conv_size = model_parameters.cnn_filters_size[n]
            model.add(layers.Conv2D(filters, conv_size, padding= padding, activation= activation))
