import tensorflow as tf
from tensorflow import keras
from app.common.model_communication import *
from app.dataset_files.dataset import *
from app.common.search_space import *
import time
import logging

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
        if self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES:
            return self.build_image_time_series_model(self.model_params, input_shape, class_count)
        
    def is_model_valid(self) -> bool:
        is_valid = True
        try:
            strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
            with strategy.scope():
                input_shape = self.dataset.get_input_shape()
                class_count = self.dataset.get_classes_count()
                self.build_model(input_shape, class_count)
        except ValueError as e:
            logging.warning(e)
            is_valid = False
        tf.keras.backend.clear_session()
        return is_valid
        
    def build_image_time_series_model(self, model_parameters: ImageTimeSeriesArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
        start_time = int(round(time.time() * 1000))
