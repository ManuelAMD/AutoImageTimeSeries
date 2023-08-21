import abc
import numpy as np
import json
from keras.preprocessing.image import ImageDataGenerator
import cv2

#Abstract class.
class Dataset(abc.ABC):
    @abc.abstractclassmethod
    def load(self):
        pass

    @abc.abstractclassmethod
    def get_train_data(self):
        pass

    @abc.abstractclassmethod
    def get_validation_data(self):
        pass

    @abc.abstractclassmethod
    def get_test_data(self):
        pass

    @abc.abstractclassmethod
    def get_input_shape(self) -> tuple:
        pass

    @abc.abstractclassmethod
    def get_tag(self) -> str:
        pass

class ImageTimeSeriesDataset(Dataset):
    def __init__ (self, dataset_name: str, shape: tuple, batch_size=16, test_split= 0.3, validation_split= 0.2, color_mode= 1):
        self.dataset_name = dataset_name
        self.test_split_float = test_split
        self.validation_split_float = validation_split
        self.batch_size = batch_size
        self.shape = shape
        self.color_mode = color_mode

    def load(self, init_route= None):
        try:
            train_split_float = np.float(1.0 - self.validation_split_float)
            val_split_percent = int(self.validation_split_float * 100)
            train_split_percent = int(train_split_float * 100)
            if init_route == None:
                route = '../AutoImageTimeSeries/app/datasets/' + self.dataset_name
            else:
                route = init_route + 'app/datasets/' + self.dataset_name
            with open(route + 'info.json') as jsonfile:
                info = json.load(jsonfile)
            color = 'grayscale'
            if self.color_mode == 1:
                preprocessing_func = self.mono_func
            elif self.color_mode == 2:
                preprocessing_func = self.categorical_func
            else:
                preprocessing_func = None
                color = 'rgb'
            datagen_train = ImageDataGenerator(rescale=1./255, data_format='channels_last', preprocessing_function=preprocessing_func, validation_split=self.validation_split_float)
            datagen_test = ImageDataGenerator(rescale=1./255, data_format='channels_last', preprocessing_function=preprocessing_func)
            self.train_original = datagen_train.flow_from_directory(route+'/Train/', batch_size=self.batch_size, target_size=(self.shape[0], self.shape[1]), color_mode=color, class_mode='input')
            self.validation = datagen_train.flow_from_directory(route+'/Train/', batch_size=self.batch_size, target_size=(self.shape[0], self.shape[1]), color_mode=color, class_mode='input',subset='validation')
            self.test = datagen_test.flow_from_directory(route+'/Test/', batch_size=self.batch_size, target_size=(self.shape[0], self.shape[1]), color_mode=color, class_mode='input')
            print("HEYYYY LISTENNNN!!!!", self.train_original.data_format)
        except:
            if init_route == None:
                path = '../AutoImageTimeSeries/app/datasets/'+self.dataset_name
    
    def mono_func(self, image):
        x_mono = []
        for i in image:
            (thres, monoImg) = cv2.threshold(i, 15, 255, cv2.THRESH_BINARY)
            x_mono.append(monoImg)
        x_mono = np.array(x_mono)
        return x_mono
    
    def categorical_func(self, image):
        return image

    def get_train_data(self):
        pass

    def get_validation_data(self):
        pass

    def get_test_data(self):
        pass

    def get_input_shape(self) -> tuple:
        self.shape

    def get_tag(self):
        return self.dataset_name