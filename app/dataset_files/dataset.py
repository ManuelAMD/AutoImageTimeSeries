import abc
import os
import numpy as np
import json
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
from app.common.load_imgs import *

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
    def get_training_steps(self) -> int:
        pass

    @abc.abstractclassmethod
    def get_validation_steps(self) -> int:
        pass

    @abc.abstractclassmethod
    def get_testing_steps(self) -> int:
        pass

    @abc.abstractclassmethod
    def get_input_shape(self) -> tuple:
        pass

    @abc.abstractclassmethod
    def get_classes_count(self) -> int:
        pass

    @abc.abstractclassmethod
    def get_tag(self) -> str:
        pass

class ImageTimeSeriesDataset(Dataset):
    def __init__ (self, dataset_name: str, shape: tuple, data_size=1, batch_size=16, test_split= 0.3, validation_split= 0.2, color_mode= 1):
        self.dataset_name = dataset_name
        self.test_split_float = test_split
        self.validation_split_float = validation_split
        self.batch_size = batch_size
        self.shape = shape
        self.data_size = data_size
        self.color_mode = color_mode

    def load(self, init_route= None):
        try:
            train_split_float = np.float16(1.0 - self.validation_split_float)
            val_split_percent = int(self.validation_split_float * 100)
            train_split_percent = int(train_split_float * 100)
            test_split_percent = int(self.test_split_float * 100)
            if init_route == None:
                route = '../AutoImageTimeSeries/app/datasets/' + self.dataset_name
            else:
                route = init_route + 'app/datasets/' + self.dataset_name
            print("** LA RUTA DE LOS DATOS ES {} **".format(route))
            arr = os.listdir(route)
            arr.sort()
            arr.remove('info.json')
            all_data = load_imgs_names(route, arr, self.shape[1], self.shape[2])
            print(all_data.shape)
            with open(route + '/info.json') as jsonfile:
                info = json.load(jsonfile)
            classes = np.array(info['color_classes'])
            print(type(info['color_classes']))
            all_cubes = self.agroup_window(all_data, self.shape[0]+1)
            print(all_cubes.shape)
            cubes, objectives = self.create_shifted_frames(all_cubes)
            print(cubes.shape, objectives.shape)
            #Change the shape
            self.x_train_original = cubes[:-int(cubes.shape[0] * self.test_split_float)]
            self.y_train_original = objectives[:-int(objectives.shape[0] * self.test_split_float)]
            print(self.x_train_original.shape, self.y_train_original.shape)
            self.x_test = cubes[-int(cubes.shape[0] * self.test_split_float):]
            self.y_test = objectives[-int(objectives.shape[0] * self.test_split_float):]
            print(self.x_test.shape, self.y_test.shape)
            self.x_validation = self.x_train_original[-int(self.x_train_original.shape[0] * self.validation_split_float):]
            self.y_validation = self.y_train_original[-int(self.y_train_original.shape[0] * self.validation_split_float):]
            print(self.x_validation.shape, self.y_validation.shape)
            self.x_train_original = self.x_train_original[:-int(self.x_train_original.shape[0] * self.validation_split_float)]
            self.y_train_original = self.y_train_original[:-int(self.y_train_original.shape[0] * self.validation_split_float)]
            print(self.x_train_original.shape, self.y_train_original.shape)
            #--- Usar los datos directamente, 
            #--- Hacer una función con solo los 7 colores
            #--- Hacer una función al final con las estimaciones a solo 7 colores
            #ImageDataGenerator.flow_from_directory(
            #    route,
            #    
            #)
        except:
            print("!!!!!!!! No se pudieron cargar los datos !!!!!!!")
            raise

    def agroup_window(self, data, window):
        new_data = [data[i:window+i] for i in range(len(data)-window+1)]
        return np.array(new_data)

    def create_shifted_frames(self, data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, data.shape[1]-1, :, :]
        return x, y
    
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
        x_train = ImageTimeSeriesDataset._scale(self.x_train_original)
        y_train = ImageTimeSeriesDataset._scale(self.y_train_original)
        return x_train, y_train

    def get_validation_data(self):
        x_val = ImageTimeSeriesDataset._scale(self.x_validation)
        y_val = ImageTimeSeriesDataset._scale(self.y_validation)
        return x_val, y_val

    def get_test_data(self):
        x_test = ImageTimeSeriesDataset._scale(self.x_test)
        y_test = ImageTimeSeriesDataset._scale(self.y_test)
        return x_test, y_test

    def get_training_steps(self):
        pass

    def get_validation_steps(self):
        pass

    def get_testing_steps(self):
        pass

    def get_input_shape(self) -> tuple:
        return self.shape

    def get_classes_count(self):
        return self.data_size

    def get_tag(self):
        return self.dataset_name
    
    @staticmethod
    def _scale(image: np.ndarray):
        image = image.astype("float32")
        image /= 255
        return image
