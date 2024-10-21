#Este archivo sirve para realizar todas las tecnicas de preprocesamiento de los datos para generar un conjunto de mapas metorológicos de mayor calidad.
#La idea es aplicar un conjunto de procesos a las imágenes para que sean mucho más sencillas de procesar por los algoritmos.

#Pasos:
#1.- Cargar mapas en memoria
#2.- En caso de ser necesario, recortar el área de interes
#3.- Cambiar escala de color a grises.
#4.- Realizar categorización de los mapas
#5.- Mascara del área de interés

import os
from PIL import Image
import app.common.load_imgs as li
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from app.common.color_tools import *

class Preprocessing:
    def __init__(self, original_data_path= '', result_data_path= ''):
        self.org_path = original_data_path
        self.res_path = result_data_path
        self.data = None
        if original_data_path == '':
            self.data_dirs = []
        else:
            self.data_dirs = os.listdir(self.org_path)
        self.data_loaded = False
    
    def load_data(self, rows: int, cols: int, names_file_path: str, color= "grayscale"):
        data = []
        if color == 'grayscale':
            channels = 1
            data = li.load_imgs(self.org_path, names_file_path, rows, cols)
            
        else:
            channels = 3
            data = li.load_imgs(self.org_path, names_file_path, rows, cols, 3, color_mode= 'rgb')
        data = data.astype('float32')
        self.data = data.reshape(len(data), rows, cols, channels)
        print('Data Shape: {}'.format(self.data.shape))
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.data_loaded = True
        return True
    
    def load_from_numpy_array(self, path, rows, cols, channels):
        self.data = np.load(path)
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.data_loaded = True
        return True

    #Esta función corta y guarda las imágenes, tener cuidado de no utilizarla junto con las demás
    def crop_images(self, initial_x: int, initial_y: int, finish_x: int, finish_y: int):
        if not self.data_loaded:
            print("Primero carga los datos de las imágenes con la función interna load_data")
            return
        cropSize = (initial_x, initial_y, finish_x, finish_y)
        print(cropSize)
        for item in self.data_dirs:
            if os.path.isfile(self.org_path + '/' + item):
                im = Image.open(self.org_path + '/' + item)
                f, e = os.path.splitext(self.res_path + '/' + item)
                imageCrop = im.crop(cropSize)
                imageCrop.save(f+'.png')

    #Esta función guarda un np array como resultado
    def map_masking(self, save_imgs= False, no_zone= True, display = False):
        x = self.data.reshape(self.data.shape[0:-1])
        print(x[0,0,0])
        print(x[1000].max())
        #Crea un mapa auxiliar para tener la mascara
        aux = np.zeros((x.shape[1], x.shape[2]), dtype = x.dtype)
        print(aux.shape)
        for i in x:
            #Suma los valores de todos los mapas para obtener aquellos que se mueven
            aux += i
        #Donde los pixeles contienen información, son pixeles que cambian en el tiempo
        mascara = np.where(aux > 0, 1, 0)
        mascara = mascara.astype('uint8')
        if display:
            plt.imshow(x[0], cmap="gray")
            plt.show()
            plt.imshow(mascara, cmap="gray")
            plt.show()
        if no_zone:
            #Asignar todos los valores 0 del mapa a 255 para activar
            x = np.where(x == 0, 255, x)
            if display:
                plt.imshow(x[0], cmap="gray")
                plt.show()
        aux = x
        index = 0
        new_array = np.array([])
        for i in x:
            img_new = cv2.bitwise_and(i, i , mask= mascara)
            new_array = np.append(new_array, img_new)
            im = Image.fromarray(img_new)
            im = im.convert('RGB')
            if save_imgs:
                im.save(self.res_path + '/{}.png'.format(self.data_dirs[index]))
                index += 1
        new_array = new_array.reshape(x.shape)
        print(new_array.shape)
        if display:
            plt.imshow(new_array[0], cmap="gray")
            plt.show()
        self.data = new_array
        #np.save(self.res_path + '.npy', new_array)


    def recolor(args):
        data, pallete = args
        res = gray_quantized(data, pallete)
        res = recolor_greys_image(res, pallete)
        return np.array(res)
    
    def agroup_window(data, window):
        new_data = [data[i : window + i] for i in range(len(data) - window + 1)]
        return np.array(new_data)
    
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y

    def create_shifted_frames_2(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, data.shape[1]-1, :, :]
        return x, y
    
    def categorize(self, categories, display = False):

        x = self.data.astype(np.uint8)
        #x = np.load("Models/DroughtDatasetMask.npy").astype(np.uint8)
        if display:
            fig, axes = plt.subplots(2,3, figsize= (10,8))
            choice = np.random.choice(range(len(x)), size= 1)[0]
            for idx, ax in enumerate(axes.flat):
                ax.imshow(np.squeeze(x[choice + idx]), cmap= 'gray')
                ax.set_title("Frame {}".format(idx + 1))
                ax.axis('off')
            plt.show()
        
        print("Categorias originales", categories)
        print("Colores actuales", get_colors(x[1000]))
        args = [(d, categories) for d in x]
        num_cores = multiprocessing.cpu_count()
        print("INICIANDO PROCEDIMIENTO DE CATEGORIZACIÓN")
        with ProcessPoolExecutor(max_workers= num_cores - 2) as pool:
            with tqdm( total = len(x)) as progress:
                futures = []
                for img in args:
                    future = pool.submit(Preprocessing.recolor, img)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
        x = np.array(results)
        print("PROCESO TERMINADO")
        print("Colores resultantes", get_colors(x[1000]))
        self.data = x.astype('float32') / 255

    def save_data_numpy_array(self, path):
        np.save(path, self.data)

    def create_STI_dataset(self, window_size, train_float= 0.7, validation_float= 0.2, save_test= True):
        x = self.data
        x = Preprocessing.agroup_window(x, window_size)
        print(x.shape)
        x_train = x[: int(len(x) * train_float)]
        x_test = x[int(len(x) * train_float) :]
        div_part = (1-validation_float)
        x_validation = x_train[int(len(x_train) * div_part) :]
        x_train = x_train[: int(len(x_train) * div_part)]
        
        x_train = x_train.reshape(len(x_train), window_size, self.rows, self.cols, self.channels)
        x_validation = x_validation.reshape(len(x_validation), window_size, self.rows, self.cols, self.channels)
        x_test = x_test.reshape(len(x_test), window_size, self.rows, self.cols, self.channels)
        print("Forma de datos de entrenamiento: {}".format(x_train.shape))
        print("Forma de datos de validación: {}".format(x_validation.shape))
        print("Forma de datos de pruebas: {}".format(x_test.shape))

        x_train, y_train = Preprocessing.create_shifted_frames_2(x_train)
        x_validation, y_validation = Preprocessing.create_shifted_frames_2(x_validation)
        x_test, y_test = Preprocessing.create_shifted_frames_2(x_test)
        print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
        print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
        print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))

        np.save("Models/x_test_data.npy", x_test)
        np.save("Models/y_test_data.npy", y_test)
        return x_train, y_train, x_validation, y_validation, x_test, y_test
        

if __name__ == '__main__':
    names_file_path = 'NamesDroughtDataset.csv'
    #p = Preprocessing('app/datasets/DroughtDataset', 'app/datasets/DroughtDatasetMainland')
    #p.load_data(480,640, names_file_path)
    #p.crop_images(240, 318, 600, 438)

    p = Preprocessing('app/datasets/DroughtDatasetMainland', 'app/datasets/DroughtDatasetMainland')
    p.load_data(120,360, names_file_path)
    p.map_masking(save_imgs= False, no_zone= True, display= True)
    categories = np.array([0, 35, 70, 119, 177, 220, 255])
    p.categorize(categories, True)
    p.save_data_numpy_array('Models/ProcessedDroughtDataset.npy')
