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

class Preprocessing:
    def __init__(self, original_data_path: str, result_data_path: str):
        self.org_path = original_data_path
        self.res_path = result_data_path
        self.data = None
        self.data_dirs = os.listdir(self.org_path)
    
    def load_data(self, rows: int, cols: int, names_file_path: str, color= "grayscale"):
        if color == 'grayscale':
            channels = 1
            self.data = li.load_imgs(self.org_path, names_file_path, rows, cols)
        else:
            channels = 3
            self.data = li.load_imgs(self.org_path, names_file_path, rows, cols, 3, color_mode= 'rgb')
        self.data = self.data.astype('float32')
        self.data = self.data.reshape(len(self.data), rows, cols, channels)
        print('Data Shape: {}'.format(self.data.shape))
        return True

    def crop_images(self, initial_x: int, initial_y: int, finish_x: int, finish_y: int):
        if self.data == None:
            print("Primero carga los datos de las imágenes con la función interna load_data")
            return
        cropSize = (initial_x, initial_y, finish_x, finish_y)
        for i in range(len(self.data)):
            item = self.data_dirs[i]
            if os.path.isfile(self.org_path + item):
                im = self.data[i]
                f, e = os.path.splitext(self.res_path + item)
                imageCrop = im.crop(cropSize)
                imageCrop.save(f + '.png', 'PNG')
        #for item in self.data_dirs:
        #    if os.path.isfile(self.org_path + item):
        #        im = Image.open(self.org_path + item)
        #        f, e = os.path.splitext(self.res_path + item)
        #        imageCrop = im.crop(cropSize)
        #        imageCrop.save(f + '.png', 'PNG')

        


if __name__ == '__main__':
    names_file_path = 'NamesDroughtDataset.csv'
    p = Preprocessing('app/datasets/DroughtDataset', 'app/datasets/DroughtDatasetMainland')
    p.load_data(480,640, names_file_path)
    p.crop_images(240, 318, 600, 438)
