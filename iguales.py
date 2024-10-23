import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import app.common.load_imgs as li

def load_and_prepare_all_data(rows= 120, cols= 360, channels= 1):
    x = li.load_imgs("app/datasets/DroughtDatasetMainland", "NamesDroughtDataset.csv", rows, cols, img_type=".png")
    x = x.astype('float32')
    x = x.reshape(len(x), rows, cols, channels)
    print('Data shape: {}'.format(x.shape))
    return x

#x = load_and_prepare_all_data(260, 640)
x = load_and_prepare_all_data()

print(x.shape)
for i in range(len(x)-1):
    if (x[i] == x[i+1]).all():
        print("Aqui uno igual, posición: {}".format(i))