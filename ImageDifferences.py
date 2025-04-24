import numpy as np
from PIL import Image
import app.common.load_imgs as li
import cv2

horizon = 12
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1731702460.npy")
#forecasts = np.load("Models/DiferencesNaive6.npy")
#forecasts = np.load("Models/DiferencesForecast9Model_MultiCNN_testing_1740655076.npy")
#original = np.load("Models/DiferencesOriginal9.npy")
forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1745498089.npy")
original = np.load("Models/DiferencesOriginal6.npy")
#naive = np.load("Models/DiferencesNaive6.npy")

#naive = naive.reshape(naive.shape[:-1])
original = original.reshape(original.shape[:-1])
forecasts = forecasts.reshape(forecasts.shape[:-1])

forecasts = np.stack((forecasts,)*3, axis=-1)
original = np.stack((original,)*3, axis=-1)
#naive = np.stack((naive,)*3, axis=-1)

print("Datos procesados para comparar imágenes")
#print(naive.shape)
print(original.shape)
print(forecasts.shape)

for i in range(len(original)):
    diff = 255 - cv2.absdiff(forecasts[i], original[i])
    
    res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    diff[mask != 0] = [0,0,255]

    result = original[i]
    result[mask != 0] = [0,0,255]

    cv2.imshow('diff', result)
    cv2.waitKey()
    cv2.imwrite("GeneratedImageComparation/DifferenceFragmentation4Articulo_Sahir_t+{}.png".format(i), result)

"""for i in range(len(original)):
    diff = 255 - cv2.absdiff(naive[i], original[i])
    
    res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    diff[mask != 0] = [0,0,255]

    result = original[i]
    result[mask != 0] = [0,0,255]

    #cv2.imshow('diff', result)
    cv2.waitKey()
    cv2.imwrite("GeneratedImageComparation/DifferenceArticuloNaive_t+{}.png".format(i), result)"""