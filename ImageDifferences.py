import numpy as np
from PIL import Image
import app.common.load_imgs as li
import cv2

horizon = 12
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1731702460.npy")
#forecasts = np.load("Models/DiferencesNaive6.npy")
#forecasts = np.load("Models/DiferencesForecast9Model_MultiCNN_testing_1740655076.npy")
#original = np.load("Models/DiferencesOriginal9.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1745498089.npy")

#No reduction method
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1749213320.npy")

#Fragmentation experiments
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1745413473.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1745491125.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1745498089.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748950913.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748968962.npy")

#Autoencoder experiments
#\
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1747658342.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1749467520.npy")
#\
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1747739072.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1749903771.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1752835210.npy")
#\
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1747746488.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1749823787.npy")
#\
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1747913078.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748014367.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1749027560.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748329014.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748331917.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748341644.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748596184.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748871566.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1749031061.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1748944410.npy")

#Forecasting strategies  
#Direct
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1753344963.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_.npy")

#MIMO
forecasts = np.load("Models/DiferencesForecast12DroughtDataset_model_testing_1753444629.npy")
#forecasts = np.load("Models/DiferencesForecast12DroughtDataset_model_testing_.npy")

#DirRec
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1753358970.npy")
#forecasts = np.load("Models/DiferencesForecast6DroughtDataset_model_testing_1753370981.npy")

#original = np.load("Models/DiferencesOriginal6.npy")
original = np.load("Models/DiferencesOriginal12.npy")
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
    #diff = 255 - cv2.absdiff(original[i], forecasts[i])
    
    res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    diff[mask != 0] = [0,0,255]

    #result = original[i]
    result = forecasts[i]
    result[mask != 0] = [0,0,255]

    cv2.imshow('diff', result)
    cv2.waitKey()
    #cv2.imwrite("GeneratedImageComparation/DifferenceNoReductionArticulo_Sahir_t+{}.png".format(i), result)
    #cv2.imwrite("GeneratedImageComparation/DifferenceFragmentation6Articulo_Sahir_t+{}.png".format(i), result)
    #cv2.imwrite("GeneratedImageComparation/DifferenceCodification2Articulo_Sahir_t+{}.png".format(i), result)
    #cv2.imwrite("GeneratedImageComparation/DifferenceDirectArticulo_t+{}.png".format(i), result)
    cv2.imwrite("GeneratedImageComparation/DifferenceMIMOArticulo_t+{}.png".format(i), result)
    #cv2.imwrite("GeneratedImageComparation/DifferenceDirRecArticulo_t+{}.png".format(i), result)

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