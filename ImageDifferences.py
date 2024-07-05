import numpy as np
from PIL import Image
import app.common.load_imgs as li
import cv2

naive = []
forecast = []
original = []

for i in range(4):
    naive.append(cv2.imread("GeneratedImageComparation/Naive_t+{}.png".format(i)))
    original.append(cv2.imread("GeneratedImageComparation/Original_t+{}.png".format(i)))
    forecast.append(cv2.imread("GeneratedImageComparation/Pronostico_t+{}.png".format(i)))
    
#print(naive)
#print(original)
#print(forecast)

diff = 255 - cv2.absdiff(original[0], forecast[0])

res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = forecast[0]
result[mask != 0] = [0,0,255]

cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceForecast_t+1.png", result)

diff = 255 - cv2.absdiff(original[1], forecast[1])
res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = forecast[1]
result[mask != 0] = [0,0,255]
cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceForecast_t+2.png", result)

diff = 255 - cv2.absdiff(original[2], forecast[2])
res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = forecast[2]
result[mask != 0] = [0,0,255]
cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceForecast_t+3.png", result)

diff = 255 - cv2.absdiff(original[3], forecast[3])
res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = forecast[3]
result[mask != 0] = [0,0,255]
cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceForecast_t+4.png", result)

#########################

diff = 255 - cv2.absdiff(original[0], naive[0])

res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = naive[0]
result[mask != 0] = [0,0,255]

cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceNaive_t+1.png", result)

diff = 255 - cv2.absdiff(original[1], naive[1])
res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = naive[1]
result[mask != 0] = [0,0,255]
cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceNaive_t+2.png", result)

diff = 255 - cv2.absdiff(original[2], naive[2])
res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = naive[2]
result[mask != 0] = [0,0,255]
cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceNaive_t+3.png", result)

diff = 255 - cv2.absdiff(original[3], naive[3])
res = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
diff[mask != 0] = [0,0,255]

result = naive[3]
result[mask != 0] = [0,0,255]
cv2.imshow('diff', result)
cv2.waitKey()
cv2.imwrite("GeneratedImageComparation/DifferenceNaive_t+4.png", result)