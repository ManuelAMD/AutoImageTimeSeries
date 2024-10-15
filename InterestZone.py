import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import app.common.load_imgs as li

def load_and_prepare_all_data(rows= 122, cols= 360, channels= 1):
    x = li.load_imgs("app/datasets/DroughtDatasetSelect", "NamesDroughtDataset.csv", rows, cols, img_type="")
    x = x.astype('float32')
    x = x.reshape(len(x), rows, cols, channels)
    print('Data shape: {}'.format(x.shape))
    return x

#x = load_and_prepare_all_data(260, 640)
x = load_and_prepare_all_data()

x = x.reshape(x.shape[0:-1])
#x = x.astype("int32")
print(x.shape)
print(x[0,0,0])
print(x[1000].max())
res = np.zeros((x.shape[1],x.shape[2]), dtype=x.dtype)
print(res)
print(res.shape)


for i in x:
    res += i
print(res)
result = np.where(res > 0, 1, 0)
result = result.astype("uint8")
print(result)

#x += 30

plt.imshow(x[0], cmap="gray")
plt.show()

new_x = np.where(x == 0, 255, x)

plt.imshow(new_x[0], cmap="gray")
plt.show()


#plt.imshow(result, cmap="gray")
#plt.show()

aux = new_x[0]

#print(aux.shape, result.shape)

#img = cv2.bitwise_and(aux, aux, mask= result)

#plt.imshow(img, cmap="gray")
#plt.show()

names = li.get_names("NamesDroughtDataset.csv")
index = 0

new_array = np.array([])
for i in new_x:
    img_new = cv2.bitwise_and(i, i, mask= result)
    #img_new = img_new.reshape(img_new.shape[0], img_new.shape[1], 1)
    new_array = np.append(new_array, img_new)
    im = Image.fromarray(img_new)
    im = im.convert("RGB")
    #im.save("app/datasets/SPIReescaleMask/{}.png".format(names[index]))
    index += 1

new_array = new_array.reshape(x.shape)
print(new_array.shape)

np.save("Models/DroughtDatasetMask", new_array)