import os
from PIL import Image
import numpy as np

#names_data = os.listdir ('DroughtDataset')
names_test = os.listdir ('ResDrought/modelo-naive_(0,0,0)_0d-0h-0m-0s_0.0')

"""for names in names_data:
    foo = Image.open('DroughtDataset/{}'.format(names))
    crop_rectangle = (240, 318, 600, 440)
    foo = foo.crop(crop_rectangle)

    foo.save('DroughtDatasetSelect/{}'.format(names))"""

for names in names_test:
    foo = Image.open('ResDrought/modelo-naive_(0,0,0)_0d-0h-0m-0s_0.0/{}'.format(names))
    crop_rectangle = (240, 318, 600, 440)
    foo = foo.crop(crop_rectangle)


    foo.save('ResDrought/modelo-naive-select_(0,0,0)_0d-0h-0m-0s_0.0/{}'.format(names))