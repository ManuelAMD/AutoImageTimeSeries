#!/usr/bin/python
from PIL import Image
import os, sys

path = "SPIIMAGES/"
destiny = "SPIReescale/"
dirs = os.listdir( path )

def crop_and_resize(cropSize, size):
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(destiny+item)
            imageCrop = im.crop(cropSize)
            imResize = imageCrop.resize(size)
            imResize.save(f + 'resized.png', 'PNG', quality=90)
#100, 270
#3300 , 1620

x1, y1 = 100, 250
x2, y2 = 3300, 1620

#crop_and_resize(x1, y1, (x2 - x1), (y2 - y1))
crop_and_resize((x1, y1, x2, y2), (640,260))