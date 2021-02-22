# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:47:14 2019

@author: youss
"""



from PIL import Image
import numpy as np
import sys
import os
import csv

#Useful function
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList('New folder')

for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    
    print(value.tolist())
    #value=str(value)
    str1 = ''.join(str(e) + ' ' for e in value.tolist())
    row=[str1, '0']
    
    file = open('img_pixels.csv', 'a')
    fields = ('image', 'label')
    wr = csv.writer(file, lineterminator = '\n')
    wr.writerow(row)
    
    
          

