# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:51:06 2019
@author: youss
"""
# Data Preprocessing Template
# Importing the libraries
#import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')

import tensorflow as tf 
import cv2
import PIL
from PIL import Image , ImageEnhance
import numpy 
import matplotlib.pyplot as plt
import pandas as pd
import xlwt 
from xlwt import Workbook
import sys
import os
import csv

# Importing the dataset
dataset = pd.read_csv('fer2013Kaggle.csv')
X= dataset.iloc[:,1:2].values
y= dataset.iloc[:,:-2].values
#print (X.shape)
#print (y.shape)
#print(y[0][0])
#print(X[0][0])

#Convert My input to 2d numpy array to image
for loop in range (35887):
    mat=numpy.zeros((48,48))
    data =X[loop][0].split()
    count = 0
    for i in range (48):
        for j in range (48):
            pixel = int(data[count])
            mat[i][j]=pixel
            count=count+1
    X[loop][0]=mat
print (X[34987][0])    

img = PIL.Image.fromarray(X[59][0])
img.mode = 'I'
img = img.convert('L')
img.show()

#histogram normalization
for loop in range(35887):
        img = Image.open('dataset/AllData_Normalized/'+str(loop)+'.jpg')
        img_ = numpy. array(img)
        test=numpy.zeros((192,192))
        c=0
        r=0
        for raw in range (2):
            for coulmn in range (2):        
                for i in range (48):
                    for j in range (48):
                        test[i+r][j+c]=img_[i][j][0]
                if(c<=47):
                    c=c+48    
            c=0
            if(r<=47):
                r=r+48        
        cv2.imwrite('newwwAllData-Mirror/'+str(loop)+'.jpg',test)

#mirror the image
img = Image.open('AllData_Normalized/'+str(0)+'.jpg')
img_ = numpy. array(img)        
test=numpy.zeros((192,192))
c=0
r=0
for raw in range (4):
    for coulmn in range (4):        
        for i in range (48):
            for j in range (48):
                test[i+r][j+c]=img_[i][j][0]
        if(c<=143):
            c=c+48    
    c=0
    if(r<=143):
        r=r+48        
cv2.imwrite('AllData-Mirror/'+str(0)+'.jpg',test)

#################################################
#to label all the image  to it's class
for loop in range(35887):
    if y[loop][0] == 6 :
        img = cv2.imread('Alldata-Blackimage/'+str(loop)+'.jpg')
        cv2.imwrite('AllData-Blackimage-Labeled/6/'+str(loop)+'.jpg',img)

#to convert the images to train and test
for loop in range(35887):
    if y[loop][0] == 6 :
        if (loop <= 28664) :
            img = cv2.imread('Alldata-Blackimage/'+str(loop)+'.jpg')
            cv2.imwrite('dataset-Blackimage/train-set/6/'+str(loop)+'.jpg',img)
        else :
             img = cv2.imread('Alldata-Blackimage/'+str(loop)+'.jpg')
             cv2.imwrite('dataset-Blackimage/test-set/6/'+str(loop)+'.jpg',img)
   


    






          
#habd
#convert and save to file TwoByTwoMirrorfer2013 and convert to csv
"""for loop in range(35887):
        img = Image.open('dataset/AllData_Normalized/'+str(loop)+'.jpg')
        img_ = numpy.array(img) 
        test=numpy.zeros((96,96))
        c=0
        r=0
        for raw in range (2):
            for coulmn in range (2):        
                for i in range (48):
                    for j in range (48):
                        test[i+r][j+c]=img_[i][j][0]
                if(c<=47):
                    c=c+48    
            c=0
            if(r<=47):
                r=r+48              
        cv2.imwrite('TwoByTwoAllMirrorData/'+str(loop)+'.jpg',test)
        value = test.flatten()
        str1 = ''.join(str(e) + ' ' for e in value.tolist())
        row=[str1]
        file = open('TwoByTwoMirrorfer2013.csv', 'a')
        fields = ('image')
        wr = csv.writer(file, lineterminator = '\n')
        wr.writerow(row) """
   
"""    
cv2.imwrite(str(34987)+'.jpg',X[34987][0])
    
img = cv2.imread(str(34987)+'.jpg')
img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite('dataset/' + str(34987)+'.jpg',hist_equalization_result)    
   """ 
    
# 192*192 black image with data
"""X_test=X[0][0]
test=numpy.zeros((192,192))
for i in range (48):
    for j in range (48):
        test[i][j]=X_test[i][j]
    
img = PIL.Image.fromarray(test)
img.mode = 'I'
img = img.convert('L')
img.show()"""

#histogram normalization
"""for loop in range(35887):
    if y[loop][0] == 1 :
        img = cv2.imread('Alldata/'+str(loop)+'.jpg')
        img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
        hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite('1/'+str(loop)+'.jpg',hist_equalization_result)"""

# 192*192 black image with data
"""X_test=X[0][0]
test=numpy.zeros((192,192))
for i in range (48):
    for j in range (48):
        test[i][j]=X_test[i][j]
    
img = PIL.Image.fromarray(test)
img.mode = 'I'
img = img.convert('L')
img.show()"""


#mirror the image
"""X_test=X[0][0]
test=numpy.zeros((192,192))
c=0
r=0
for raw in range (4):
    for coulmn in range (4):        
        for i in range (48):
            for j in range (48):
                test[i+r][j+c]=X_test[i][j]
        if(c<=143):
            c=c+48    
    c=0
    if(r<=143):
        r=r+48        
img = PIL.Image.fromarray(test)
img.mode = 'I'
img = img.convert('L')
img.show()"""




#histogram normalization
"""for loop in range(35887):
    if y[loop][0] == 1 :
        img = cv2.imread('Alldata/'+str(loop)+'.jpg')
        img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
        hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite('1/'+str(loop)+'.jpg',hist_equalization_result)"""


#mirror the image
"""
X_test=X[0][0]
test=numpy.zeros((192,192))
i=0
j=0
for z in range (192):
    counti = 0
    countj = 0
    if (i == 47 & counti<=3 ):
        i=0
        j=0
        counti+=1
    for y in range (192):
            if (j == 47 & countj <= 3):
                j=0
                countj+=1
            test[z][y]=X_test[i][j]
            if(j < 47):
                j+=1
    if(i < 47):
        i+=1    
img = PIL.Image.fromarray(test)
img.mode = 'I'
img = img.convert('L')
img.show()
"""
"""#mirror image 
X_test=X[0][0]
test=numpy.zeros((192,192))
z = 0
y = 0
for i  in range (192):
    if (z<192):    
        c=0
        for j  in range (192):
            if (j <= 48  ):
                if(j == 48 & c != 4 & y <192):
                    i=0
                    j=0
                    test[z][y]=X_test[i][j]    
                    
                    c=c+1
                    y=y+1
    z=z+1    
img = PIL.Image.fromarray(test)
img.mode = 'I'
img = img.convert('L')
img.show()"""

#Convert My input to 2d numpy array to image and make it image in its folder
"""
for loop in range (35887):
    mat=numpy.zeros((48,48))
    data =X[loop][0].split()
    count = 0
    if(y[loop][0] == 0):
        for i in range (48):
            for j in range (48):
                pixel = int(data[count])
                mat[i][j]=pixel
                count=count+1
        X[loop][0]=mat        
        img = PIL.Image.fromarray(X[1][0])
        img.mode = 'I'
        img = img.convert('L')
        img.save('AllData/'+ str(loop) +'.jpg') 
 
"""        `

#Convert My input to 2d numpy array to image    
"""
for loop in range (35887):
    mat=numpy.zeros((48,48))
    data =X[loop][0].split()
    count = 0
    for i in range (48):
        for j in range (48):
            pixel = int(data[count])
            mat[i][j]=pixel
            count=count+1
    X[loop][0]=mat        
""" 

#test tensorflow 
"""
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
"""

#ImageEnhance
"""
img = PIL.Image.fromarray(X[1][0])
img.mode = 'I'
img = img.convert('L')
enhancer = ImageEnhance.Color(img)
img = enhancer.enhance(2.0)
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1.0)
enhancer = ImageEnhance.ImageFilter(img)
img = enhancer.enhance(1.0)

img = img.resize((200,200))
img.show()
print (X[1][0])
"""

#resize image
"""img = Image.open("emma.jpg")
mat=mat.resize((200,200));
img.resize ((1500,2000)).save("emma2.jpg")
img.resize ((48,48)).save("emma.jpg")
print (img)"""            
   
#Exel sheet  create 
"""#create woekbook
wb = Workbook()
#add sheet
sheet1=wb.add_sheet('sheet1')
sheet1.write(loop+1,1,pd.DataFrame(X[loop][0]))
sheet1.write(loop+1,0,y[loop][0])
wb.save ('dataset.xls')    """

#one record of the input 
"""X1=dataset.iloc[0:1,1:2].values
print (X1.shape)"""
            
# convert to image
"""imm=numpy.concatenate(X1).astype(None)
print (imm.shape)
np_im = numpy.array(X1[0][0])
print (np_im.shape)
print (X1[0][0])
img = PIL.Image.fromarray(np_im , 'RGB')
img.save("image1.png")"""

#Taking care of missing data
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])"""

# Encoding categorical data 
"""from sklearn.preprocessing import LabelEncoder , OneHotEncoder
#labelEncoder_X = LabelEncoder()
#X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X= oneHotEncoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)"""

# Splitting the dataset into the Training set and Test set
"""#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""