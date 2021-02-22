# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:51:06 2019

@author: youss
"""

# Convolutional Neural Network (Cats vs Dogs)
# Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import tensorflow as tf 
import cv2
import PIL
from PIL import Image , ImageEnhance
import numpy 
import matplotlib.pyplot as plt
import pandas as pd
import xlwt 
from xlwt import Workbook

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('fer2013.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,:-2].values

#load the dataset mirror
path  = 'E:\Projects\GP project\Project\AllData-Mirror' 
def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])
imlist = os.listdir(path)
imlist.sort(key=sortKeyFunc)
immatrix = array([array(Image.open('AllData-Mirror'+ '\\' + im2)).flatten()
              for im2 in imlist])

#test one image
img=immatrix[3].reshape(192,192)
plt.imshow(img)
plt.imshow(img,cmap='gray')

#create the data and the labels
data = immatrix
Label = y
train_data=[data,Label]
print (train_data[0].shape)
print (train_data[1].shape)

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 7
# number of epochs to train
nb_epoch = 3
#row and coulmn 
img_rows , img_cols =192 , 192

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#split the data to test and train
(X, y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#X_train = np.expand_dims(X_train,3)
#X_test = np.expand_dims(X_test, 3)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols , 1)
X_test  = X_test.reshape(X_test.shape[0], img_rows, img_cols , 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#test
i = 100
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        input_shape=( img_rows, img_cols , 1 )))
convout1 = Activation('relu')
model.add(convout1)
# Pooling
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

# Adding a second convolutional layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.5))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Compiling the CNN
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#fit the data
hist = model.fit(X_train, Y_train, batch_size = batch_size , nb_epoch = nb_epoch,
                          verbose=1, validation_data=(X_test, Y_test))


# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#Score
score = model.evaluate(X_test, Y_test, verbose = 0)
print(score)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:10]))
print(Y_test[1:10])


##Prediction Part
import numpy as np
from keras.preprocessing import image

img_pred = Image.open('C:\\Users\\youss\\Downloads\\test2.jpg').convert('L')
img_pred = img_pred.resize((192,192))
print(img_pred.size)
img_pred = numpy. array(img_pred)
print(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
img_pred = img_pred.reshape(img_pred.shape[0], 192, 192 , 1)
rslt = model.predict_classes(img_pred)
print(rslt)

if rslt == 0:
    prediction = "Angry"
elif rslt == 1:
    prediction = "Disgust"
elif rslt == 2:
    prediction = "Fear"
elif rslt == 3:
    prediction = "Happy"
elif rslt == 4:
    prediction = "Sad"
elif rslt == 5:
    prediction = "Surprise"
elif rslt == 6:
    prediction = "Neutral"   

print(prediction)    



##Save model to json
import os
from keras.models import model_from_json

clssf = model.to_json()
with open("EmotionRecognition.json", "w") as json_file:
    json_file.write(clssf)
model.save_weights("ERweights.h5")
print("model saved to disk....")



"""
#model cat and dog
# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (192, 192, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:/deeplearning/[FreeTutorials.Us] deeplearning/10 Building a CNN/Convolutional_Neural_Networks/dataset/train_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('D:/deeplearning/[FreeTutorials.Us] deeplearning/10 Building a CNN/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 3,
                         validation_data = test_set,
                         validation_steps = 2000)
##Prediction Part
import numpy as np
from keras.preprocessing import image

img_pred = image.load_img('D:/deeplearning/[FreeTutorials.Us] deeplearning/10 Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_5.jpg', target_size = (64, 64))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = classifier.predict(img_pred)

ind = training_set.class_indices

if rslt[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"
    
##Save model to json
import os
from keras.models import model_from_json

clssf = model.to_json()
with open("EmotionRecognition.json", "w") as json_file:
    json_file.write(clssf)
model.save_weights("ERweights.h5")
print("model saved to disk....")  
"""

