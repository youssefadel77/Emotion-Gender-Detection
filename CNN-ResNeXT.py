# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:04:41 2019

@author: youss
"""

# -*- coding: utf-8 -*-
""" Aggregated Residual Transformations for Deep Neural Network.
Applying a 'ResNeXT' to CIFAR-10 Dataset classification task.
References:
    - S. Xie, R. Girshick, P. Dollar, Z. Tu and K. He. Aggregated Residual
        Transformations for Deep Neural Networks, 2016.
Links:
    - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

#from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
from numpy import genfromtxt
from tflearn.data_preprocessing import ImagePreprocessing
import pandas as pd
import cv2

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5
#load all data 
X_All = np.asarray(genfromtxt('fer2013.csv', delimiter=' ',  skip_header=1, dtype=float))
Y_All = np.asarray(genfromtxt('fer2013_y.csv', delimiter=',',  skip_header=1, dtype=int))

# Reshape the images into 48x4
X_All = X_All.reshape([-1, 48, 48, 1])

# 80% of the data for trainning and it's labels
X =  X_All[:int(35887 *.8)]
print(X.shape)

Y =  Y_All[:int(35887 *.8)]
print(Y.shape)


# 20% of the data for testing and it's labels
X_test = X_All[int(35887 *.8):]
print(X_test.shape)

Y_test = Y_All[int(35887 *.8):]
print(Y_test.shape)


# One hot encode the labels
Y = tflearn.data_utils.to_categorical(Y, 7)
Y_test = tflearn.data_utils.to_categorical(Y_test, 7)


# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([48, 48], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 48, 48, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 7, activation='softmax')
opt = tflearn.Momentum(learning_rate=0.1, lr_decay=0.1, decay_step=32000, staircase=True , momentum=0.9)
net = tflearn.regression(net, optimizer=opt,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='models/model_resnet_emotion',
                    max_checkpoints=20, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=150, validation_set=(X_test, Y_test),
              snapshot_epoch=False, snapshot_step=500,
              show_metric=True, batch_size=128, shuffle=True,
              run_id='resnext_emotion')