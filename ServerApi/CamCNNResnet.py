# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 04:35:22 2019

@author: youss
"""

import cv2
import numpy as np
import tflearn
import operator
from tflearn.data_preprocessing import ImagePreprocessing
from collections import deque

class EmotionRecognition:
    def __init__(self):
        # Create emotion queue of last 'x' emotions to smooth the output
        self.emotion_queue = deque(maxlen=10)
        self.n = 5
        self.model = None


    def smooth_emotions(self, prediction):
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            emotion_values = {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0, 'Neutral': 0.0}
    
            emotion_probability, emotion_index = max((val, idx) for (idx, val) in enumerate(prediction[0]))
            emotion = emotions[emotion_index]
    
            # Append the new emotion and if the max length is reached pop the oldest value out
            self.emotion_queue.appendleft((emotion_probability, emotion))
    
            # Iterate through each emotion in the queue and create an average of the emotions
            for pair in self.emotion_queue:
                emotion_values[pair[1]] += pair[0]
    
            # Select the current emotion based on the one that has the highest value
            average_emotion = max(emotion_values.items(), key=operator.itemgetter(1))[0]
    
            return average_emotion
            
    
    def process_image(self, roi_gray, frame):
            image_scaled = np.array(cv2.resize(roi_gray, (48, 48)), dtype=float)
            image_processed = image_scaled.flatten()
            image_processed = image_processed.reshape([-1, 48, 48, 1])
            #predict the input from camera
            prediction = self.model.predict(image_processed)
            #print('asdasdasdasdasd')
            #print(prediction)
            emotion = self.smooth_emotions(prediction)
    
    
            return emotion
        
    def model_emotion (self):
             # Real-time pre-processing of the image data
            img_prep = ImagePreprocessing()
            img_prep.add_featurewise_zero_center()
            img_prep.add_featurewise_stdnorm()
    
            # Real-time data augmentation
            img_aug = tflearn.ImageAugmentation()
            img_aug.add_random_flip_leftright()
            # img_aug.add_random_crop([48, 48], padding=8)
    
            # Building Residual Network
            net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
            net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
            net = tflearn.residual_block(net, self.n, 16)
            net = tflearn.residual_block(net, 1, 32, downsample=True)
            net = tflearn.residual_block(net, self.n - 1, 32)
            net = tflearn.residual_block(net, 1, 64, downsample=True)
            net = tflearn.residual_block(net, self.n - 1, 64)
            net = tflearn.batch_normalization(net)
            net = tflearn.activation(net, 'relu')
            net = tflearn.global_avg_pool(net)
    
            # Regression
            net = tflearn.fully_connected(net, 7, activation='softmax')
            mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)
            net = tflearn.regression(net, optimizer=mom,
                                     loss='categorical_crossentropy')
    
            self.model = tflearn.DNN(net, checkpoint_path='models/model_resnet_emotion',
                                max_checkpoints=10, tensorboard_verbose=0,
                                clip_gradients=0.)
    
            self.model.load('New_models/model_resnet_emotion-33500') 

"""
if __name__ == "__main__":
    emotion_recognition = EmotionRecognition()
    emotion_recognition.run()
"""    
    