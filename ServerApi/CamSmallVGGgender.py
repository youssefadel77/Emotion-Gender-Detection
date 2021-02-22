# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#from keras.utils import get_file
import numpy as np
#import argparse
import cv2
#import os
#import cvlib as cv

class GenderRecognition:
        def __init__(self):
            # Create emotion queue of last 'x' emotions to smooth the output
            self.model = None
            
        def gendermodel (self):
                # load model
                model_path = 'pre-trained/gender_detection.model'
                self.model = load_model(model_path)
                
        def GenderRun(self,face_crop):
               
                #classes
                classes = ['man','woman']
         
                # preprocessing for gender detection model
                
                face_crop = cv2.resize(face_crop, (96,96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)
                
                # apply gender detection on face
                conf = self.model.predict(face_crop)[0]
                print(conf)
                print(classes)
        
                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]
        
                #label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        
                return label
                
""" # download pre-trained model file (one-time download)
            dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
            model_path = get_file("gender_detection.model", dwnld_link,
                                 cache_subdir="pre-trained", cache_dir=os.getcwd())"""