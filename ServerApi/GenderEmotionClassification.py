# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 01:28:52 2019

@author: Yahia
"""


from flask import  jsonify
#import pandas as pd
import cv2
import numpy as np
import cvlib as cv


from CamCNNResnet import EmotionRecognition
from CamSmallVGGgender import GenderRecognition


#run model for emotion
E = EmotionRecognition()
E.model_emotion()
G=GenderRecognition()
G.gendermodel()



UPLOAD_FOLDER= "./images/"

File_Path = "E:/Projects/GP project/Project/ServerApi/images/"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def classifayEmotionGender(filename):
    
    
    frame = cv2.imread( UPLOAD_FOLDER + filename)
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces, confidence = cv.detect_face(frame)
    
    for idx, f in enumerate(faces):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue
        gender = G.GenderRun(face_crop)
        emotion = E.process_image(face_crop, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, "Gender : " + gender , (startX, Y) ,font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Emotion : " + emotion , (startX, Y-50) , font, 0.7, (0, 255, 0), 2)
                
    cv2.imwrite( UPLOAD_FOLDER + 'new' + filename, frame)
    imagebas64 = File_Path + 'new' + filename
    return jsonify({"image":imagebas64})

