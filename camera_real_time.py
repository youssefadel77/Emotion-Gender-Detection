# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 04:35:22 2019

@author: youssef
"""
import sys
import cv2 
import numpy as np
from CamCNNResnet import EmotionRecognition
from CamSmallVGGgender import GenderRecognition
from backend import back
#import cvlib as cv
#import time

B=back()

class Camera:
     def __init__(self):
        #run model for emotion
        self.E = EmotionRecognition()
        self.G=GenderRecognition()
        self.video=None
        self.EmotionArr=[]
        self.NumofFrame=0
        self.ClientPhone=None
        self.AgentId=None
        self.GenderArr=[]
        
     def run(self):
        self.E.model_emotion()
        self.G.gendermodel()
        #camera real time 
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        #Object for external camera
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FPS, 1)
        
        #loop for stream
        while True:
            #to Check how miliesecond will take
            self.NumofFrame=self.NumofFrame+1
            #check a frame object
            check , frame = self.video.read()    
            #print(check)
            #print(frame)  #the image    
            
            #convert to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            #loop for faces
            for  (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_color = frame[y:y + h, x:x + w]
                        
                        face_crop = np.copy(frame[y:y + h, x:x + w])
                        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                            continue
                        
                        emotion = self.E.process_image(roi_gray, frame)
                        gender = self.G.GenderRun(face_crop)
                        
                        self.EmotionArr.append(emotion)
                        self.GenderArr.append(gender)
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, "Emotion: " + emotion , (x, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, "Gender: " + gender , (x, y-50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        
            #show the result      
            cv2.imshow('frame', frame)
            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
      
        """self.video.release()
        cv2.destroyAllWindows()
        """
     def GetPhoneNum (self , Phone) :
         self.ClientPhone = Phone
         
     def GetAgentId (self , AId) :
         self.AgentId = AId
         print(self.AgentId)
         
     def Exit(self):
        print(self.AgentId)
        print(self.ClientPhone)
        print(self.EmotionArr)
        print(self.GenderArr)
        #print(len(self.EmotionArr)) 
        #print(self.NumofFrame)
        
        result=B.report(self.AgentId,self.ClientPhone,self.GenderArr,self.EmotionArr)
        print(result)
        #shut down the camera
        self.video.release()
        cv2.destroyAllWindows()
        

#take much time
"""# loop through frames
while video.isOpened():
    # read frame from webcam 
    status, frame = video.read()
    
    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue
        gender = G.GenderRun(idx,face_crop)
        emotion = E.process_image(face_crop, frame)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Gender : " + gender, (startX, Y) , cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Emotion : " + emotion , (startX, Y-50) ,  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
    # display output
    cv2.imshow("gender detection", frame)
    
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
video.release()
cv2.destroyAllWindows()"""