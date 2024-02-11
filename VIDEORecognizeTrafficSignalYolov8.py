# -*- coding: utf-8 -*-
"""
Created on Jan 2024

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
#
# Downloaded from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data

dirVideo ="traffic-sign-to-test.mp4"

#Downloaded from https://www.pexels.com/video/road-trip-4434242/
#dirVideo ="production_id_4434242 (2160p).mp4"
#dirVideo="production_id_4606790 (2160p).mp4"

#dirnameYolo="runs\\detect\\train2\\weights\\best.pt"
dirnameYolo="bestDetectTrafficSign.pt"

from keras.models import Sequential, load_model
# Downloaded from https://github.com/faeya/traffic-sign-classification
#modelTrafficSignRecognition=load_model('traffic_classifier.h5')
# downloaded from https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition
modelTrafficSignRecognition=load_model('model.h5')


#creating dictionary to label all traffic sign classes.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

import cv2
import time

TimeIni=time.time()
# in  14 minutes = 800 seconds finish  
TimeLimit=1000


# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
print(class_list)

import numpy as np

import numpy

# https://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectTrafficSignWithYolov8 (img):
  
   TabcropTrafficSign=[]
   
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   Tabclass_name=[]
   results = model.predict(img)
   for i in range(len(results)):
       # may be several signals in a frame
       # there is no ROI
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       print(class_id)
       out_image = img.copy()
       for j in range(len(class_id)):
           con=confidence[j]
           label=class_list[class_id[j]] + " " + str(con)
           box=xyxy[j]
           
           cropTrafficSign=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           
           TabcropTrafficSign.append(cropTrafficSign)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))
           
           print(label)
           Tabclass_name.append(label)
            
      
   return TabcropTrafficSign, y,yMax,x,xMax, Tabclass_name


###########################################################
# MAIN
##########################################################
cap = cv2.VideoCapture(dirVideo)
# https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps=5.0
frame_width = 680
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

video_writer = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size) 
ContFrames=0
ContDetected=0
ContNoDetected=0
while (cap.isOpened()):
        ret, img = cap.read()
        if ret != True: break
        
        else:

            gray=img

            #cv2.imshow('Gray', gray)
            #cv2.waitKey(0)
            
            TabImgSelect, y, yMax, x, xMax, Tabclass_name =DetectTrafficSignWithYolov8(gray)
            
            if TabImgSelect==[]:
                #print( " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                #continue
            else:
                ContDetected=ContDetected+1
                #print( " DETECTED ")
                for z in range(len(TabImgSelect)):
                     #if TabImgSelect[z] == []: continue
                     gray1=TabImgSelect[z]
                     #cv2.waitKey(0)
                     start_point=(x[z],y[z]) 
                     end_point=(xMax[z], yMax[z])
                     color=(0,0,255)
                     # Using cv2.rectangle() method
                     # Draw a rectangle with blue line borders of thickness of 5 px
                     img = cv2.rectangle(gray, start_point, end_point,(255,0,0), 15)
                     # Put text
                     text_location = (x[z], y[z])
                     text_color = (255,255,255)
                     
                     # recognize signal
                     data=[]
                     SignalToRecognize=cv2.resize(gray1,(32,32))
                     data.append(np.array(SignalToRecognize))
                     X_test=np.array(data)
                     predict_x=modelTrafficSignRecognition.predict(X_test) 
                     classes_x=np.argmax(predict_x,axis=1)
                     print(str(classes[int(classes_x)+1]))
                     cv2.putText(img, str(classes[int(classes_x)+1]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 2 ,cv2.LINE_AA)
                     cv2.putText(gray1, str(classes[int(classes_x)+1]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 2 ,cv2.LINE_AA)
                             
                 #cv2.imshow('Trafic Sign', gray1)
                 #cv2.waitKey(0)
                 # para     
                 #show_image=cv2.resize(img,(1000,700))
                 #cv2.imshow('Frame', show_image)
                 #cv2.waitKey(0)
            img_show=cv2.resize(img,(frame_width,frame_height))     
            cv2.imshow('Frame', img_show)
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'): break 
            # saving video
            video_writer.write(img)    
            #a los 10 minutos = 600 segundos acaba     
            if time.time() - TimeIni > TimeLimit:
                    
                    break
                   
          
cap.release()
video_writer.release()
cv2.destroyAllWindows()
           
              
print("")           

print( " Time in seconds "+ str(time.time()-TimeIni))
