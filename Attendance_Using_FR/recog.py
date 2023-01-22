#Importing Required Modules
import numpy as np
import csv
import pickle
import cv2
from datetime import datetime
#Initializing HAAR Cascades
cascadeface = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
#Initializing Recognizers
recognizer = cv2.face.LBPHFaceRecognizer_create()
#Reading Training File
recognizer.read("trainerHack.yml")
entry = []
val = []
lbl = {}
#Curent Date and Time
currdt = datetime.today()
#Reading Pickle File
with open("lbl.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    lbl = {v : k for k,v in og_labels.items()}
fields = ["Name", "Time"]
#Seting up Video Capture    
capture = cv2.VideoCapture(0)
while(True):
    ret, frame= capture.read()
    #Convert color image to grayscale
    grayclr= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= cascadeface.detectMultiScale(grayclr, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #Initializing Region of Interest
        gray_roi=grayclr[y:y+h , x:x+w]   
        gray_clr=frame[y:y+h , x:x+w]    
        id,conf= recognizer.predict(gray_roi)
        #setting up confidence values 
        if conf>= 45 and conf<=85:
            #printing Names and Current Time
            print(lbl[id])
            print(datetime.now())
            print("\n")
            entry.append(lbl[id])
            val.append(datetime.now())      
        #Opening and Updating CSV file          
        #with open("Attendance.csv", 'w', encoding='utf-8') as csvfile:
        #    writer = csv.writer(f)
        #    writer.writerow(fields)
        #    writer.writerow(entry)
        #    writer.writerow(val)
        rec_color=(0,0,255)               #Blue Green Red
        brush=3
        width=x+w
        height=y+h
        #Drawing Region of Interest
        cv2.rectangle(frame, (x,y),(width,height),rec_color,brush)
        
    #show result
    cv2.imshow("video",frame)
    if cv2.waitKey(20)&0xFF== ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()    