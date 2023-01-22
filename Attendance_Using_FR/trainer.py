#Importing Required Modules
import os
from PIL import Image
import numpy as np
import cv2
import csv
import pickle

#Stting Up directoriesto access stored trainable images of students
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "media")
#Initializing HAAR Cascades and LBPH Face Recognizer
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

ylabel = []
xlabel = []
current_id = 0
label_id = {}
#Scanning through directories for images
for root, dir, files in os.walk(IMG_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))
        
        if not label in label_id:
            label_id[label] = current_id
            current_id += 1
        id = label_id[label]
        #Converting Images to NumPy Arrays
        pilimg = Image.open(path).convert("L")
        imgarr = np.array(pilimg, "uint8")
        #Sending arrays to HAAR Cascades
        #In this method we check neighbours of a selected pixel to check the integrity of the NumPy array        
        faces = cascade_face.detectMultiScale(imgarr, scaleFactor = 1.5, minNeighbors = 5)
        for(x,y,w,h) in faces:
            region_of_interest = imgarr[y:y+h, x:x+w]
            xlabel.append(region_of_interest)
            ylabel.append(id)
#Creating a Pickle file
with open("lbl.pickle", 'wb') as f:
    pickle.dump(label_id, f)
#Training
recognizer.train(xlabel, np.array(ylabel))
recognizer.save("trainerHack.yml")

print("Training Complete!!")                        