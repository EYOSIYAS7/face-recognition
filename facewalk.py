import os
import numpy as np
from PIL import Image
import cv2

import pickle 
trained_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

base_dir = os.path.dirname(os.path.abspath(__file__))
imge_dir = os.path.join(base_dir, "Images")
current_id = 0
labels_id  = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(imge_dir):
    for file in files: 

        if file.endswith("png") or file.endswith("jpg"):

            path = os.path.join(root, file)
            label = os.path.basename(root).lower()

            

            if not label in labels_id:
                labels_id[label] = current_id
                current_id += 1

            id_ = labels_id[label] 
            img = Image.open(path).convert("L")
            image_array = np.array(img, "uint8")


            faces = trained_face.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:

                roi =image_array[y:y+h,x:x+h]
                x_train.append(roi)
                y_labels.append(id_)

print(labels_id)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("faceRecog.yml")
