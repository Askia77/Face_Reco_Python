import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Creer une liste afin d'y stocker les images
path = 'ImagesDatabase'
images = []
classNames = []
myList = os.listdir(path)


#Ajouter les images dans cette liste
for cls in myList:
    crntImg= cv2.imread(f'{path}/{cls}')
    images.append(crntImg)
    classNames.append(os.path.splitext(cls)[0])

#cette fonction defini ou trouve l'encodage d'une image
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList




encodeListKnown = findEncodings(images)
print('Encoding successful')

#initialisation et manipulation de la webcam

cap =cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    SImg = cv2.resize(img,(0,0),None,0.25,0.25)
    SImg = cv2.cvtColor(SImg, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(SImg)
    encodesCurFrame = face_recognition.face_encodings(SImg, facesCurFrame)

    #Prendre un visage de facesCurframe et lui attribuer son encodage dans encodesCurFrame
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        #Faire les comparaisons entre images connues et ce que la webcam voit
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDist)
        matchIndex = np.argmin(faceDist)

        #Si ce que la webcam voit correspond a une image connu, afficher le nom de la personne
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
