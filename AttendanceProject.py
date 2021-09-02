import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Generer une liste a partir de notre reperoire de base d'images
path = 'ImagesDatabase'
images = []
classNames = []
myArray = os.listdir(path)
print(myArray)


#Ajouter les images dans cette liste
for cls in myArray:
    curImg= cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

#cette fonction defini ou trouve l'encodage d'une image
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#Fonction pour generer un fichier csv enregistrant les noms de perosnnes reconnues et le temps.
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString}')


#markAttendance('Askia')

encodeListKnown = findEncodings(images)
print('Encoding finished')

#On utilise open pour initialiser et manipuler la Webcam

cap =cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    SImg = cv2.resize(img,(0,0),None,0.25,0.25)
    SImg = cv2.cvtColor(SImg, cv2.COLOR_BGR2RGB)

    actualFaceLoc = face_recognition.face_locations(SImg)
    actualFaceEncode = face_recognition.face_encodings(SImg, actualFaceLoc)

    #Prendre un visage de actualFaceLoc et lui attribuer son encodage dans actualFaceEncode
    for encodeFace,faceLoc in zip(actualFaceEncode,actualFaceLoc):
        #Faire les comparaisons entre images dans la base d'images et ce qui est devant la Webcam
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDist)
        matchIndex = np.argmin(faceDist)

        #Lorsqu'il ya correspondance, alors le noms de l'individu apparait dans un cadran rectangulaire
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

# faceLoc= face_recognition.face_locations(imgFoxx)[0]
# encodeFoxx= face_recognition.face_encodings(imgFoxx)[0]
# cv2.rectangle(imgFoxx,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest= face_recognition.face_locations(imgTest)[0]
# encodeTest= face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
#
# #Comparer les encodages de deux visages
# results = face_recognition.compare_faces([encodeFoxx],encodeTest)
# faceDist = face_recognition.face_distance([encodeFoxx],encodeTest)