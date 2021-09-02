import cv2
import numpy as np
import face_recognition

#Real image
imgFoxx = face_recognition.load_image_file('imagesBasic/jamie-foxx-3.jpg')
imgFoxx =cv2.cvtColor(imgFoxx,cv2.COLOR_BGR2RGB)
#test image
imgTest = face_recognition.load_image_file('imagesBasic/Askia.jpg')
imgTest =cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#Trouver les visages et obtenir leur encodage
faceLoc= face_recognition.face_locations(imgFoxx)[0]
encodeFoxx= face_recognition.face_encodings(imgFoxx)[0]
cv2.rectangle(imgFoxx,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest= face_recognition.face_locations(imgTest)[0]
encodeTest= face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


#Comparer les encodages de deux visages
results = face_recognition.compare_faces([encodeFoxx],encodeTest)
faceDist = face_recognition.face_distance([encodeFoxx],encodeTest)
print(results, faceDist)

cv2.putText(imgTest,f'{results}{round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Jaime Foxx', imgFoxx)
cv2.imshow('Jaime Foxx test', imgTest)
cv2.waitKey(0)