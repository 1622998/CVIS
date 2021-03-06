#-*- coding: cp1252 -*-

import cv2
import numpy as np
import glob
print "----------------------"
print "Computer Vision (CVIS)"
print "�bungsblatt 06"
print "---------------"
print "Aufgabe 1: Kamera Kalibrierung"

objp = np.zeros((9*6,3), np.float32) # Array f�r Schachbrettmuster erstellen 9x6
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # Array bef�llen f�r Schachbrettmuster

objectPoints = [] # objectPoints beinhalten die Reihen, Spalten f�r das Schachbrettmuster (immer gleich)
imagePoints = [] # imagePoints beinhalten die 2D-Koordinaten f�r die Ecken des Schachbretts

print "1. *lese Kalibrierungsbilder ein*"
image_pathes = glob.glob('calib_images/*.jpg')
for image in image_pathes:
    img = cv2.imread(image) # einlesen
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # img in Graustufenbild konvertieren
    
    print "2. *Erkenne Ecken des Schachbretts im Bild: ", image, "*"
    retval, corners = cv2.findChessboardCorners(img, (9,6))
    
    if retval == True:
        objectPoints.append(objp) # F�ge ausgef�lltes Array den objectPoints hinzu
        imagePoints.append(corners) # F�ge gefundene 2D-Koordinaten des Schachbrettmusters hinzu

        img = cv2.drawChessboardCorners(img, (9,6), corners,retval)
        cv2.imshow(image,img) # 3. *zeige erkannte Ecken im Bild*"
        cv2.waitKey(100)

cv2.destroyAllWindows()
print "4. *F�hre Kamerakalibrierung durch*"
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, gray.shape, None, None)

print "cameraMatrix: "
print cameraMatrix
print "5. 3D Punkte der Schachbrettecken mit Hilfe der bestimmten Kameraparameter \
zur�ck ins Bild projizieren und anzeigen (gem�� letzter �bung)"

def projektion3D(images, points3D, rotationM, transM, camM, distC, corners, flags):
    i = 0
    for image in images:
        img = cv2.imread(image)
        imagePoints2, jacobian = cv2.projectPoints(points3D, rotationM[i], transM[i], camM, distC)
        img = cv2.drawChessboardCorners(img, (corners,flags), imagePoints2, True)
        cv2.imshow(image,img)
        cv2.waitKey(100)
        i = i + 1
    cv2.destroyAllWindows()

projektion3D(image_pathes, objp, rvecs, tvecs, cameraMatrix, distCoeffs, 9, 6)
print "------------------------------------------"
print "Aufgabe 2: Modifizierte 3D Punktprojektion"
print "---------"
cameraMatrix2 = cameraMatrix
print "Geben Sie einen neuen Wert f�r fx ein. Alter Wert: ", cameraMatrix[0,0]
fx = input("Neuer Wert f�r fx: ")
cameraMatrix2[0,0] = fx
print "Geben Sie einen neuen Wert f�r cx ein. Alter Wert: ", cameraMatrix[0,2]
cx = input("Neuer Wert f�r cx: ")
cameraMatrix2[0,2] = cx
print "neue cameraMatrix ist: "
print cameraMatrix
print "---------"
print " 2.2: *Neue Projektion aufgrund neuer Kameraparameter wird durchgef�hrt*"
projektion3D(image_pathes, objp, rvecs, tvecs, cameraMatrix2, distCoeffs, 9, 6)
print("[ENDE]")
