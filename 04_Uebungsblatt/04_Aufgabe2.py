# -*- coding: cp1252 -*-
# Computer Vision (CVIS)
# Uebungsblatt 04
# Aufgabe 2: Bildausschnitte
#
# Schreibe ein Python Programm, das
# 1) das angehaengte Bild einliest
# 2) die Region um das weiße Auto kopiert (Pixel Position kann "hard gecoded" werden)
# 3) diese links daneben auf der Strasse platziert
# 4) das Bild speichert
print("[START]")
print("*importiere*")

import cv2
import numpy as np

# 1) Farbbild einlesen
print("*Lese Farbbild ein*")

img = cv2.imread("CVIS_Uebung4.png",1) # Datei am selben Ort, wie diese Aufgabe


# 2) Region kopieren
print("*Region wird kopiert*")

region = img[175:275, 760:1060]
# print region.shape
cv2.imwrite("CVIS_Ue4_Region.png",region)
# cv2.imshow('Ausgeschnittene Region',region)

# 3) Region auf Strasse platzieren
print("*Region wird eingefuegt*")
img[170:270, 460:760] = region
cv2.imwrite("CVIS_Ue4_2Cars.png",img)
# cv2.imshow('Eingefuegte Region',img)







print("[ENDE]")
