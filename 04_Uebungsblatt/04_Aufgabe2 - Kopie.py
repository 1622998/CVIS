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
x = img.shape[0]
y = img.shape[1]

print "x:", x
print "y:", y
vorne = img[:,:y/2]
hinten = img[:,y/2:]

cv2.imwrite("CVIS_vorne.png",vorne)
cv2.imwrite("CVIS_hinten.png",hinten)

# 3) Region auf Strasse platzieren
print("*Region wird eingefuegt*")
img[x+1:x+x, :y/2] = hinten
cv2.imwrite("CVIS_neu.png",img)








print("[ENDE]")
