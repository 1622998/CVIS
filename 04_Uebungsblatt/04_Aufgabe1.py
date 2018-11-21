# Computer Vision (CVIS)
# Uebungsblatt 04
# Aufgabe 1: Farbkanaele
#
# Schreibe ein Python Programm, das
# 1) ein Farbbild einliest
# 2) dieses Bild nach den Farbkanaelen aufteilt
#       a) mit Hilfe der split Methode
#       b) mit Hilfe des Doppelpunkts (:)
# 3) Jeden Farbkanal als extra Farbbild abspeichert

import cv2
import numpy as np

# 1) Farbbild einlesen
print("*Lese Farbbild ein*")

img = cv2.imread("CVIS_Uebung4.png",1) # Datei am selben Ort, wie diese Aufgabe


# 2) Farbkanaele aufteilen
print("*Teile Farbkanaele auf*")
#   a) mit split
print("(mit split)")

b,g,r = cv2.split(img)
zeros = b * 0
b = cv2.merge((b, zeros, zeros))
g = cv2.merge((zeros, g, zeros))
r = cv2.merge((zeros, zeros, r))


#   b) mit Doppelpunkt:
print ("(mit Doppelpunkt)")
blau = img.copy()
gruen = img.copy()
rot = img.copy()

# blau o
blau[:,:,1] = 0
blau[:,:,2] = 0
#gruen 1
gruen[:,:,0] = 0
gruen[:,:,2] = 0
#rot 2
rot[:,:,0] = 0
rot[:,:,1] = 0


# 3) Jeden Farbkanal als extra Farbbild speichern
print("*Speichere Farbkanaele als extra Farbbild*")

cv2.imwrite("CVIS_Ue4_b.png",b)
cv2.imwrite("CVIS_Ue4_g.png",g)
cv2.imwrite("CVIS_Ue4_r.png",r)

cv2.imwrite("CVIS_Ue4_blau.png",blau)
cv2.imwrite("CVIS_Ue4_gruen.png",gruen)
cv2.imwrite("CVIS_Ue4_rot.png",rot)
