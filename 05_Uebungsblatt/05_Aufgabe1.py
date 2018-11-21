import cv2
import numpy as np
print "Computer Vision (CVIS)"
print "Uebungsblatt 05"
print "Aufgabe 1: Punktwolken-Projektion"
# Gegeben ist eine Kamera mit fx=fy=460, cx=320, cy=240 und einer Auflösung von 640x480.
fx = 460
fy = 460
cx = 320
cy = 240
# Weiterhin gibt es vier 3D Punkte:
# X1 = (10,10,100), X2 = (33,22,111), X3 = (100,100,1000), X4 = (20,-100,100)
# Das Welt- und Kamera-Koordinatensystem sind identisch.
X1 = np.array([10, 10, 100], dtype=float)
X2 = np.array([33, 22, 111], dtype=float)
X3 = np.array([100, 100, 1000], dtype=float)
X4 = np.array([20, -100, 100], dtype=float)
# Kalibrierungsmatrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1], ], dtype=float)
# Rotationsmatrix
R = np.eye(3, dtype=float)  # 3x3
# Translationsvektor
t = np.zeros((3, 1))  # 3 vertikal, 1 horizontal
# Projektionsmatrix
P = K.dot(np.hstack((R, t)))
# Pixelpositionen
x1 = P.dot(np.hstack((X1, 1)))
x2 = P.dot(np.hstack((X2, 1)))
x3 = P.dot(np.hstack((X3, 1)))
x4 = P.dot(np.hstack((X4, 1)))
# jeweils durch letzten Wert teilen
x1 = np.rint(x1[:2] / x1[2]) # :2 alles bis zur Position 2 (exklusive der 2. Position)
x2 = np.rint(x2[:2] / x2[2])
x3 = np.rint(x3[:2] / x3[2])
x4 = np.rint(x4[:2] / x4[2])
#       (Hinweis: Die händige Berechnung dieser Pixelposition könnte beispielsweise eine Klausuraufgabe sein)
print "1) Pixelpositionen mit Hilfe homogener Koordinaten bestimmen:"
print "x1: ", x1
print "x2: ", x2
print "x3: ", x3
print "x4: ", x4
print "----------"
print "2) Pixelpositionen mit Hilfe der openCV Funktion bestimmen:"
imagePoints, jacobian = cv2.projectPoints(np.array([X1, X2, X3, X4]), R, t, K, np.array([], dtype=float))
print np.rint(imagePoints)
print "-----"
print "Frage: Liegen alle Pixel im Bild?"
def istPixelImBild(pixel, bild):
    if pixel[0] <= bild[0] and pixel[1] <= bild[1] and pixel[0] >= 0 and pixel[1] >= 0:
        print "Der Pixel", pixel, "liegt im Bild", bild, "."
    else:
        print "Der Pixel", pixel, "liegt NICHT im Bild", bild, "!"
bild = np.array([640, 480], dtype=float)
print istPixelImBild(x1,bild)
print istPixelImBild(x2,bild)
print istPixelImBild(x3,bild)
print istPixelImBild(x4,bild)
print "-----"
print "Frage: Was faellt bei den Bildpunkten von x1 und x3 auf?"
print "Die Bildpunkte x1 und x3 sind im 2D identisch, da sie sich in 3D ueberlagern."
print "[ENDE]"
