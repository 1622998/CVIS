# -*- coding: cp1252 -*-

# Code besser in Kommandozeile ausführen

import cv2
import numpy as np
import glob
from siftdetector import detect_keypoints

print "----------------------"
print "Computer Vision (CVIS)"
print "Übungsblatt 07"
print "--------------"
print "Aufgabe 1: Feature"

# je kleiner Distanz im Feature Diskriptor, desto besser der Match

def to_cv2_kplist(kp):
    return list(map(to_cv2_kp, kp))

def to_cv2_kp(kp):
    return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3] / np.pi * 180)

def to_cv2_di(di):
    return np.asarray(di, np.float32)


print "1. *lese Bildpaar ein*"


def bildpaarEinlesen(pics, pairName):
    img1 = cv2.imread(pics[0])
    img2 = cv2.imread(pics[1])

    print "2. *extrahiere SIFT Keypoints von", pics[0], "*"
    [detected_keypoints1, descriptors1] = detect_keypoints(pics[0], 5)
    print "2. *extrahiere SIFT Keypoints von", pics[1], "*"
    [detected_keypoints2, descriptors2] = detect_keypoints(pics[1], 5)

    kp1_cv2 = to_cv2_kplist(detected_keypoints1)
    kp2_cv2 = to_cv2_kplist(detected_keypoints2)
    descriptors_cv2_1 = to_cv2_di(descriptors1)
    descriptors_cv2_2 = to_cv2_di(descriptors2)

    print "2. *zeichne Keypoints in Bild ein*"
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_out = np.array([])
    img2_out = np.array([])
    img1_with_kp = cv2.drawKeypoints(gray1, kp1_cv2, img1_out, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_kp = cv2.drawKeypoints(gray2, kp2_cv2, img2_out, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print "2. *speichere Bilder ab*"
    cv2.imwrite(pairName + "_kpOut1.png", img1_with_kp)
    cv2.imwrite(pairName + "_kpOut2.png", img2_with_kp)
    print "2. Anzahl Keypoints vom Bild ", pics[0], "ist: ", len(kp1_cv2)
    print "2. Anzahl Keypoints vom Bild ", pics[1], "ist: ", len(kp2_cv2)
    print "----------------------------"
    print "3. Matches durchführen, einzeichnen, speichern und Anzahl ausgeben:"
    bf = cv2.BFMatcher()
    print "3.a. *führe alle Matches durch*"
    matches = bf.match(descriptors_cv2_1, descriptors_cv2_2)
    print "Anzahl Matches:", len(matches)
    img_out = cv2.drawMatches(img1, kp1_cv2, img2, kp2_cv2, matches, None)
    cv2.imwrite(pairName + "_3a.png", img_out)
    print "3.b. Die 30 besten Matches: "
    matches = sorted(matches, key=lambda x: x.distance)
    print "Die 30 besten Matches:"
    for i in range(30):
        print i + 1, ".:", matches[i]
    img_out = cv2.drawMatches(img1, kp1_cv2, img2, kp2_cv2, matches[:30], None)
    cv2.imwrite(pairName + "_3b.png", img_out)
    print "3.c. *führe alle Matches mit threshold von 0.7 durch*"
    matches = bf.knnMatch(descriptors_cv2_1, descriptors_cv2_2, k=2)
    print "Anzahl Matches:", len(matches)
    good = []
    pts1 = []
    pts2 = []
    threshold_matching = 0.7
    for m, n in matches:
        if m.distance < threshold_matching * n.distance:
            good.append([m])
            pts1.append(kp1_cv2[m.queryIdx].pt)
            pts2.append(kp2_cv2[m.trainIdx].pt)

    img_out = cv2.drawMatchesKnn(img1, kp1_cv2, img2, kp2_cv2, good, None)
    cv2.imwrite(pairName + '_3c.png', img_out)


picture_pair1 = glob.glob('images/KITTI11_*.png')
picture_pair2 = glob.glob('images/KITTI14_*.png')

bildpaarEinlesen(picture_pair1, 'KITTI11')
bildpaarEinlesen(picture_pair2,'KITTI14')


print("[ENDE]")
