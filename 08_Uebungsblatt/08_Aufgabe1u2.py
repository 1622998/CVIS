# -*- coding: cp1252 -*-

import cv2
import numpy as np
import glob
from siftdetector import detect_keypoints
import pickle
#import pathlib


# Folgende Funktion speichert die durchgeführten Berechnungen im Cache (Quelle: David Nadoba)
cache_dir = 'cache'
def once(func, *args, **kwargs):
    args_as_string = ', '.join(str(e) for e in args)
    key = func.__name__ + args_as_string
    filename = cache_dir + '/' + key.replace('/', '_').replace('\\', '_') + '.once'
    try:
        file = open(filename, 'rb')
        print("found " + filename + " in cache")
        value = pickle.load(file)
        file.close()
        return value
    except EnvironmentError:
        value = func(*args, **kwargs)
        #pathlib.Path(cache_dir).mkdir(parents=False, exist_ok=True)
        file = open(filename, 'wb')
        pickle.dump(value, file)
        print("write " + filename + " to cache")
        return value

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''
def write_ply(fn, verts):
    verts = verts.reshape(-1, 3)
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f')


print "----------------------"
print "Computer Vision (CVIS)"
print "Übungsblatt 08"
print "--------------"
print "Aufgabe 1: Epipolarlinien"

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
    [detected_keypoints1, descriptors1] = once(detect_keypoints, pics[0], 5)
    print "2. *extrahiere SIFT Keypoints von", pics[1], "*"
    [detected_keypoints2, descriptors2] = once(detect_keypoints, pics[1], 5)

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
    print "3. Matches mit KNN durchführen, einzeichnen, speichern und Anzahl ausgeben:"
    bf = cv2.BFMatcher() # Objekt erzeugen
    matches = bf.knnMatch(descriptors_cv2_1, descriptors_cv2_2, k=2)

    print "3. *führe alle Matches mit threshold von 0.7 durch*"
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
    print "Anzahl Matches:", len(good)
    cv2.imwrite(pairName + '_KNN07.png', img_out)

    print "3. *führe alle Matches mit threshold von 0.8 durch*"
    good = []
    pts1 = []
    pts2 = []
    threshold_matching = 0.8
    for m, n in matches:
        if m.distance < threshold_matching * n.distance:
            good.append([m])
            pts1.append(kp1_cv2[m.queryIdx].pt)
            pts2.append(kp2_cv2[m.trainIdx].pt)

    img_out = cv2.drawMatchesKnn(img1, kp1_cv2, img2, kp2_cv2, good, None)
    print "Anzahl Matches:", len(good)
    cv2.imwrite(pairName + '_KNN08.png', img_out)

    print "4. *Berechne Epipolarlinien für alle Keypoints des ersten Bildes*"
    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS) # ggf. selbst berechnen
    lines = cv2.computeCorrespondEpilines(pts1, 2, F)

    def drawlines(img1,img2,lines,pts1,pts2):
        print(img1.shape)
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,(int(pt1[0]), int(pt1[1])),5,color,-1)
            img2 = cv2.circle(img2,(int(pt2[0]), int(pt2[1])),5,color,-1)
        return img1,img2
    
    lines = lines.reshape(-1,3)
    img1,img2 = drawlines(gray1,gray2,lines,pts1,pts2)
    cv2.imwrite(pairName + '_1_mitLinien.png', img1)
    cv2.imwrite(pairName + '_2_mitLinien.png', img2)
    # Dekomposition der Fundamentalmatrix
    fx = 707
    fy = 707
    cx = 604
    cy = 180
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1], ], dtype=float)
    E = K.T*np.mat(F)*K
    R1, R2, t = cv2.decomposeEssentialMat(E)
    X1 = np.hstack((R1, t))
    X2 = np.hstack((R1, t*(-1)))
    X3 = np.hstack((R2, t))
    X4 = np.hstack((R2, t*(-1)))
    P1 = K.dot(X1)
    P2 = K.dot(X2)
    P3 = K.dot(X3)
    P4 = K.dot(X4)

    # Triangulieren von Punkten
    pointcloud_homo = cv2.triangulatePoints(P1,P2,pts1.T,pts2.T)
    pointcloud = cv2.convertPointsFromHomogeneous(pointcloud_homo.T)
    write_ply('punktwolke.ply', pointcloud)


picture_pair1 = glob.glob('images/KITTI11_*.png')
picture_pair2 = glob.glob('images/KITTI14_*.png')

bildpaarEinlesen(picture_pair1, 'KITTI11')
bildpaarEinlesen(picture_pair2,'KITTI14')


print("[ENDE]")
