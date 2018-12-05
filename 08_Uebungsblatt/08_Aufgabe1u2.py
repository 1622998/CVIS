# coding=utf-8
# -*- coding: cp1252 -*-

import cv2
import numpy as np
import glob
from siftdetector import detect_keypoints
import pickle


# Funktionen zum Konvertieren in cv2-Format (Quelle: Oliver Wasenmüller)
def to_cv2_kplist(kp):
    return list(map(to_cv2_kp, kp))


def to_cv2_kp(kp):
    return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3] / np.pi * 180)


def to_cv2_di(di):
    return np.asarray(di, np.float32)


# ------------------------------------

cache_dir = 'cache'  # Ordner muss manuell erstellt werden


def once(func, *args, **kwargs):
    """
    Speichert durchgeführte Berechnungen im Cache (Quelle: David Nadoba)
    :param func: aufzurufende Funktion
    :param args: Argumente
    :param kwargs: Argumente
    :return:
    """
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
        file = open(filename, 'wb')
        pickle.dump(value, file)
        print("write " + filename + " to cache")
        return value


# -----------------------------------------------

# Zeichnen der Epipolarlinien (Quelle: Oliver Wasenmüller)
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, color, -1)  # Cast wegen Fehler
        img2 = cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, color, -1)  # Cast wegen Fehler
    return img1, img2


# -------------------

# Speichern von Punktwolken (Quelle: Oliver Wasenmüller)
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


# ---------------------------------------

print "----------------------"
print "Computer Vision (CVIS)"
print "Übungsblatt 08"
print "--------------"

fx = 707
fy = 707
cx = 604
cy = 180


# Funktion zum Einlesen der Bilder (Quelle: Benjamin Schönke)
def bildpaar_einlesen(path, pair_name, ending):
    print "Aufgabe 1: Epipolarlinien"
    print "1. *lese Bildpaar ein*"
    pics = glob.glob(path + pair_name + ending)
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
    cv2.imwrite(pair_name + "_kpOut1.png", img1_with_kp)
    cv2.imwrite(pair_name + "_kpOut2.png", img2_with_kp)
    print "2. Anzahl Keypoints vom Bild ", pics[0], "ist: ", len(kp1_cv2)
    print "2. Anzahl Keypoints vom Bild ", pics[1], "ist: ", len(kp2_cv2)
    print "---"
    print "3. Matches mit KNN durchführen, einzeichnen, speichern und Anzahl ausgeben:"
    bf = cv2.BFMatcher()  # Objekt erzeugen
    matches = bf.knnMatch(descriptors_cv2_1, descriptors_cv2_2, k=2)

    # je kleiner Distanz im Feature Deskriptor, desto besser der Match

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
    cv2.imwrite(pair_name + '_KNN07.png', img_out)

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
    cv2.imwrite(pair_name + '_KNN08.png', img_out)
    print "---"
    print "4. *Berechne Epipolarlinien für alle Keypoints des ersten Bildes*"
    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)
    f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)  # Fundamentalmatrix
    lines = cv2.computeCorrespondEpilines(pts1, 2, f)

    print "4. *Epipolarlinien einzeichnen*"
    lines = lines.reshape(-1, 3)
    img1, img2 = drawlines(gray1, gray2, lines, pts1, pts2)
    cv2.imwrite(pair_name + '_1_mitLinien.png', img1)
    cv2.imwrite(pair_name + '_2_mitLinien.png', img2)

    print "----------------------------"
    print "Aufgabe 2: Triangulierung"
    print "1. *Möglichkeiten r1, r2 und t für Pose zwischen Kameras bestimmen*"
    # Dekomposition der Fundamentalmatrix
    k = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1], ], dtype=float)
    e = k.T * np.mat(f) * k
    r1, r2, t = cv2.decomposeEssentialMat(e)

    # 4 Möglichkeiten für die Kamerapose
    p0 = k.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    p1 = k.dot(np.hstack((r1, t)))
    p2 = k.dot(np.hstack((r1, t * (-1))))
    p3 = k.dot(np.hstack((r2, t)))
    p4 = k.dot(np.hstack((r2, t * (-1))))

    # Triangulieren von Punkten
    pointcloud1 = cv2.convertPointsFromHomogeneous(cv2.triangulatePoints(p0, p1, pts1.T, pts2.T).T)
    pointcloud2 = cv2.convertPointsFromHomogeneous(cv2.triangulatePoints(p0, p2, pts1.T, pts2.T).T)
    pointcloud3 = cv2.convertPointsFromHomogeneous(cv2.triangulatePoints(p0, p3, pts1.T, pts2.T).T)
    pointcloud4 = cv2.convertPointsFromHomogeneous(cv2.triangulatePoints(p0, p4, pts1.T, pts2.T).T)

    def pruefe_3d_punkte(cloud):
        zaehler = 0
        for i in cloud:
            if i[0][2] >= 0:
                zaehler += 1
        return zaehler

    ergebnisse = [pruefe_3d_punkte(pointcloud1), pruefe_3d_punkte(pointcloud2), pruefe_3d_punkte(pointcloud3),
                  pruefe_3d_punkte(pointcloud4)]

    max_wert = max(ergebnisse)
    max_index = ergebnisse.index(max_wert)

    print "Ergebnisse: ", ergebnisse
    print "max_value: ", max_wert
    print "max_index: ", max_index

    if max_index == 0:
        pointcloud_final = pointcloud1
    elif max_index == 1:
        pointcloud_final = pointcloud2
    elif max_index == 2:
        pointcloud_final = pointcloud3
    else:
        pointcloud_final = pointcloud4

    write_ply('Punktwolke_' + pair_name + '.ply', pointcloud_final)
    print "------------------------------------------------------------------------------------"


# [START] Aufruf der Funktionen
bildpaar_einlesen('images/', 'KITTI11', '*.png')
bildpaar_einlesen('images/', 'KITTI14', '*.png')

print("[ENDE]")
