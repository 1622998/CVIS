# coding=utf-8
# -*- coding: cp1252 -*-

import cv2
import numpy as np
import glob

# Speichern einer kolorierten Punktwolke (Quelle: Oliver Wasenmüller)
ply_headerC = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_plyc(filename, verts):
    verts = verts.reshape(-1, 6)
    with open(filename, 'w') as f0:
        f0.write(ply_headerC % dict(vert_num=len(verts)))
        np.savetxt(f0, verts, '%f %f %f %d %d %d')


# ---------------------------------------

print "----------------------"
print "Computer Vision (CVIS)"
print "Übungsblatt 09"
print "--------------"

f = fx = fy = 721.5
cx = 690.5
cy = 172.8
tx = baseline = 0.54  # in Meter


# Funktion zum Einlesen der Bilder (Quelle: Benjamin Schönke)
def bildpaar_einlesen(path, pair_name, ending):
    print "Aufgabe 1: Disparität"
    print "1.1 *lese Bildpaar ein*"
    pics = glob.glob(path + pair_name + "*" + ending)
    img1 = cv2.imread(pics[0])
    img2 = cv2.imread(pics[1])

    print "1.2 *finde dichte Matches zwischen linken und rechten Bild*"

    def finde_dichte_matches(blocksize):
        print "BlockSize: ", blocksize
        min_disp = 1  # 0-10 Gibt die minimale Disparität an
        num_disp = 16 * 5  # 16*y Gibt die maximale Disparität an; immer vielfaches von 16

        # Erzeugen des Matcher Objekts
        stereo = cv2.StereoSGBM_create(min_disp, num_disp, blocksize)

        # Berechnen der Disparität
        disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0

        def tiefe_berechnen(disparity_local):
            """
            Wandelt die Disparität in Tiefe um
            :return: Tiefe als zweidimensionales Array
            """
            tiefe_local = np.zeros(disparity_local.shape)
            for x in range(0, disparity_local.shape[0]):
                for y in range(0, disparity_local.shape[1]):
                    if disparity_local[x, y] == 0:
                        tiefe_local[x, y] = 0
                    else:
                        tiefe_local[x, y] = np.divide(f * tx, disparity_local[x, y])

            return tiefe_local

        tiefe = tiefe_berechnen(disparity)

        # Farbmapping des Graustufenbildes
        depth_img_out = cv2.applyColorMap((tiefe * (1 / tiefe.max() * 255)).astype(np.uint8), cv2.COLORMAP_JET)
        disp_img_out = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imwrite(pair_name + "_Tiefenbild_blockSize" + str(blocksize) + ending, depth_img_out)
        cv2.imwrite(pair_name + "_Disparitaetsbild_blockSize" + str(blocksize) + ending, disp_img_out)

        print "1.2: *Punktwolken-Generierung*"

        def generiere_kolorierte_punktwolke(tiefe_lokal, img):
            """
            Generiert eine kolorierte Punktwolke
            :return: Array, wobei jeder Punkt den Vektor [X,Y,Z,R,G,B] enthält
            """
            array = []
            for x in range(0, img.shape[0]):
                for y in range(0, img.shape[1]):
                    if tiefe_lokal[x, y] > 0:
                        z = tiefe[x, y]
                        r = img[x, y][2]
                        g = img[x, y][1]
                        b = img[x, y][0]
                        array.append([x, y, z, r, g, b])
            return array

        kolorierte_punktwolke = generiere_kolorierte_punktwolke(tiefe, depth_img_out)
        write_plyc('TiefenPunktwolke_' + pair_name + '.ply', np.array(kolorierte_punktwolke))
        print "Aufgabe 3: Rausch-Entfernung"
        stereo = cv2.StereoSGBM_create(min_disp, num_disp, blocksize, speckleWindowSize=100, speckleRange=1)
        disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
        tiefe = tiefe_berechnen(disparity)
        depth_img_out = cv2.applyColorMap((tiefe * (1 / tiefe.max() * 255)).astype(np.uint8), cv2.COLORMAP_JET)
        disp_img_out = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(pair_name + "_Tiefenbild_Rausch-entfernt_blockSize" + str(blocksize) + ending, depth_img_out)
        cv2.imwrite(pair_name + "_Disparitaetsbild_Rausch-entfernt_blockSize" + str(blocksize) + ending, disp_img_out)
        kolorierte_punktwolke = generiere_kolorierte_punktwolke(tiefe, depth_img_out)

        print "1.3: extrinsische Transformation auf Punktwolke anwenden"
        r = np.eye(3, dtype=float)

        extrinsics1 = np.vstack((np.hstack((r, np.array([0.3, 0.3, 0.3]))), np.array([0., 0., 0., 1.])))
        extrinsics2 = np.vstack((np.hstack((r, np.array([0.6, 0.6, 0.6]))), np.array([0., 0., 0., 1.])))
        extrinsics3 = np.vstack((np.hstack((r, np.array([0.9, 0.9, 0.9]))), np.array([0., 0., 0., 1.])))
        extrinsics4 = np.vstack((np.hstack((r, np.array([1.2, 1.2, 1.2]))), np.array([0., 0., 0., 1.])))

        def multipliziere_punktwolke(punktwolke, extrinsics):
            for x in range(0, punktwolke.shape[0]):
                for y in range(0, punktwolke.shape[1]):
                    for z in range(0, punktwolke.shape[2]):
                        punktwolke[x, y, z] = punktwolke.dot(extrinsics)
            return punktwolke

        transformierte_punktwolke1 = multipliziere_punktwolke(kolorierte_punktwolke, extrinsics1)
        transformierte_punktwolke2 = multipliziere_punktwolke(kolorierte_punktwolke, extrinsics2)
        transformierte_punktwolke3 = multipliziere_punktwolke(kolorierte_punktwolke, extrinsics3)
        transformierte_punktwolke4 = multipliziere_punktwolke(kolorierte_punktwolke, extrinsics4)

        write_plyc('TiefenPunktwolke_Rausch-entfernt_' + pair_name + '.ply', np.array(kolorierte_punktwolke))

    finde_dichte_matches(12)

    print "------------------------------------------------------------------------------------"


# [START] Aufruf der Funktion
bildpaar_einlesen('images/', 'KITTI14', '.png')

print("[ENDE]")
