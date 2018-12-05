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

        print "1.3 *stelle Disparitätsbilder dar*"
        disparity = cv2.normalize(disparity, np.zeros(disparity.shape), 0, 255, cv2.NORM_MINMAX)
        disp_img_out = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(pair_name + "_Disparitaetsbild_blockSize" + str(blocksize) + ending, disp_img_out)

        print "Aufgabe 2: Punktwolken-Generierung"
        print "2.1 *wandle Disparitätsbild in Tiefenbild um und stelle dar*"
        tiefe_norm = cv2.normalize(tiefe, np.zeros(tiefe.shape), 0, 255, cv2.NORM_MINMAX)
        depth_img_out = cv2.applyColorMap(tiefe_norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(pair_name + "_Tiefenbild_blockSize" + str(blocksize) + ending, depth_img_out)

        print "2.2 *wandle Tiefen- und Farbbild in kolorierte Punktwolke um*"

        def generiere_kolorierte_punktwolke(tiefe_lokal):
            """
            Generiert eine kolorierte Punktwolke
            :return: Array, wobei jeder Punkt den Vektor [X,Y,Z,R,G,B] enthält
            """
            array = []
            for y in range(0, img1.shape[1]):
                for x in range(0, img1.shape[0]):
                    if tiefe_lokal[x, y] > 0:
                        Z = tiefe[x, y]
                        X = ((x - cx) * Z) / fx
                        Y = ((y - cy) * Z) / fy
                        r = img1[x, y][2]
                        g = img1[x, y][1]
                        b = img1[x, y][0]
                        array.append([X, Y, Z, r, g, b])
            return np.array(array)

        kolorierte_punktwolke = generiere_kolorierte_punktwolke(tiefe)
        print "2.2 *speichere kolorierte Punktwolke im PLY Format ab*"
        write_plyc('TiefenPunktwolke_' + str(blocksize) + '.ply', np.array(kolorierte_punktwolke))

        print "Aufgabe 3: Rausch-Entfernung"
        print "3.1 *entferne Rauschen aus Disparitätsbilder*"
        stereo = cv2.StereoSGBM_create(min_disp, num_disp, blocksize, speckleWindowSize=100, speckleRange=1)
        disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
        tiefe = tiefe_berechnen(disparity)
        tiefe = cv2.normalize(tiefe, np.zeros(tiefe.shape), 0, 255, cv2.NORM_MINMAX)
        kolorierte_punktwolke = generiere_kolorierte_punktwolke(tiefe)
        print "3.2 *speichere resultierende Punktwolken ab*"
        write_plyc('TiefenPunktwolke_Rausch-entfernt_' + str(blocksize) + '.ply', np.array(kolorierte_punktwolke))

    finde_dichte_matches(4)
    finde_dichte_matches(8)
    finde_dichte_matches(12)
    finde_dichte_matches(16)

    print "------------------------------------------------------------------------------------"


# [START] Aufruf der Funktion
bildpaar_einlesen('images/', 'KITTI14', '.png')

print("[ENDE]")
