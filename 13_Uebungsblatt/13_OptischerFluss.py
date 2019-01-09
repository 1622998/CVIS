# coding=utf-8
# -*- coding: cp1252 -*-

import cv2
import numpy as np
import glob

print "----------------------"
print "Computer Vision (CVIS)"
print "Übungsblatt 13"
print "--------------"
print "Optischer Fluss"


def bildpaar_einlesen(path, pair_name, ending):
    print "1 *lese Bilder ein*"
    pics = glob.glob(path + pair_name + "*" + ending)

    print "2 *berechne optischen Fluss für ", len(pics), " Bilder*"

    def optischen_fluss_berechnen(first0, second0):
        """
        Berechnet den optischen Fluss zwischen zwei Bildern
        :return: Farbbild mit optischen Fluss
        """
        first = cv2.imread(first0, 0)
        second = cv2.imread(second0, 0)
        flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 20, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((first.shape[0], first.shape[1], 3))
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = hsv.astype(np.uint8)
        bgr0 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr0

    x = 0
    while x < len(pics) - 1:
        bgr = optischen_fluss_berechnen(pics[x], pics[x + 1])
        bildname = 'OptischerFluss_' + pair_name + str(1 + x) + ending
        cv2.imwrite(bildname, bgr)
        print "*", bildname, " gespeichert*"
        x = x + 1


# [START] Aufruf der Funktion
bildpaar_einlesen('images/', '000042_', '.png')
print("[ENDE]")
