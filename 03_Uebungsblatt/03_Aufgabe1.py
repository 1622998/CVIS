# -*- coding: cp1252 -*-

# Computer Vision (CVIS)
# Uebungsblatt 03
# Aufgabe 1
#
# Schreibe ein Python Programm, das
# * Bei dem Benutzer eine Zahl n abfragt
# * Den Buchstaben 'd' danach n-Mal ausgibt
# * Der Buchstabe 'd' soll abwechselnd klein und groß geschrieben werden
#
# 1a: verwende dazu eine for-schleife
# 1b: verwende dazu eine while-schleife

n = input("Zahl n: ")

print "For-Schleife:"
for i in range(n):
        if i%2 ==0:
                print "d"
        else:
                print "D"
               
print "While-Schleife:"
i = 0
while i < n:
        print "d"
        i += 1
        if i < n:
                print "D"
                i += 1
        if i == n:
                break
