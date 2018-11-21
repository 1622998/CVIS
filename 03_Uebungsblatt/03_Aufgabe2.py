# -*- coding: cp1252 -*-
# Computer Vision (CVIS)
# Uebungsblatt 03
# Aufgabe 2
#
# Schreibe ein Python Programm, das
# * Die Konvertierung von Temperaturangaben in Celsius nach Fahrenheit oder Kelvin ermöglicht
# * Zuerst wird beim Benutzer abgefragt, welche Konvertierung er machen möchte
# * Danach muss der Benutzer eine Temperatur in Celsius angeben
# * Es wird die Temperatur in Fahrenheit ODER Kelvin ausgegeben
#Hinweise:
# * Celsius = 5/9*(Fahrenheit - 32).
# * Celsius = Kelvin - 273.15.
# * Die tiefste mögliche Temperatur ist der absolute Nullpunkt von -273.15 Grad Celsius
print "Konvertierung von Celsius nach Fahrenheit und umgekehrt"
wiederholung = 1
while wiederholung == 1:
    print "-------------------------------------------------------"
    print ""
    print "Wonach möchten Sie Celsius konvertieren?"
    print "[1] nach Fahrenheit"
    print "[2] nach Kelvin"
    print ""
    auswahl = input("Geben Sie bitte die Zahl Ihrer Auswahl an: ")
    celsius = -274
    while celsius < -273.15:
        celsius = input("Bitte geben Sie eine Temperatur in Celsius ein: ")
        if celsius < -273.15:
            print "Die Temperatur darf nicht unter dem Nullpunkt liegen!"
    if auswahl == 1:
        print "Sie haben Fahrenheit gewählt."
        fahrenheit = ((celsius*9)/5)+32
        print fahrenheit
    elif auswahl == 2:
        print "Sie haben Kelvin gewählt."
        kelvin = celsius + 273.15
        print kelvin
    else:
        print "Ungültige Eingabe!"
    print "Programm mit STRG+C abbrechen"
