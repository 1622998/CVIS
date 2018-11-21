#Begruessung
print("\nHerzlich willkommen zu der Uebungsaufgabe 03!")

#########################################################
#Aufgabe 1
print("\nAufgabe 1")

#Abfrage Anzahl x
Anzahl_d = input("Bitte waehlen Sie die Anzahl der d: ")

#Ausgabe d
#for schleife
print("\nAusgabe in for-schleife")
for x in range(Anzahl_d):
	if (x % 2 == 0):
		print("d")
	else:
		print("D")
		
#Ausgabe d
#while schleife
print("\nAusgabe in while-schleife")
x=0
while x < Anzahl_d:
	if (x % 2 == 0):
		print("d")
	else:
		print("D")
	x=x+1

#Ende
print("Ende Aufgabe 1")

#########################################################
#Aufgabe 2
print("\nAufgabe 2")

#Auswahl Programm
print("Waehlen Sie im folgenden zwischen den beiden folgenden Programmen:")
print("(1) Umrechnung von Celsius nach Kelvin")
print("(2) Umrechnung von Celsius nach Fahrenheit")
wahl = input("Bitte waehlen Sie ein Programm: ")

if wahl == 1:
	print("Sie haben sich fuer Umrechnung von Celsius nach Kelvin entscheiden")
	celsius = float(input("Temperatur in Celsius: "))
	if celsius >= -273.15:
		kelvin = celsius + 273.15
		print(celsius, 'Grad = ', kelvin, 'K')
	else:
		print("Fehler: unmoegliche Temperatur!")
elif wahl == 2:
	print("Sie haben sich fuer Umrechnung von Celsius nach Fahrenheit entscheiden")
	celsius = float(input("Temperatur in Celsius: "))
	if celsius >= -273.15:
		fahrenheit = 32.0 + 1.8*celsius
		print(celsius, "Grad = ", fahrenheit, "F")    
	else:
		print("Fehler: unmoegliche Temperatur!") 
else:
	print("falsche Eingabe bei Programmwahl")
	
#Ende
print("Ende Aufgabe 2")