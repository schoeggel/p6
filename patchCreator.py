# Damit kann ein viereckiger Patch aus einem Foto extrahiert werden.
# Die Messreferenz kann in die Mitte des Patchs gesetzt werden.
# Das Bild wird auf ein Quadrat gewarpt und gespeichert.
# Es dient dann als Bildquelle für die Klasse trainfeature
#
# Quelle für Mouseclick in open cv image: Adrian Rosebrock
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
#
# Ablauf:
#  1. Bild anzeigen
#  2. 4 Klicks für 4 Ecken
#  3. Vorschau des gewarpten Bilds
#  4. Klick auf Zentrum oder zurück zu (2)
#  5. Vorschau mit neuem Zentrum
#  6. Speichern oder zurück zu (4)

# verschiedene zustände des Bildes
# srcimagename: name der Quell Datei
# image: das aktuelle bild, das in Bearbeitung ist
# clone: ein Sicherungskopie mit dem usrprünglichen zustand des geladenen Bilds.

srcimagename = "data\\test3a.png"


# import the necessary packages
import cv2
import numpy as np

# refPt sind die Koordinaten der 4 Ecken im Quellbild
refPt =np.zeros((4,2), dtype=int)
ecke = 0

def click_center(event, x, y, flags, param):
	# TODO: Funktioniert noch nicht !!!!!!!


	# setze das Messzentrum neu,
	global refPt

	# Linker Mausklick liest die Bildkoordinaten
	if event == cv2.EVENT_LBUTTONDOWN:

		print("old pts:\n",refPt)

		newcenter = np.array((x, y))
		ofs = newcenter - np.array((150,150))

		# der Offset soll noch skaliert werden.
		# Dazu werden die quadratsummen der seiten von Quell und warp-Bild verglichen

		u2 =  np.array([300, 300])**2
		u1 = 0
		for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
			u1 = u1 + (refPt[a] - refPt[b])**2

		u1 = u1.sum()**0.5
		u2 = u2.sum()**0.5

		print("u1, u2:\n",u1, u2)

		s = (u2/u1)**0.5
		ofs = (s * ofs).astype(int)
		refPt = refPt + np.tile(ofs,(4,1))

		print("ofs:\n", ofs)
		print("new pts:\n", refPt)
		updateImages()


	pass
	# TODO





def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, ecke, image

	# Linker Mausklick liest die Bildkoordinatenspeichert sie in die akteulle ecke
	if event == cv2.EVENT_LBUTTONDOWN:
		print("Mbutton down, ecke:", ecke)
		refPt[ecke] = (x, y)

	# beim Loslassen der linken Maustaste werdenbeide Anzeigen aktualisiert
	elif event == cv2.EVENT_LBUTTONUP:
		updateImages()


	elif event == cv2.EVENT_RBUTTONDOWN:
		#nächste Ecke
		ecke = ecke + 1
		if ecke == 4: ecke = 0

	elif event == cv2.EVENT_RBUTTONDBLCLK:
		# Ecken sind fertig ausgewählt
		pass #TODO Center clicken

def updateImages():
	global refPt, image
	# Viereck zeichnen
	image = clone.copy()
	for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
		cv2.line(image, tuple(refPt[a]), tuple(refPt[b]), (0, 255, 255), 1)
	cv2.imshow("image", image)
	print("update warp")
	warp(image, refPt)

def warp(image: np.array, pts: np.array) -> None:
	# warpt das gewählte viereck in ein quadrat und zeigt es an. Clone verwenden, damit ohne linien
	M = cv2.getPerspectiveTransform( pts.astype(np.float32), np.array([[0,0],[0,300],[300,300],[300,0]],dtype=np.float32))
	warpedimg = cv2.warpPerspective(clone, M, (300, 300))

	# Zentrum = späterer Messpunkt. Hier schon anzeigen
	cv2.drawMarker(warpedimg,(150,150),(255,255,0),cv2.MARKER_CROSS, 15, 1,1)
	cv2.imshow("warped", warpedimg)


# load the image, clone it, and setup the mouse callback function
image = cv2.imread(srcimagename)
clone = image.copy()
cv2.namedWindow("image")
cv2.namedWindow("warped")
cv2.setMouseCallback("image", click_and_crop)
cv2.setMouseCallback("warped", click_center)


# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()