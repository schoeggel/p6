# Damit kann ein viereckiger Patch aus einem Foto extrahiert werden.
# Das Bild wird auf ein Quadrat gewarpt und gespeichert.
# Es dient dann als Bildquelle (patch) für die Klasse trainfeature
#
# Quelle für Mouseclick in open cv image: Adrian Rosebrock
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
#
# Funktion:
# Linker Mausklick: Ecke setzen
# Rechter Mausklick: Zur nächsten Ecke wechseln
# Linker Mausklick und ziehen: ganzes Viereck verschieben
# Tastatur 's' : Template speichern, name in Konsole tippen und bestätigen.

# import the necessary packages
import cv2
import numpy as np

#Quellbild
srcimagename = "tmp\\13Lcrop1.png"
#srcimagename = "data\\test3a.png"
#srcimagename = "data\\test3b.png"
#srcimagename = "data\\test-contrast1.png"

#Für den Export (ohne Fadenkreuz)
warpexport = []

# refPt sind die Koordinaten der 4 Ecken im Quellbild
refPt =np.zeros((4,2), dtype=int)
letzterKlick = np.zeros((1,2), dtype=int)
ecke = 0
warpdim = 300

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, ecke, image, letzterKlick

	# Linker klick setzt aktuelle Position der Ecke. Mit der Maus ziehen verschiebt alle Ecken
	if event == cv2.EVENT_LBUTTONDOWN:
		letzterKlick[0] = (x, y)

	# Beim Loslassen prüfen, ob die Koordinaten noch übereinstimmen.
	# Nein: Mousedrag, alle eckpunkte ändern
	#  Ja: Nur die aktuelle Ecke ändern
	elif event == cv2.EVENT_LBUTTONUP:
		drag =  np.array((x,y)) - letzterKlick[0]
		if drag.sum() != 0:
			refPt = refPt + drag
		else:
			refPt[ecke] =  letzterKlick[0]
		updateImages()

	elif event == cv2.EVENT_RBUTTONDOWN:
		#nächste Ecke
		ecke = ecke + 1
		if ecke >= 4: ecke = 0

def updateImages():
	global refPt, image
	image = clone.copy()
	for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0),(0,2),(1,3)]:
		cv2.line(image, tuple(refPt[a]), tuple(refPt[b]), (0, 255, 255), 1)
	cv2.imshow("image", image)
	warp()

def warp():
	global image, refPt, warpexport
	# warpt das gewählte viereck in ein quadrat und zeigt es an. Clone verwenden, damit ohne linien
	M = cv2.getPerspectiveTransform( refPt.astype(np.float32), np.array([[0,0],[0,warpdim],[warpdim,warpdim],[warpdim,0]],dtype=np.float32))
	warpedimg = cv2.warpPerspective(clone, M, (warpdim, warpdim))
	warpexport = warpedimg.copy()

	# Kontrast verbessern
	clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
	for n in range(0,3):
		warpedimg[:,:,n] = clahe.apply(warpedimg[:,:,n])

	# "Zielhilfe" anzeigen: Fadenkreuz und zwei konzentrische Kreise
	cv2.drawMarker(warpedimg,(warpdim//2,warpdim//2),(255,255,0),cv2.MARKER_CROSS, 15, 1,1)
	cv2.circle(warpedimg,(warpdim//2,warpdim//2),warpdim//2, (255,255,0),1,cv2.LINE_4)
	cv2.circle(warpedimg,(warpdim//2,warpdim//2),warpdim//4, (255,255,0),1,cv2.LINE_4)
	cv2.imshow("warped", warpedimg)


# load the image, clone it, and setup the mouse callback function
image = cv2.imread(srcimagename)
clone = image.copy()
cv2.namedWindow("image")
cv2.namedWindow("warped")
cv2.setMouseCallback("image", click_and_crop)


# keep looping until the 'q' key is pressed
savename = ""
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'c' key is pressed, break from the loop
	if key == ord("s"):
		savename =  str(input("Save as: "))
		break

if len(savename) != 0:
	if savename.find(".png") < 2:
		savename = savename + ".png"
	savename = "tmp/" + savename
	print("writing ", savename)
	cv2.imwrite(savename,warpexport)


# close all open windows
cv2.destroyAllWindows()