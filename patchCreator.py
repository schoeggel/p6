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
# Tastatur:
#   s: Template speichern, name in Konsole tippen und bestätigen.
#   i, j, k, l : Ecken um 1 pixel verschieben
#   u, o : Rotieren ccw, cw
#   b, n : breiter, schmaler
#   c, v : höher, niederiger
# TODO : Rotation funktioniert noch nicht korrekt


# import the necessary packages
import cv2
import numpy as np
from enum import Enum
from math import cos,sin



class edit(Enum):
	UP1PX = 1
	DOWN1PX =2
	LEFT1PX = 3
	RIGHT1PX =4
	ROT_CCW = 5
	ROT_CW =6
	SCALE_X_POS = 7
	SCALE_X_NEG = 8
	SCALE_Y_POS = 9
	SCALE_Y_NEG = 10
	SWAP_CORNERS =11


#Quellbild
srcimagename = "tmp\\13Lcrop1.png"
#srcimagename = "data\\test3a.png"
#srcimagename = "data\\test3b.png"
#srcimagename = "data\\test-contrast1.png"

#Für den Export (ohne Fadenkreuz)
warpexport = []

# refPt sind die Koordinaten der 4 Ecken im Quellbild
refPt =np.zeros((4,2), dtype=int)
undo = refPt.copy()
letzterKlick = np.zeros((2,), dtype=int)
ecke = 0
warpdim = 300

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, ecke, image, letzterKlick, undo

	# Linker klick setzt aktuelle Position der Ecke. Mit der Maus ziehen verschiebt alle Ecken
	if event == cv2.EVENT_LBUTTONDOWN:
		letzterKlick = (x, y)
		print(f'(x,y) = ({x}, {y})')

	# Beim Loslassen prüfen, ob die Koordinaten noch übereinstimmen.
	# Nein: Mousedrag, alle eckpunkte ändern
	#  Ja: Nur die aktuelle Ecke ändern
	elif event == cv2.EVENT_LBUTTONUP:
		undo = refPt.copy()
		drag =  np.array((x,y)) - letzterKlick
		if drag.sum() != 0:
			refPt = refPt + drag
		else:
			refPt[ecke] =  letzterKlick
		updateImages()

	elif event == cv2.EVENT_RBUTTONDOWN:
		#nächste Ecke
		ecke = ecke + 1
		if ecke >= 4: ecke = 0
		updateImages()

def updateImages():
	global refPt, image
	image = clone.copy()
	for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0),(0,2),(1,3)]:
		cv2.line(image, tuple(refPt[a]), tuple(refPt[b]), (0, 255, 255), 1)
	cv2.drawMarker(image, tuple(refPt[ecke]), (255,255,255),cv2.MARKER_DIAMOND, 12, 1)
	cv2.drawMarker(image, tuple(refPt[ecke]), (255,0,255),cv2.MARKER_DIAMOND, 11, 1)
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
	cv2.circle(warpedimg,(warpdim//2,warpdim//2),warpdim//3, (255,255,0),1,cv2.LINE_4)
	cv2.circle(warpedimg,(warpdim//2,warpdim//2),warpdim//6, (255,255,0),1,cv2.LINE_4)
	cv2.imshow("warped", warpedimg)


def editSrc(e: edit):
	global refPt, undo
	undo = refPt.copy()
	if e == edit.DOWN1PX:
		refPt = refPt + [[0, 1]] * 4

	elif e == edit.UP1PX:
		refPt = refPt - [[0, 1]] * 4

	elif e == edit.LEFT1PX:
		refPt = refPt + [[-1, 0]] * 4

	elif e == edit.RIGHT1PX:
		refPt = refPt - [[-1, 0]] * 4

	elif e == edit.SCALE_X_NEG:
		refPt = refPt + [[1, 0], [1, 0], [-1, 0], [-1, 0]]

	elif e == edit.SCALE_X_POS:
		refPt = refPt - [[1, 0], [1, 0], [-1, 0], [-1, 0]]

	elif e == edit.SCALE_Y_NEG:
		refPt = refPt + [[0, 1], [0, -1], [0, -1], [0, 1]]

	elif e == edit.SCALE_Y_POS:
		refPt = refPt - [[0, 1], [0, -1], [0, -1], [0, 1]]

	elif e in [edit.ROT_CCW, edit.ROT_CW]:
		alpha = 0.01
		if e == edit.ROT_CW:
			alpha = -alpha
		m = np.average(refPt, 0)
		print(f'Schwepunkt:  {m}')
		mod = refPt - np.tile(m, (4,1))
		rot = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])  # rad
		mod = np.matmul(mod, rot)
		mod = mod +  np.tile(m, (1, 4))
		refPt = mod.astype(int)

	elif e == edit.SWAP_CORNERS:
		refPt = np.roll(refPt,2)  # rollt 2 stellen im flachen array

	updateImages()



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

	# q --> quit
	if key == ord("q"):
		break

	# z --> undo
	if key == ord("z"):
		refPt = undo.copy()
		updateImages()

	if key == ord('i'):
		editSrc(edit.UP1PX)
	if key == ord('j'):
		editSrc(edit.LEFT1PX)
	if key == ord('k'):
		editSrc(edit.DOWN1PX)
	if key == ord('l'):
		editSrc(edit.RIGHT1PX)

	if key == ord('b'):
		editSrc(edit.SCALE_X_POS)
	if key == ord('n'):
		editSrc(edit.SCALE_X_NEG)
	if key == ord('c'):
		editSrc(edit.SCALE_Y_POS)
	if key == ord('v'):
		editSrc(edit.SCALE_Y_NEG)

	if key == ord('u'):
		editSrc(edit.ROT_CCW)
	if key == ord('o'):
		editSrc(edit.ROT_CW)
	if key == ord('x'):
		editSrc(edit.SWAP_CORNERS)



if len(savename) != 0:
	if savename.find(".png") < 2:
		savename = savename + ".png"
	savename = "tmp/" + savename
	print("writing ", savename)
	cv2.imwrite(savename,warpexport)


# close all open windows
cv2.destroyAllWindows()