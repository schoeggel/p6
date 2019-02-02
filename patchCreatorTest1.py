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
#  3. Vorschau des gwarpten Bilds
#  4. Klick auf Zentrum oder zurück zu (2)
#  5. Vorschau mit neuem Zentrum
#  6. Speichern oder zurück zu (4)

imgname = "data\\test3a.png"

# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)


# load the image, clone it, and setup the mouse callback function
image = cv2.imread(imgname)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

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