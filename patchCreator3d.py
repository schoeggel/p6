#  Neuer PatchCreator 3d
#  Parameter werden im 3d Raum gesetzt, updates erfolgen dual für beide Seiten (2d)
#  Die Koordinatenbasis Zug wird in die Mitte des Bild gesetzt. (So, als wäre das Gitter möglichst in der Mitte)
#  es können keine Zugeschnittenen Bilder mehr verwendet werden

import cv2
import numpy as np
import calibMatrix
from rigid_transform_3d import rigid_transform_3D, rmserror
import math

class Param3d:


	def __init__(self):
		self.posx = 20
		self.posy = 20
		self.posz = 20
		self.rotx = 0   # rad
		self.roty = 0   # rad
		self.rotz = 0	# rad
		self.sizex = 100
		self.sizey = 100
		self.corners3d = np.zeros((5,3))


	def updateCorners(self):
		# Ausgehend von der Grösse des quadratischen Patchs und dessen Zentrumskoordinaten
		# werden die Koordinaten der vier Eckpunkte berechnet. Bezugssystem: sys_zug
		# Ohne Rotation liegt der Patch auf xy Ebene mit dem Zentrum des Quadrats bei patchCenter3d

		# Patchmitte bis Patch Rand (in x oder y Richtung)
		dx = self.sizex / 2
		dy = self.sizey / 2

		# Alle Ecken erhalten vorerst den Mittelpunkt als Koordinaten
		self.corners3d = np.tile([self.posx,self.posy,self.posz], (5, 1))

		# Patchmitte bis Patch Ecken, die Differenz vom Mittelpunkt zur Ecke
		d = np.array([[-dx, +dy, 0],  # oben links
					  [+dx, +dy, 0],  # oben rechts
					  [-dx, -dy, 0],  # unten links
					  [+dx, -dy, 0],  # unten rechts
					  [0, 0, 0]])  # Mitte


		# Ecken erstellen
		self.corners3d =  self.corners3d + d

		# Rotieren
		self.rotatePoints()



	def rotatePoints(self):
		for  idx, e in enumerate(self.corners3d):
			self.corners3d[idx] = self.rotatePoint(e)

	def rotatePoint(self, pt):
		# Rotationsmatrizen mit den Winkeln in [rad] erstellen
		a, b, c = self.rotx, self.roty, self.rotz
		Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
		Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
		Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])

		# Multiplizere Matrizen (@ statt np.matmul)
		res = pt @ Rx @ Ry @ Rz
		#self.corners3d = res
		return res



class Param2d:
	def __init__(self):
		self.roi = np.ones(4, dtype= int)		  	# oben links und unten rechts
		self.corners2d = np.zeros((5,2))			# die vier Ecken plus Mitte
		self.extend  = 250							# ROI geht vom Mittelpunkt diese Distanz pro Achse

	def setROI(self, extend = None):
		if extend is None:
			extend = self.extend
		else:
			self.extend = extend

		roi = np.array([self.corners2d[4], self.corners2d[4]]).flatten()
		d = np.array([-extend, -extend, +extend, +extend])
		roi = roi + d
		roi[roi < 0] = 0			# begrenzen, keine negativen
		self.roi = roi.astype(int)




class Stereo:
	# alles was mit Koordinaten umrechnen, triangulieren, stereo etc zu tun hat.
	# Kamera Calibrierdaten
	cal = calibMatrix.CalibData()

	# die Kanonischen Einheitsvektoren des sys_zug, aber mit dem Ursprung noch bei [0,0,0] von sys_cam
	systemzug = np.array([[0, 0, 0],
						  [0.94423342, -0.2282705, 0.23731],
						  [-0.32667794, -0.5590511, 0.76207],
						  [-0.0412888, -0.7970912, -0.60245]])

	# Ursprung = Gittermittelpunkt aus Bild 13 (dort ist das Gitter etwa in der Bildmitte zu finden)
	a3xN = np.float64([[1966],[1343]])
	b3xN = np.float64([[2200],[1303]])
	center = cv2.triangulatePoints(cal.pl[:3], cal.pr[:3], a3xN[:2], b3xN[:2])

	# homogen --> karthesisch
	center /= center[3]
	center = center[:3]
	print(f'\nCenter:\n{center}')

	# Einheitsvektoren an die richtige Position verschieben
	center = np.tile(center.T, (4, 1))  # zeilen vervielfachen
	systemzug = systemzug + center  # Translation

	# Rotation und Translation berechnen und in Klassenvariablen schreiben
	systemcam = np.diag(np.float64([1, 1, 1]))  # kanonische Einheitsvektoren
	systemcam = np.append([np.zeros(3)], systemcam, axis=0)  # erste Zeile = Ursprung

	# Rotation und Translation zwischen den beiden Bezugssystem berechnen
	R, t = rigid_transform_3D(systemcam, systemzug)



	def updateCorners2d(self,param3d:Param3d, param2dL:Param2d, param2dR:Param2d):
		global stereo
		# corners umrechnen in sys_Cam, dann projezieren auf 2d Bild
		pts3d_cam = stereo.transformsys(param3d.corners3d, direction=0)

		# ein Punkt in dieser Form:  np.array([[-284.24021771, -119.18978814, 7760.07162296]])
		# Mehrere Punkte: Shape = (N, 1, 3)
		param2dL.corners2d, _ = cv2.projectPoints(pts3d_cam.reshape(5, 1, 3),
												  self.cal.rl, self.cal.tl,
												  self.cal.kl, self.cal.drl)
		param2dR.corners2d, _ = cv2.projectPoints(pts3d_cam.reshape(5, 1, 3),
												  self.cal.rr, self.cal.tr,
												  self.cal.kr, self.cal.drr)

	#self.refPt = tuple(self.refPt.flatten().astype(int))


	def transformsys(self, pts, direction):
		# rechnet punkte von einem Bezugssystem ins andere um
		# dir == 0: sys_zug --> sys_cam
		# dir == 1: sys_cam --> sys_zug
		# pts müssen in shape (n,3) sein, bspw: [[x,y,z]] oder [[x,y,z],[x2,y2,z2]]

		n = pts.shape[0]  # wieviele punkte ?

		if direction == 0:
			A = pts
			B2 = (self.R @ A.T) + np.tile(self.t, (1, n))
			B2 = B2.T
			return B2
		elif direction == 1:
			B = pts
			A2 = (B - np.tile(self.t, (1, n)).T) @ self.R
			return A2

		else:
			assert False



def updateAllViews():
	# rechnete 2d neu, aktualisiert die Bilder
	# TODO: nicht nur test
	update3d()
	update2d()
	updateScreen()


def update3d():
	global par3d
	par3d.updateCorners()


def update2d():
	global par3d, par2dL, par2dL, stereo
	stereo.updateCorners2d(par3d, par2dL, par2dR)
	par2dL.setROI()
	par2dR.setROI()



def updateScreen():
	global par2dL, par2dR, cloneL, cloneR, winmain, winwarp, winzoom
	imgL = cloneL.copy()
	imgR = cloneR.copy()
	# xyL = tuple([par2dL.corners2d[4][0][0].astype(int), par2dL.corners2d[4][0][1].astype(int)])
	# xyR = tuple([par2dR.corners2d[4][0][0].astype(int), par2dR.corners2d[4][0][1].astype(int)])
	# imgL = cv2.circle(imgL, xyL, 25, (255, 0, 255), -1)
	# imgR = cv2.circle(imgR, xyL, 25, (255, 0, 255), -1)

	#polygon einzeichnen
	drawPolygon(imgL, par2dL)
	drawPolygon(imgR, par2dR)

	# ROIS ausschneiden
	print(par2dL.roi)
	ya, yb = par2dL.roi[1], par2dL.roi[3]
	xa, xb = par2dL.roi[0], par2dL.roi[2]
	zoomL = imgL[ya:yb, xa:xb, :]


	cv2.imshow(winzoom, zoomL)
	cv2.imshow(winmain, imgL)

def drawPolygon(img, par2d:Param2d):
	# zeichnet ein Polygon ol --> ul --> ur --> or
	poly = np.zeros((1, 4, 2), dtype=np.int32)
	poly[0][0] = par2d.corners2d[0]
	poly[0][1] = par2d.corners2d[1]
	poly[0][2] = par2d.corners2d[3]
	poly[0][3] = par2d.corners2d[2]
	img = cv2.polylines(img, poly, True, (0,255,255), 3)


def posx_callback(val):
	global par3d
	par3d.posx = val - 512
	updateAllViews()

def posy_callback(val):
	global par3d
	par3d.posy = val - 512
	updateAllViews()

def posz_callback(val):
	global par3d
	par3d.posz = val - 512
	updateAllViews()

def rotx_callback(val):
	global par3d
	par3d.rotx = (val - 90) / 180 * math.pi
	updateAllViews()

def roty_callback(val):
	global par3d
	par3d.roty = (val - 90) / 180 * math.pi
	updateAllViews()

def rotz_callback(val):
	global par3d
	par3d.rotz = (val - 90) / 180 * math.pi
	updateAllViews()

def sizex_callback(val):
	global par3d
	par3d.sizex = val
	updateAllViews()

def sizey_callback(val):
	global par3d
	par3d.sizey = val
	updateAllViews()

def zoom_callback(val):
	global par2dL, par2dR
	par2dL.setROI(val)
	par2dR.setROI(val)
	updateAllViews()

	pass


# Parameter
par3d = Param3d()
par2dL = Param2d()
par2dR = Param2d()
stereo = Stereo()




# Lade Bilder
basename = "sbb/13"
imgL = cv2.imread(basename + "L.png")
imgR = cv2.imread(basename + "R.png")
cloneL = imgL.copy()
cloneR = imgR.copy()

#Fenster und Bars
winmain = "Navigation"
winzoom = "Zoom L+R"
winwarp = "warped L+R"
winparam = "3d param"

# Navigation Window
cv2.namedWindow(winmain, cv2.WINDOW_NORMAL)
cv2.namedWindow(winzoom)
cv2.namedWindow(winwarp, cv2.WINDOW_NORMAL)
cv2.namedWindow(winparam, cv2.WINDOW_NORMAL)

# Parameter Trackbars
val = 512
val_max = 2 * val
cv2.createTrackbar('Position x', winparam, val, val_max, posx_callback)
cv2.createTrackbar('Position y', winparam, val, val_max, posy_callback)
cv2.createTrackbar('Position z', winparam, val, val_max, posz_callback)
cv2.createTrackbar('Rotation x', winparam, 0, 180, rotx_callback)
cv2.createTrackbar('Rotation y', winparam, 0, 180, roty_callback)
cv2.createTrackbar('Rotation z', winparam, 0, 180, rotz_callback)
cv2.createTrackbar('Size x', winparam, val, val_max, sizex_callback)
cv2.createTrackbar('Size y', winparam, val, val_max, sizey_callback)
cv2.createTrackbar('Zoom', winparam, val, val_max, zoom_callback)



# ROI setzen
par2dL.setROI()
par2dR.setROI()


cv2.imshow(winmain, imgL)
cv2.waitKey(0)



# liefert die eckpunkte für den suchbereich.
# im Format [oly, ury, olx, urx]
def getROIptsL(self, extend=100):
	return self.getROIsingleSide(self.corners2DimgL, extend)


def getROIptsR(self, extend=100):
	return self.getROIsingleSide(self.corners2DimgR, extend)


@staticmethod
def getROIsingleSide(corners, extend):
	# Liefert den Suchbereich einer Seite
	minx, miny = corners.min(0)[0]  # liefert individuell, nicht paarweise
	maxx, maxy = corners.max(0)[0]
	olx, oly = minx - extend, miny - extend
	urx, ury = maxx + extend, maxy + extend

	# begrenzen, keine negativen
	olx, oly = max(olx, 0), max(oly, 0)
	urx, ury = max(urx, 0), max(ury, 0)

	return list(map(int, [oly, ury, olx, urx]))

# close all open windows
cv2.destroyAllWindows()