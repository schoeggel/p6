#  Neuer PatchCreator 3d
#  Parameter werden im 3d Raum gesetzt, updates erfolgen dual für beide Seiten (2d)
#  Die Koordinatenbasis Zug wird in die Mitte des Bild gesetzt. (So, als wäre das Gitter möglichst in der Mitte)
#  es können keine Zugeschnittenen Bilder mehr verwendet werden

# Tastaturbefehle:
# I J K L : verschieben auf x und y Achse
# O P     : verschieben auf z Achse
# W S     : Kleiner Grösser y Achse
# A D     : Kleiner Grösser x Achse
# E F     : Rotiere x Achse
# R G     : Rotiere y Achse
# T H     : Rotiere z Achse
# Z       : Undo
# X       : eXport




import cv2
import numpy as np
import wtmCalib
from rigid_transform_3d import rigid_transform_3D
import math
from archive.trainfeature import imgMergerH

warpdim = 300
exportDone = False
warpexpL = np.zeros((warpdim, warpdim))
warpexpR = np.zeros((warpdim, warpdim))
BASENAME = "SBB/17"



class Param3d:

    def __init__(self):
        self.name = "noname"
        self.posx = 240
        self.posy = -240
        self.posz = 0
        self.rotx = 0  # rad
        self.roty = 0  # rad
        self.rotz = 0  # rad
        self.sizex = 100
        self.sizey = 100
        self.corners3d = np.zeros((5, 3))

    def __str__(self):
        s = str()
        s += f'name:  {self.name}\n'
        s += f'posx:  {self.posx}\n'
        s += f'posy:  {self.posy}\n'
        s += f'posz:  {self.posz}\n'
        s += f'rotx:  {self.rotx}\n'
        s += f'roty:  {self.roty}\n'
        s += f'rotz:  {self.rotz}\n'
        s += f'sizex: {self.sizex}\n'
        s += f'sizey: {self.sizey}\n'
        s += f'corners: \n{self.corners3d}\n'
        return s

    def updateCorners(self):
        # Ausgehend von der Grösse des quadratischen Patchs und dessen Zentrumskoordinaten
        # werden die Koordinaten der vier Eckpunkte berechnet. Bezugssystem: sys_zug
        # Ohne Rotation liegt der Patch auf xy Ebene mit dem Zentrum des Quadrats bei patchCenter3d

        # Patchmitte bis Patch Rand (in x oder y Richtung)
        dx = self.sizex / 2
        dy = self.sizey / 2

        # Alle Ecken erhalten vorerst den Mittelpunkt als Koordinaten
        self.corners3d = np.tile([self.posx, self.posy, self.posz], (5, 1))

        # Patchmitte bis Patch Ecken, die Differenz vom Mittelpunkt zur Ecke
        d = np.array([[-dx, +dy, 0],  # oben links
                      [+dx, +dy, 0],  # oben rechts
                      [-dx, -dy, 0],  # unten links
                      [+dx, -dy, 0],  # unten rechts
                      [0, 0, 0]])  # Mitte

        # Ecken erstellen
        self.corners3d = self.corners3d + d

        # Rotieren
        self.rotatePoints()

    def rotatePoints(self):
        pt = self.corners3d

        # Schwerpunkt auf Ursprung setzen
        t = np.average(pt, 0)
        pt = pt - t

        # Rotationsmatrizen mit den Winkeln in [rad] erstellen
        a, b, c = self.rotx, self.roty, self.rotz
        Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])

        # Multiplizere Matrizen (@ statt np.matmul)
        pt = pt @ Rx @ Ry @ Rz

        # Translation wieder rückgängig machen
        self.corners3d = pt + t


class Param2d:
    def __init__(self):
        self.roi = np.ones(4, dtype=int)  # oben links und unten rechts
        self.corners2d = np.zeros((5, 2))  # die vier Ecken plus Mitte
        self.extend = 250  # ROI geht vom Mittelpunkt diese Distanz pro Achse

    def setROI(self, extend=None):
        if extend is None:
            extend = self.extend
        else:
            self.extend = extend

        roi = np.array([self.corners2d[4], self.corners2d[4]]).flatten()
        d = np.array([-extend, -extend, +extend, +extend])
        roi = roi + d
        roi[roi < 0] = 0  # begrenzen, keine negativen
        self.roi = roi.astype(int)

    def getCenter(self):
        return int(self.corners2d[4][0][0]), int(self.corners2d[4][0][1])

    def getWarpCorners(self):
        c = np.zeros((4, 1, 2))
        c[0] = self.corners2d[0]  #[0]
        c[1] = self.corners2d[2]  #[1]
        c[2] = self.corners2d[3]  #[3]
        c[3] = self.corners2d[1]  #[2]
        return c


class Stereo:
    # alles was mit Koordinaten umrechnen, triangulieren, stereo etc zu tun hat.
    # Kamera Calibrierdaten
    cal = wtmCalib.CalibData()

    # die Kanonischen Einheitsvektoren des sys_zug, aber mit dem Ursprung noch bei [0,0,0] von sys_cam
    systemzug = np.array([[0, 0, 0],
                          [0.94423342, -0.2282705, 0.23731],
                          [-0.32667794, -0.5590511, 0.76207],
                          [-0.0412888, -0.7970912, -0.60245]])

    # Ursprung = Gittermittelpunkt aus Bild 13 (dort ist das Gitter etwa in der Bildmitte zu finden)
    a3xN = np.float64([[1966], [1343]])
    b3xN = np.float64([[2200], [1303]])
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

    def updateCorners2d(self, param3d: Param3d, param2dL: Param2d, param2dR: Param2d):
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
    global par3d, par2dL, par2dR, stereo
    stereo.updateCorners2d(par3d, par2dL, par2dR)
    par2dL.setROI()
    par2dR.setROI()


def updateScreen():
    global par2dL, par2dR, cloneL, cloneR, winmain, winwarp, winzoom, overlay, warpexpL, warpexpR
    imgL = cloneL.copy()
    imgR = cloneR.copy()

    # polygon einzeichnen
    drawPolygon(imgL, par2dL)
    drawPolygon(imgR, par2dR)

    # centerMarker setzen
    cv2.drawMarker(imgL, par2dL.getCenter(), (255, 255, 0), cv2.MARKER_CROSS, 5, 1)
    cv2.drawMarker(imgR, par2dR.getCenter(), (255, 255, 0), cv2.MARKER_CROSS, 5, 1)

    # ROIS ausschneiden
    ya, yb = par2dL.roi[1], par2dL.roi[3]
    xa, xb = par2dL.roi[0], par2dL.roi[2]
    zoomL = imgL[ya:yb, xa:xb, :]
    ya, yb = par2dR.roi[1], par2dR.roi[3]
    xa, xb = par2dR.roi[0], par2dR.roi[2]
    zoomR = imgR[ya:yb, xa:xb, :]

    # ROIS haben nicht identische Grösse, wenn am Rand des Bilds
    zoom = imgMergerH([zoomL, zoomR])

    # Verzerrte Bilder erstellen
    cv2.imshow(winzoom, zoom)
    cv2.imshow(winmain, imgL)


    warpL, warpexpL = warp(cloneL.copy(), par2dL.getWarpCorners())
    warpR, warpexpR = warp(cloneR.copy(), par2dR.getWarpCorners())
    overlay.draw(warpL)
    overlay.draw(warpR)
    warps = imgMergerH([warpL, warpR])

    cv2.imshow(winwarp, warps)


def warp(img, refPt):
    global warpdim
    M = cv2.getPerspectiveTransform(refPt.astype(np.float32),
                                    np.array([[0, 0], [0, warpdim], [warpdim, warpdim], [warpdim, 0]],
                                             dtype=np.float32))
    warped = cv2.warpPerspective(img.copy(), M, (warpdim, warpdim))

    # Bild in diesem zustand speichern für späteren Export
    warpexport = warped.copy()

    # Kontrast verbessern
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    for n in range(0, 3):
        warped[:, :, n] = clahe.apply(warped[:, :, n])

    # Overlay

    return warped, warpexport


def drawPolygon(img, par2d: Param2d):
    # zeichnet ein Polygon ol --> ul --> ur --> or
    poly = np.zeros((1, 4, 2), dtype=np.int32)
    poly[0][0] = par2d.corners2d[0]
    poly[0][1] = par2d.corners2d[1]
    poly[0][2] = par2d.corners2d[3]
    poly[0][3] = par2d.corners2d[2]
    img = cv2.polylines(img, poly, True, (0, 255, 255), 3)


class Overlay:
    # für das Overlay auf dem quadratischen Template
    otype  = -1                     # aktives Overplay Nr
    maxtypes = 14                   # wieviele sind definiert
    templateSize = -1               # die Kantenlänge des quadratischen Templates
    rgb = (255, 255, 0)             # Farbe des Overlays

    def __init__(self, templateSize):
        self.otype = 1
        self.templateSize = templateSize

    def next(self):
        self.otype += 1
        if self.otype > self.maxtypes:
            self.otype = 0

    def draw(self, img):
        t = self.otype
        d = self.templateSize
        x = d // 50

        # Die Kreise
        if t in [0, 1, 2, 3, 4, 5, 6, 7]:
            cv2.drawMarker(img, (d // 2, d // 2), self.rgb, cv2.MARKER_CROSS, 15, 1, 1)
            cv2.circle(img, (d // 2, d // 2), d // 2 - x * t, self.rgb, 1, cv2.LINE_4)
            cv2.circle(img, (d // 2, d // 2), d // 3 - x * t, self.rgb, 1, cv2.LINE_4)
            cv2.circle(img, (d // 2, d // 2), d // 6 - x * t, self.rgb, 1, cv2.LINE_4)
        elif t == 8:
            pass
        elif t == 9:
            cv2.drawMarker(img, (d // 2, d // 2), self.rgb, cv2.MARKER_CROSS, 15, 1, 1)
        # Das Gitter
        elif t in [10, 11, 12, 13, 14]:
            n = t - 9  # --> range 1 .. 5
            n = 2 * n + 1  # --> [3, 5, 7, 8, 9, 11]
            grid = np.linspace(0, d, n).astype(int)
            for xy in grid:
                cv2.line(img, (0, xy), (d, xy), self.rgb, 1)
                cv2.line(img, (xy, 0), (xy, d), self.rgb, 1)
        return img


def export():
    global exportDone, warpexpL, warpexpR, par3d, exportDone
    print(par3d)
    name = str(input("export: "))
    if len(name) < 2: return
    par3d.name = name
    savename = f'tmp/tcr3d-{name}'
    print("writing ", savename)
    cv2.imwrite(savename + "_L.png", warpexpL)
    cv2.imwrite(savename + "_R.png", warpexpR)
    savename += "__.txt"
    f = open(savename , "w")
    f.write(str(par3d))
    f.close()
    print('Done.')
    exportDone = True


def posx_callback(val):
    global par3d
    par3d.posx = val - 750
    updateAllViews()


def posy_callback(val):
    global par3d
    par3d.posy = val - 750
    updateAllViews()


def posz_callback(val):
    global par3d
    par3d.posz = val - 750
    updateAllViews()


def rotx_callback(val):
    global par3d
    par3d.rotx = val * math.pi / 180
    updateAllViews()


def roty_callback(val):
    global par3d
    par3d.roty = val * math.pi / 180
    updateAllViews()


def rotz_callback(val):
    global par3d
    par3d.rotz = val * math.pi / 180
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


def modTrackbarPos(trackbarname, winname, delta):
    value = cv2.getTrackbarPos(trackbarname, winname)
    value += delta
    cv2.setTrackbarPos(trackbarname, winname, value)


# Parameter
par3d = Param3d()
par2dL = Param2d()
par2dR = Param2d()
stereo = Stereo()

# Lade Bilder
basename = BASENAME
imgL = cv2.imread(basename + "L.png")
imgR = cv2.imread(basename + "R.png")
cloneL = imgL.copy()
cloneR = imgR.copy()

# Fenster und Bars
winmain = "Navigation"
winzoom = "Zoom L+R"
winwarp = "warped L+R"
winparam = "3d param"

# Navigation Window
cv2.namedWindow(winmain, cv2.WINDOW_NORMAL)
cv2.namedWindow(winzoom, cv2.WINDOW_NORMAL)
cv2.namedWindow(winwarp, cv2.WINDOW_NORMAL)
cv2.namedWindow(winparam, cv2.WINDOW_NORMAL)

# Parameter Trackbars

cv2.createTrackbar('Position x', winparam, par3d.posx + 750, 1500, posx_callback)
cv2.createTrackbar('Position y', winparam, par3d.posy + 750, 1500, posy_callback)
cv2.createTrackbar('Position z', winparam, par3d.posz + 750, 1500, posz_callback)
cv2.createTrackbar('Rotation x', winparam, par3d.rotx, 360, rotx_callback)
cv2.createTrackbar('Rotation y', winparam, par3d.roty, 360, roty_callback)
cv2.createTrackbar('Rotation z', winparam, par3d.rotz, 360, rotz_callback)
cv2.createTrackbar('Size x', winparam, par3d.sizex, 300, sizex_callback)
cv2.createTrackbar('Size y', winparam, par3d.sizex, 300, sizey_callback)
cv2.createTrackbar('Zoom', winparam, 200, 1000, zoom_callback)

# ROI setzen
par2dL.setROI()
par2dR.setROI()

#Overlay für warp Template
overlay = Overlay(warpdim)

#Anzeige Refresh
updateAllViews()


while True:
    key = cv2.waitKey(1) & 0xFF

    ############################    Translation    ###########################
    if key == ord('i'):
        modTrackbarPos('Position y', winparam, 1)
    if key == ord('k'):
        modTrackbarPos('Position y', winparam, -1)
    if key == ord('j'):
        modTrackbarPos('Position x', winparam, -1)
    if key == ord('l'):
        modTrackbarPos('Position x', winparam, 1)
    if key == ord('o'):
        modTrackbarPos('Position z', winparam, -1)
    if key == ord('p'):
        modTrackbarPos('Position z', winparam, 1)

    ############################    Rotation    ###########################
    if key == ord('e'):
        modTrackbarPos('Rotation x', winparam, 1)
    if key == ord('f'):
        modTrackbarPos('Rotation x', winparam, -1)
    if key == ord('r'):
        modTrackbarPos('Rotation y', winparam, 1)
    if key == ord('g'):
        modTrackbarPos('Rotation y', winparam, -1)
    if key == ord('t'):
        modTrackbarPos('Rotation z', winparam, 1)
    if key == ord('h'):
        modTrackbarPos('Rotation z', winparam, -1)

  ############################    Grösse    ###########################
    if key == ord('w'):
        modTrackbarPos('Size x', winparam, 1)
    if key == ord('s'):
        modTrackbarPos('Size x', winparam, -1)
    if key == ord('a'):
        modTrackbarPos('Size y', winparam, 1)
    if key == ord('d'):
        modTrackbarPos('Size y', winparam, -1)

    ############################    Diverse    ###########################
    if key == 32:
        overlay.next()
        updateAllViews()
    if key == ord('x'):
        export()
    if key == ord('q'):
        if exportDone:
            break

# close all open windows
cv2.destroyAllWindows()
exit(0)
