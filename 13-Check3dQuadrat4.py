import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import calibMatrix
from GitterEckpunkte import eckpunkte
from equalAxis3d import set_axes_equal


# Es sollen die 4 Eckpunkte trianguliert werden, um zu kontrollieren, ob
# die 4 Punkte im Raum ein Quadrat ergeben.
# Problem : 3d koordinaten stimmen noch nicht ganz ??
# Es handelt sich grob im ein längliches Rechteck anstelle eines Quadrats.
# einer der Ecken reisst aus.
# versuch, die entzerrungsdaten nicht zu laden bei den Kamerdaten



SWITCH_UNDISTORT = True  # kamera korrektur nicht ausführen --> schneller
SWITCH_VERBOSE = False

# Bild vom Zug laden
sbbL_gray = cv2.imread("sbb/13L.png", cv2.IMREAD_GRAYSCALE)
sbbR_gray = cv2.imread("sbb/13R.png", cv2.IMREAD_GRAYSCALE)

# Die Gitter Templates für die Groblokalisierung werden immer vom Bild 13 geladen
templateL_gray = cv2.imread("sbb/13L.png", cv2.IMREAD_GRAYSCALE)
templateR_gray = cv2.imread("sbb/13R.png", cv2.IMREAD_GRAYSCALE)

# L und R identische Abmessungen
h, w = sbbL_gray.shape[:2]
assert sbbL_gray.shape[:2] == sbbR_gray.shape[:2], "Bilder müssen gleich gross ein."
assert sbbR_gray.shape[:2] == templateL_gray.shape[:2], "Bilder müssen gleich gross ein."
assert templateL_gray.shape[:2] == templateR_gray.shape[:2], "Bilder müssen gleich gross ein."

# Interne Kamera Verzerrung korrigieren
cal = calibMatrix.CalibData()
if SWITCH_UNDISTORT:
    calibmtxL, roiL = cv2.getOptimalNewCameraMatrix(cal.kl, cal.drl, (w, h), 1, (w, h))
    sbbL_gray = cv2.undistort(sbbL_gray, cal.kl, cal.drl, None, calibmtxL)
    templateL_gray = cv2.undistort(templateL_gray, cal.kl, cal.drl, None, calibmtxL)

    calibmtxR, roiR = cv2.getOptimalNewCameraMatrix(cal.kr, cal.drr, (w, h), 1, (w, h))
    sbbR_gray = cv2.undistort(sbbR_gray, cal.kr, cal.drr, None, calibmtxR)
    templateR_gray = cv2.undistort(templateR_gray, cal.kr, cal.drr, None, calibmtxR)

# Bilder kopieren, umwandeln
sbbL_rgb = cv2.cvtColor(sbbL_gray, cv2.COLOR_GRAY2BGR)
sbbR_rgb = cv2.cvtColor(sbbR_gray, cv2.COLOR_GRAY2BGR)
imgL = sbbL_gray.copy()
imgR = sbbR_gray.copy()
templateL = cv2.cvtColor(templateL_gray, cv2.COLOR_GRAY2BGR)
templateR = cv2.cvtColor(templateR_gray, cv2.COLOR_GRAY2BGR)

# GROBE LOKALISIERUNG DES GITTERS
# Perspektivische Verzerrung korrigieren, Vogelperspektive:
# bessere Messung mit Hilfe der 4 Schrauben des Gitters
pts1L = np.float32([[1110, 1111], [2376, 814], [1529, 1881], [2850, 1557]])
pts1R = np.float32([[1715, 820], [3010, 1004], [1357, 1591], [2712, 1795]])



#Kontrolle der Punkte
colorcube = [(217, 39, 240), (240, 230, 39), (62, 240, 39), (240, 39, 62)]
for i in range(0, 4):
    checkL = cv2.drawMarker(sbbL_rgb, (pts1L[i][0], pts1L[i][1]), colorcube[i], cv2.MARKER_CROSS, 100, 10)
    checkR = cv2.drawMarker(sbbR_rgb, (pts1R[i][0], pts1R[i][1]), colorcube[i], cv2.MARKER_CROSS, 100, 10)


# cv2.namedWindow("checkL", cv2.WINDOW_NORMAL)
# cv2.namedWindow("checkR", cv2.WINDOW_NORMAL)
# cv2.imshow("checkL", checkL)
# cv2.imshow("checkR", checkR)
# cv2.waitKey(0)


# Die Leinwand für das transformierte Bild wird so gross
canvas = (1500, 4000)

# Der erste Punkt kommt auf der Leinwand auf (ofsx, ofsy)
ofsxL, ofsyL = 250, 1500
ofsxR, ofsyR = 250, 1400  # TODO: stimmt das auch für die rechte Seite??

# Skalieren des transformierten Bildes
# der Abstand zwischen den Eckpunkten im Zielbild beträgt d1
d1 = 1000
pts2L = np.float32([[ofsxL, ofsyL], [ofsxL + d1, ofsyL], [ofsxL, ofsyL + d1], [ofsxL + d1, ofsyL + d1]])
pts2R = np.float32([[ofsxR, ofsyR], [ofsxR + d1, ofsyR], [ofsxR, ofsyR + d1], [ofsxR + d1, ofsyR + d1]])

# Transformation am Bild durchführen
ML = cv2.getPerspectiveTransform(pts1L, pts2L)
ML_inv = np.linalg.inv(ML)
imgL = cv2.warpPerspective(imgL, ML, canvas, borderMode=cv2.BORDER_TRANSPARENT)

MR = cv2.getPerspectiveTransform(pts1R, pts2R)
MR_inv = np.linalg.inv(MR)
imgR = cv2.warpPerspective(imgR, MR, canvas, borderMode=cv2.BORDER_TRANSPARENT)

# Auf der gewählten Canvas-Grösse bleiben zwei Ecken schwarz.
# Dort können können keine Gitterstrukturen gefunden werden,
# dies verfälscht die Suche nach dem Gittermittelpunkt unnötig.
# Die leeren Ecken werden mit der gespiegelten Seite ersetzt
partialL = imgL.copy()
mask = imgL == 0
mirror = cv2.flip(imgL, 1)
imgL[mask] = mirror[mask]

partialR = imgR.copy()
mask = imgR == 0
mirror = cv2.flip(imgR, 1)
imgR[mask] = mirror[mask]

# plt.subplot(141), plt.imshow(sbbL_gray), plt.title('Photo')
# plt.subplot(141), plt.imshow(partialL), plt.title('L warpPerspective')
# plt.subpl# ot(142), plt.imshow(imgL), plt.title('L warpPerspective + mirror')
# plt.subplot(143), plt.imshow(partialR), plt.title('R warpPerspective')
# plt.subplot(144), plt.imshow(imgR), plt.title('R warpPerspective + mirror')

# Vorverarbeitung, nur interessante Frequenzen behalten (8 pixel breite gitter)
# DoG: Difference of Gauss, Werte experimentell bestimmt.
dogband = 12
fpoint = 34
g1 = cv2.getGaussianKernel(49, fpoint + 0.5 * dogband)
g2 = cv2.getGaussianKernel(49, fpoint - 0.5 * dogband)
g3 = (g1 - g2) * 225
imgL = cv2.sepFilter2D(imgL, -1, g3, g3)
imgR = cv2.sepFilter2D(imgR, -1, g3, g3)

# threshold --> was nicht max. weiss ist, wird schwarz gesetzt.
th, imgL = cv2.threshold(imgL, 254, 255, cv2.THRESH_BINARY)
th, imgR = cv2.threshold(imgR, 254, 255, cv2.THRESH_BINARY)

# Summieren nach Zeile
freqHitsL = imgL.sum(1)  # Zeilenweise
freqHitsL = freqHitsL.astype(np.float64)

freqHitsR = imgR.sum(1)  # Zeilenweise
freqHitsR = freqHitsR.astype(np.float64)

# glätten
# 21:29 -- 8.11.2018 :  k = cv2.getGaussianKernel(1331, 333): Abgeschnittene Gitter weichen ab
k = cv2.getGaussianKernel(1101, 300)
freqHitsSmoothL = cv2.filter2D(freqHitsL, -1, k)
gitterposL = freqHitsSmoothL.argmax()

freqHitsSmoothR = cv2.filter2D(freqHitsR, -1, k)
gitterposR = freqHitsSmoothR.argmax()

# plt.figure(2)  # TODO : Triggert eine warnung
# plt.subplot(121), plt.plot(freqHitsL), plt.title('L')
# plt.subplot(121), plt.plot(freqHitsSmoothL)
# plt.subplot(122), plt.plot(freqHitsR), plt.title('R')
# plt.subplot(122), plt.plot(freqHitsSmoothR)
# plt.show()

# Farbige Markierungen setzen
imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
imgL = cv2.circle(imgL, (int(canvas[0] / 2), gitterposL), 25, (255, 0, 255), -1)
# cv2.namedWindow('DoG Left', cv2.WINDOW_NORMAL)
# cv2.imshow("DoG Left", imgL)

imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2RGB)
imgR = cv2.circle(imgR, (int(canvas[0] / 2), gitterposR), 25, (255, 0, 255), -1)
# cv2.namedWindow('DoG Right', cv2.WINDOW_NORMAL)
# cv2.imshow("DoG Right", imgR)

# gefundene vertikale Position in opencv Punkt x y wandeln
gitterL = np.array([[[canvas[0] / 2, gitterposL]]], dtype=np.float32)
gitterL = cv2.perspectiveTransform(gitterL, ML_inv)

gitterR = np.array([[[canvas[0] / 2, gitterposR]]], dtype=np.float32)
gitterR = cv2.perspectiveTransform(gitterR, MR_inv)

cv2.destroyAllWindows()

# Perspektivisch korrigiertes Template erstellen
# TODO: camera intrinics !?

# der Abstand zwischen den Eckpunkten im Zielbild beträgt d1
# Die Leinwand für das transformierte Bild wird so gross
d1 = 2000
canvas = (d1 + 600, d1 + 600)

# eckpunkte für eine exaktere perspektivische Korrektur laden
pts1L = eckpunkte(1964, 0)
print(pts1L)

# Der erste Punkt kommt auf der Leinwand auf (ofsx, ofsy)
ofsxL, ofsyL = 300, 300
ofsxR, ofsyR = 0, 0  # TODO: stimmt das auch für die rechte Seite??

# Skalieren des transformierten Templates
pts2L = np.float32([[ofsxL, ofsyL], [ofsxL + d1, ofsyL], [ofsxL, ofsyL + d1], [ofsxL + d1, ofsyL + d1]])
pts2R = np.float32([[ofsxR, ofsyR], [ofsxR + d1, ofsyR], [ofsxR, ofsyR + d1], [ofsxR + d1, ofsyR + d1]])
print(pts2L)

ML = cv2.getPerspectiveTransform(pts1L, pts2L)
ML_inv = np.linalg.inv(ML)
templateL = cv2.warpPerspective(templateL, ML, canvas, borderMode=cv2.BORDER_TRANSPARENT)
# cv2.namedWindow("warpedTemplate", cv2.WINDOW_NORMAL)
# cv2.imshow("warpedTemplate", templateL)
# cv2.imwrite("warpedTemplate13L.png", templateL)

# Perspektivisch korrigiertes Bild erstellen
# TODO: camera intrinics !?

# der Abstand zwischen den Eckpunkten im Zielbild beträgt d1
# Die Leinwand für das transformierte Bild wird so gross
border = 400
canvas = (d1 + 2 * border, d1 + 2 * border)

# eckpunkte für eine exaktere perspektivische Korrektur laden
pts1L = eckpunkte(gitterL[0][0][0], 0)

# Der erste Punkt kommt auf der Leinwand auf (ofsx, ofsy)
ofsxL, ofsyL = border, border
ofsxR, ofsyR = border, border

# Skalieren des transformierten Templates
pts2L = np.float32([[ofsxL, ofsyL], [ofsxL + d1, ofsyL], [ofsxL, ofsyL + d1], [ofsxL + d1, ofsyL + d1]])
pts2R = np.float32([[ofsxR, ofsyR], [ofsxR + d1, ofsyR], [ofsxR, ofsyR + d1], [ofsxR + d1, ofsyR + d1]])

ML = cv2.getPerspectiveTransform(pts1L, pts2L)
ML_inv = np.linalg.inv(ML)
matchL = cv2.warpPerspective(sbbL_rgb, ML, canvas, borderMode=cv2.BORDER_TRANSPARENT)


# TRIANGULATION DER ECKPUNKTE UND DES GITTERMITTELPUNKT 3-D
# Punkte müssen im 2xN Format sein und float: l = np.array([[ 304],[ 277]], dtype=np.float)
# https://stackoverflow.com/questions/46163831/output-3d-points-change-everytime-in-triangulatepoints-using-in-python-2-7-with
# in "10triangulation2.py" funktioniert es hingegen mit shape (1, 1, 2) :
# pts = [[[1460.8145   475.48917]]]

# TEST-DATEN ECKEN plus MITTELPUNKT (UZS)
pts1L = np.float32([[[1110, 1111], [2376, 814], [2850, 1557], [1529, 1881], [1110, 1111]]])
pts1R = np.float32([[[1715, 820], [3010, 1004], [2712, 1795], [1357, 1591], [1715, 820]]])


# TEST-DATEN ECKEN plus MITTELPUNKT (UZS) X/Y SWAP TEST
pts1L = np.float32([[[1111, 1110], [814, 2376], [1557, 2850], [1881, 1529], [1111, 1110]]])
pts1R = np.float32([[[820, 1715], [1004, 3010], [1795, 2712], [1591, 1357], [820, 1715]]])

# TEST-DATEN ECKEN plus MITTELPUNKT (UZS) MIT FORMAT WIE SO: https://stackoverflow.com/questions/46163831 GEHT NICHT
pts1L = np.array([[[1110], [1111]], [[2376],  [814]], [[2850], [1557]], [[1529], [1881]], [[1110], [1111]]], dtype=np.float)
pts1R = np.array([[[1715],  [820]], [[3010], [1004]], [[2712], [1795]], [[1357], [1591]], [[1715],  [820]]], dtype=np.float)


# VIELLEICHT MIT EINEM EINZELNEN PUNKT: MIT FORMAT WIE SO: https://stackoverflow.com/questions/46163831 GEHT NICHT
pts1L = np.array([[1110], [1111]], dtype=np.float)
pts1R = np.array([[1715],  [820]], dtype=np.float)

pts1L = np.array([[2376],  [814]], dtype=np.float)
pts1R = np.array([[3010], [1004]], dtype=np.float)

pts1L = np.array([[2850], [1557]], dtype=np.float)
pts1R = np.array([[2712], [1795]], dtype=np.float)

pts1L = np.array([[1529], [1881]], dtype=np.float)
pts1R = np.array([[1357], [1591]], dtype=np.float)



# TEST-DATEN mit y invertiert --> Resultat: Ändert nichts an der Form des möchtegern "Quadrats"
ih = 3000
pts1L = np.float32([[[1110, ih - 1111], [2376, ih - 814], [2850, ih - 1557], [1529, ih - 1881], [1110, ih - 1111]]])
pts1R = np.float32([[[1715, ih - 820], [3010, ih - 1004], [2712, ih - 1795], [1357, ih - 1591], [1715, ih - 820]]])

# 3000-y und  X/Y SWAP TEST --> Resultat : ändern nicht viel an der Form
pts1L = np.float32([[[ih-1111, 1110], [ih-814, 2376], [ih-1557, 2850], [ih-1881, 1529], [ih-1111, 1110]]])
pts1R = np.float32([[[ih-820, 1715], [ih-1004, 3010], [ih-1795, 2712], [ih-1591, 1357], [ih-820, 1715]]])


# TEST-DATEN ECKEN
pts1L = np.float32([[[1110, 1111], [2376, 814], [2850, 1557], [1529, 1881], [1110, 1111]]])
pts1R = np.float32([[[1715, 820], [3010, 1004], [2712, 1795], [1357, 1591], [1715, 820]]])


# im format gemäss pythonpath.wordpress.com
pts1L = np.float32([[1110, 2376, 2850, 1529, 1110],
                    [1111,  814, 1557, 1881, 1111]])

pts1R = np.float32([[1715, 3010, 2712, 1357, 1715],
                    [820, 1004, 1795, 1591, 820]])






pt3d = cv2.triangulatePoints(cal.pl, cal.pr, pts1L, pts1R)


print("Triangulation 3d pt:", pt3d)
fn = "tmp/3dpoints-Ecken"


test = cv2.convertPointsFromHomogeneous(pt3d.T)
print("TEST: ")
print(test)

# pt3d = pt3d[:-1] / pt3d[-1]  # https://pythonpath.wordpress.com/import-cv2/
# pt3d = pt3d / np.max(pt3d)
# np.save(fn + ".npy", pt3d.T)
# np.savetxt(fn + ".asc", pt3d.T, "%10.8f")
# pt3d = cv2.convertPointsFromHomogeneous(pt3d.T)


# Remember to divide out the 4th row. Make it homogeneous
pt3d /= pt3d[3]
X = pt3d
# Recover the origin arrays from PX
x1 = np.dot(cal.pl, X)
x2 = np.dot(cal.pr, X)
# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]
print('X\n', X)
print('x1\n', x1)
print('x2\n', x2)

# pt3d = pt3d.T
fig = plt.figure()
ax = Axes3D(fig)
X, Y, Z = pt3d[0], pt3d[1], pt3d[2]
#X, Y, Z = pt3d[0][0], pt3d[1][0], pt3d[2][0]
ax.plot(X, Y, Z)


#weil ax scaled nicht funktioniert bei 3d plot, dann sieh aber das rechteck nicht mehr nach rechteck aus ?!?!!
set_axes_equal(ax)
plt.show()

# exit(0)

# Template matching
# templateL = cv2.Canny(templateL, threshold1=50, threshold2=90)
# matchL = cv2.Canny(matchL, threshold1=50, threshold2=90)
w, h = d1, d1

# Nur den besten Match behalten
resultL = cv2.matchTemplate(matchL, templateL, cv2.TM_CCOEFF)

print(resultL.shape)
loc = np.where(resultL >= resultL.max())

# bounding Box mit Diagonalen zeichnen
pt = None
for pt in zip(*loc[::-1]):
    cv2.rectangle(matchL, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
    cv2.line(matchL, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
    cv2.line(matchL, (pt[0] + w, pt[1]), (pt[0], pt[1] + h), (0, 255, 0), 1)

cv2.namedWindow("warpedMatch", cv2.WINDOW_NORMAL)
cv2.imshow("warpedMatch", matchL)

# plt.figure(3)
# plt.imshow(resultL, cmap='gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.show()

# auf Original Foto einzeichnen
sbbL = cv2.circle(sbbL_rgb, (gitterL[0][0][0], gitterL[0][0][1]), 25, (255, 0, 255), -1)
cv2.namedWindow('sbb L', cv2.WINDOW_NORMAL)
cv2.imshow("sbb L", sbbL)

sbbR = cv2.circle(sbbR_rgb, (gitterR[0][0][0], gitterR[0][0][1]), 25, (255, 0, 255), -1)
cv2.namedWindow('sbb R', cv2.WINDOW_NORMAL)
cv2.imshow("sbb R", sbbR)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Mittelpunkt
print("Gittermittpunkt L)")
print(gitterL)
print("\nGittermittpunkt R)")
print(gitterR)
