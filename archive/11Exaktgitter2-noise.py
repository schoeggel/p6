import cv2
import numpy as np
from matplotlib import pyplot as plt
from archive import calibMatrix
from archive.GitterEckpunkte import eckpunkte

# das sbb-Bild wird so verarbeitet:
# unter der Annahme, dass sich das Gitter in der Bildmitte befindet wird eine
# grobe perspektivische Verformung  durchgeführt. In dieser wird nach den
# Gitterstrukturen gesucht uns so eine Groblokalisierung des Gitters möglich.
# Die Funktion "eckpunkte" liefert zu diesem ungefähren Gittermittelpunkt
# die Gittereckpunkte (Die vier Schrauben) zurück somit kann das Bild optimal
# perspektivisch korrigiert werden. Damit wird eine genaue Lokalisierung mittels
# Template matching möglich.

# was passiert, wenn ein Teil des tmeplates mit rauschen verdeckt wird? ==> Kein Probmlem.

SWITCH_UNDISTORT = True  # kamera korrektur nicht ausführen --> schneller
SWITCH_VERBOSE = False

# Bild vom Zug laden
sbbL_gray = cv2.imread("sbb/4-OK1L.png", cv2.IMREAD_GRAYSCALE)
sbbR_gray = cv2.imread("sbb/16R.png", cv2.IMREAD_GRAYSCALE)

# Die Gitter Templates für die Groblokalisierung werden immer vom Bild 13 geladen
templateL_gray = cv2.imread("sbb/13L-noise.png", cv2.IMREAD_GRAYSCALE)
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
plt.subplot(141), plt.imshow(partialL), plt.title('L warpPerspective')
plt.subplot(142), plt.imshow(imgL), plt.title('L warpPerspective + mirror')
plt.subplot(143), plt.imshow(partialR), plt.title('R warpPerspective')
plt.subplot(144), plt.imshow(imgR), plt.title('R warpPerspective + mirror')

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

# glätten (Filter Paramter wurden experimentell bestimmt)
# 21:29 -- 8.11.2018 :  k = cv2.getGaussianKernel(1331, 333): Abgeschnittene Gitter weichen ab
k = cv2.getGaussianKernel(1101, 300)
freqHitsSmoothL = cv2.filter2D(freqHitsL, -1, k)
gitterposL = freqHitsSmoothL.argmax()

freqHitsSmoothR = cv2.filter2D(freqHitsR, -1, k)
gitterposR = freqHitsSmoothR.argmax()

plt.figure(2)  # TODO : Triggert eine warnung
plt.subplot(121), plt.plot(freqHitsL), plt.title('L')
plt.subplot(121), plt.plot(freqHitsSmoothL)
plt.subplot(122), plt.plot(freqHitsR), plt.title('R')
plt.subplot(122), plt.plot(freqHitsSmoothR)
# plt.show()

# Farbige Markierungen setzen
imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
imgL = cv2.circle(imgL, (int(canvas[0] / 2), gitterposL), 25, (255, 0, 255), -1)
cv2.namedWindow('DoG Left', cv2.WINDOW_NORMAL)
cv2.imshow("DoG Left", imgL)

imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2RGB)
imgR = cv2.circle(imgR, (int(canvas[0] / 2), gitterposR), 25, (255, 0, 255), -1)
cv2.namedWindow('DoG Right', cv2.WINDOW_NORMAL)
cv2.imshow("DoG Right", imgR)

# gefundene vertikale Position in opencv Punkt x y wandeln
gitterL = np.array([[[canvas[0] / 2, gitterposL]]], dtype=np.float32)
gitterL = cv2.perspectiveTransform(gitterL, ML_inv)

gitterR = np.array([[[canvas[0] / 2, gitterposR]]], dtype=np.float32)
gitterR = cv2.perspectiveTransform(gitterR, MR_inv)

cv2.destroyAllWindows()

# Perspektivisch korrigiertes Template erstellen
# Hier wird das Gitter zuerst aus dem Bild extrahiert und "normalisiert",
# indem es in die Vogelperspektive "gewarpt" wird .
# Das Bild des zugs wird dann ebenfalls in die Vogelperspektive gewarpt.
# Das Template matching erfolgt aus der "Vogelperspektive", die Koordinaten werden
# zurückgerechnet

# TODO: camera intrinics !?

# der Abstand zwischen den Eckpunkten im Zielbild beträgt d1
# Die Leinwand für das transformierte Bild wird so gross
d1 = 1000
canvas = (d1, d1)

# eckpunkte für eine exaktere perspektivische Korrektur laden
pts1L = eckpunkte(1964, 0)
print(pts1L)

# Der erste Punkt kommt auf der Leinwand auf (ofsx, ofsy)
ofsxL, ofsyL = 0, 0
ofsxR, ofsyR = 0, 0  # TODO: stimmt das auch für die rechte Seite??

# Skalieren des transformierten Templates
pts2L = np.float32([[ofsxL, ofsyL], [ofsxL + d1, ofsyL], [ofsxL, ofsyL + d1], [ofsxL + d1, ofsyL + d1]])
pts2R = np.float32([[ofsxR, ofsyR], [ofsxR + d1, ofsyR], [ofsxR, ofsyR + d1], [ofsxR + d1, ofsyR + d1]])
print(pts2L)

ML = cv2.getPerspectiveTransform(pts1L, pts2L)
ML_inv = np.linalg.inv(ML)
templateL = cv2.warpPerspective(templateL, ML, canvas, borderMode=cv2.BORDER_TRANSPARENT)
cv2.namedWindow("warpedTemplate", cv2.WINDOW_NORMAL)
cv2.imshow("warpedTemplate", templateL)

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

plt.figure(3)
plt.imshow(resultL, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.show()




# auf Original Foto einzeichnen
sbbL = cv2.circle(sbbL_rgb, (gitterL[0][0][0], gitterL[0][0][1]), 25, (255, 0, 255), -1)
cv2.namedWindow('sbb L', cv2.WINDOW_NORMAL)
cv2.imshow("sbb L", sbbL)

sbbR = cv2.circle(sbbR_rgb, (gitterR[0][0][0], gitterR[0][0][1]), 25, (255, 0, 255), -1)
cv2.namedWindow('sbb R', cv2.WINDOW_NORMAL)
cv2.imshow("sbb R", sbbR)

cv2.waitKey(0)
cv2.destroyAllWindows()
