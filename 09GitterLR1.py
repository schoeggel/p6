import cv2
import numpy as np
from matplotlib import pyplot as plt
import calibMatrix

# das sbb-Bild wird so verarbeitet:
#     0. Kamera Kalibrier Daten verwenden um das Bild zu entzerren
#     1. möglichst nur die kanten des runden Gitters angezeigt werden: DoG Filter
#     2. Maskieren und korrigieren: nur der Diagonale Streifen, auf dem des Gitter erwartet wird verwenden
#     --> TODO: die angeschnittenen Dreiecke: Bild spiegeln
#     3. Summieren pro Zeile
#     4. Glätten mit der richtigen Grösse, damit das Gitter zu EINEM Peak wird
#

SWITCH_UNDISTORT = True    # kamera korrektur nicht ausführen --> schneller
SWITCH_VERBOSE = False


# Bild vom Zug laden
sbb_gray = cv2.imread("sbb/13L.png", cv2.IMREAD_GRAYSCALE)
sbb_rgb = cv2.cvtColor(sbb_gray, cv2.COLOR_GRAY2BGR)
img = sbb_gray.copy()

h, w = img.shape[:2]

# Interne Kamera Verzerrung korrigieren
if SWITCH_UNDISTORT:
    cal = calibMatrix.CalibData()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal.kl, cal.drl, (w, h), 1, (w, h))
    img = cv2.undistort(img, cal.kl, cal.drl, None, newcameramtx)
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(dst)
    # plt.show()
    # TODO: img = dst und abmessungen kontrollieren
    # cv2.imwrite("test-undistort-14L.png", dst)
    # exit(0)




# Perspektivische Verzerrung korrigieren, Vogelperspektive:

# bessere Messung mit Hilfe der 4 Schrauben des Gitters
pts1 = np.float32([[1110, 1111], [2376, 814], [1529, 1881], [2850, 1557]])

# Die Leinwand für das transformierte Bild wird so gross
canvas = (1500, 4000)

# Der erste Punkt kommt auf der Leinwand auf (ofsx, ofsy)
ofsx, ofsy = 250, 1500

# Skalieren des transformierten Bildes
# der Abstand zwischen den Eckpunkten im Zielbild beträgt d1
d1 = 1000
pts2 = np.float32([[ofsx, ofsy], [ofsx+d1, ofsy], [ofsx, ofsy+d1], [ofsx+d1, ofsy+d1]])

# Transformation am Bild durchführen
M = cv2.getPerspectiveTransform(pts1, pts2)
M_inv = np.linalg.inv(M)
img = cv2.warpPerspective(img, M, canvas, borderMode=cv2.BORDER_TRANSPARENT)

# Auf der gewählten Canvas-Grösse bleiben zwei Ecken schwarz.
# Dort können können keine Gitterstrukturen gefunden werden,
# dies verfälscht die Suche nach dem Gittermittelpunkt unnötig.
# Die leeren Ecken werden mit der gespiegelten Seite ersetzt
partial = img.copy()
mask = img == 0
mirror = cv2.flip(img, 1)
img[mask] = mirror[mask]

plt.subplot(131), plt.imshow(sbb_gray), plt.title('Photo')
plt.subplot(132), plt.imshow(partial), plt.title('warpPerspective')
plt.subplot(133), plt.imshow(img), plt.title('warpPerspective + mirror')


# Vorverarbeitung, nur interessante Frequenzen behalten (8 pixel breite gitter)
# DoG: Difference of Gauss, Werte experimentell bestimmt.
dogband = 12
fpoint = 34
g1 = cv2.getGaussianKernel(49, fpoint + 0.5 * dogband)
g2 = cv2.getGaussianKernel(49, fpoint - 0.5 * dogband)
g3 = (g1 - g2) * 225
img = cv2.sepFilter2D(img, -1, g3, g3)

# threshold --> was nicht max. weiss ist, wird schwarz gesetzt.
th, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)

# Summieren nach Zeile
freqHits = img.sum(1)  # Zeilenweise
freqHits = freqHits.astype(np.float64)
plt.figure(2)
plt.plot(freqHits)

# glätten
# 21:29 -- 8.11.2018 :  k = cv2.getGaussianKernel(1331, 333): Abgeschnittene Gitter weichen ab
k = cv2.getGaussianKernel(1101, 300)
freqHitsSmooth = cv2.filter2D(freqHits, -1, k)
gitterposy = freqHitsSmooth.argmax()
plt.plot(freqHitsSmooth)
plt.show()

# Farbige Markierungen setzen
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = cv2.circle(img, (int(canvas[0] / 2), gitterposy), 15, (255, 0, 255), -1)
cv2.namedWindow('DoG', cv2.WINDOW_NORMAL)
cv2.imshow("DoG", img)

# gefundene vertikale Position in opencv Punkt x y wandeln
gitterM = np.array([[[canvas[0] / 2, gitterposy]]], dtype=np.float32)
gitterM = cv2.perspectiveTransform(gitterM, M_inv)
print(gitterM)

# auf Original Foto einzeichnen
sbb = cv2.circle(sbb_rgb, (gitterM[0][0][0], gitterM[0][0][1]), 15, (255, 0, 255), -1)
cv2.namedWindow('sbb', cv2.WINDOW_NORMAL)
cv2.imshow("sbb", sbb)

cv2.waitKey(0)
cv2.destroyAllWindows()
