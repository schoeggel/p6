import cv2
import numpy as np
from matplotlib import pyplot as plt
import calibMatrix

# das sbb-Bild wird so verarbeitet:
#     0. Kamera Kalibrier Daten verwenden um das Bild zu entzerren
#     1. möglichst nur die kanten des runden Gitters angezeigt werden: DoG Filter
#     2. Maskieren und korrigieren: nur der Diagonale Streifen, auf dem des Gitter erwartet wird verwenden
#     3. Summieren pro Zeile
#     4. Glätten mit der richtigen Grösse, damit das Gitter zu EINEM Peak wird
#



# Bild vom Zug laden
gray_img = cv2.imread("sbb/13L.png", cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
h, w = img.shape[:2]

# Verzerrung korrigieren
cal = calibMatrix.CalibData()
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cal.kl, cal.drl, (w, h), 1, (w, h))
dst = cv2.undistort(img, cal.kl, cal.drl, None, newcameramtx)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
plt.show()

# cv2.imwrite("test-undistort-14L.png", dst)
# exit(0)

# TODO: img = dst und abmessungen kontrollieren
# crop the image
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]


# Perspektivische Korrektur
rows, cols, ch = img.shape

# Version 1: Alt
pts1 = np.float32([[377, 400], [2177, 0], [1697, 3000], [3829, 2453]])
pts2 = np.float32([[0, 0], [2017, 0], [0, 2958], [2017, 2958]])

# bessere Messung mit Hilfe der 4 Schrauben des Gitters TODO
pts1 = np.float32([[1110, 1111], [2376, 814], [1529, 1881], [2850, 1557]])
ptdist = np.array([np.linalg.norm(pts1[0]-pts1[1]),
                  np.linalg.norm(pts1[2]-pts1[3]),
                  np.linalg.norm(pts1[0]-pts1[2]),
                  np.linalg.norm(pts1[1]-pts1[3])])

# trsratio = ptdist[0]/ptdist[1]
trssize = (1500, 4000)
ofsx, ofsy = 250, 1500    # der Eckpunkt im Zielbild liegt auf (ofs,ofs)
d1 = 1000   # der Abstand zwischen den Eckpunkten im Zielbild beträgt d1
pts2 = np.float32([[ofsx, ofsy], [ofsx+d1, ofsy], [ofsx, ofsy+d1], [ofsx+d1, ofsy+d1]])
M = cv2.getPerspectiveTransform(pts1, pts2)
img = cv2.warpPerspective(img, M, trssize)
plt.subplot(121), plt.imshow(gray_img), plt.title('Input')
plt.subplot(122), plt.imshow(img), plt.title('Output: persp. warp')

# Vorverarbeitung, nur interessante Frequenzen behalten (8 pixel breite gitter)
# DoG: Difference of Gauss
dogband = 12
fpoint = 34
g1 = cv2.getGaussianKernel(49, fpoint + 0.5 * dogband)
g2 = cv2.getGaussianKernel(49, fpoint - 0.5 * dogband)
g3 = (g1 - g2) * 225


img = cv2.sepFilter2D(img, -1, g3, g3)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# debug
# plt.imshow(img)
# plt.show()
# img = cv2.rectangle(img, (0, 0), (2000, 1000), 0, -1)
# img = cv2.rectangle(img, (100, 3000), (4000, 1100), (255, 255, 255), -1)

# threshold --> was nicht max. weiss ist, wird schwarz gesetzt.
th, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)

# Summieren nach Zeile
test = img.sum(1)  # Zeilenweise
test = test.astype(np.float64)
plt.figure(2)
plt.plot(test)

# glätten
k = cv2.getGaussianKernel(1331, 333)
test2 = cv2.filter2D(test, -1, k)
gitterposy = test2.argmax()
print(gitterposy)

plt.plot(test2)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = cv2.circle(img, (int(trssize[0]/2), gitterposy), 15, (255, 0, 255), -1)
cv2.namedWindow('DoG', cv2.WINDOW_NORMAL)
cv2.imshow("DoG", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
