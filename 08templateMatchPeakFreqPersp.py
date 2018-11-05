import cv2
import numpy as np
from cvaux import cvaux
from matplotlib import pyplot as plt
from tqdm import tqdm


# das sbb-Bild wird so verarbeitet:
#     1. möglichst nur die kanten des runden Gitters angezeigt werden: DoG Filter
#     2. Maskieren und korrigieren: nur der Diagonale Streifen, auf dem des Gitter erwartet wird verwenden
#     3. Summieren pro Zeile
#     4. Glätten mit der richtigen Grösse, damit das Gitter zu EINEM Peak wird
#
#


# Bild vom Zug laden
gray_img = cv2.imread("data/NurGitterL.png",  cv2.IMREAD_GRAYSCALE)
gray_img = cv2.imread("sbb/4-OK1L.png",  cv2.IMREAD_GRAYSCALE)
# gray_img = cv2.imread("data/testDoG1.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)


# Perspektivische Korrektur
rows, cols, ch = img.shape
pts1 = np.float32([[377, 353], [2177,0], [1697, 3000], [3829, 2453]])
pts2 = np.float32([[0, 0], [2017, 0], [0, 2958], [2017,2958]])
M = cv2.getPerspectiveTransform(pts1, pts2)
img = cv2.warpPerspective(img, M, (2017, 2958))
plt.subplot(121), plt.imshow(gray_img), plt.title('Input')
plt.subplot(122), plt.imshow(img), plt.title('Output')

# Vorverarbeitung, nur interessante Frequenzen behalten (8 pixel breite gitter)
# DoG: Difference of Gauss
filterband = 12
filterpoint = 34
g1 = cv2.getGaussianKernel(49, filterpoint + 0.5*filterband)
g2 = cv2.getGaussianKernel(49, filterpoint - 0.5*filterband)
g3 = (g1-g2)*10
img = cv2.sepFilter2D(img, -1, g3, g3)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.equalizeHist(img)


#debug
# img = cv2.rectangle(img, (0, 0), (2000, 1000), 0, -1)
# img = cv2.rectangle(img, (100, 3000), (4000, 1100), (255, 255, 255), -1)


# threshold
th, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)




# Summieren nach Zeile

test = img.sum(1)   # Zeilenweise
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
img = cv2.circle(img, (1010, gitterposy), 15, (255, 0, 255), -1)
cv2.namedWindow('DoG', cv2.WINDOW_NORMAL)
cv2.imshow("DoG", img)
cv2.waitKey(0)
cv2.destroyAllWindos()