import cv2
import numpy as np
from cvaux import cvaux
from matplotlib import pyplot as plt


# Bild vom Zug laden
gray_img = cv2.imread("SBB/14L.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
img = cv2.Canny(img, threshold1=60, threshold2=110,)


# Objekt für Template matching laden
template = cv2.imread("data/05GitterLCanny1.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]   # w,h und r,c sind vertauscht

res_CCOEFF = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
res_CCORR = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
res_SQDIFF = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)



plt.subplot(221), plt.imshow(res_CCOEFF, cmap='gray')
plt.title('template matching score (TM_CCOEFF_NORMED)'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(res_CCORR, cmap='gray')
plt.title('template matching score (TM_CCORR_NORMED)'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(res_SQDIFF, cmap='gray')
plt.title('template matching score (TM_SQDIFF)'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(gray_img, cmap='gray')
plt.title('Original Photo'), plt.xticks([]), plt.yticks([])


plt.show()