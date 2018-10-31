import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
# Versuch, möglichst genau das Zentrum des Gitters zu finden.
# Es wird ein Bild verwendet, auf dem nur das Gitter zu sehen ist.
# Ellipsen finden
# Mittelpunkte der (gültigen) Ellipsen mitteln --> Mittelpunkt
# Code Sample von http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
#
# 31.10.2019: Ellipsen finden dauert viel zu lange.
# ==> Versuch abgebrochen. Implementierung einer verbesserten Ellipsen-erkennung zu aufwändig.

gray_img = cv2.imread("data/nurGitterLs2.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# template = cv2.imread("data/pattern-g1.png", cv2.IMREAD_GRAYSCALE)
# w, h = template.shape[::-1]   # w,h und r,c sind vertauscht

#
# result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
# loc = np.where(result >= result.max())
#
#
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
#     cv2.line(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
#     cv2.line(img, (pt[0]+w, pt[1]), (pt[0], pt[1] + h), (0, 255, 0), 1)
#
# print(pt[0])  # 1055
# print(pt[1])  # 811
# print(w)      # 1818
# print(h)      # 1094
#
# # roi  = img[811  :     811+1094, 1055    :   1055+1818]
# roi = img[pt[1]: pt[1] + h, pt[0]: pt[0] + w]
#
#
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', roi)
# cv2.resizeWindow('image', int(w/1.5), int(h/1.5))

# Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420]
# image_gray = color.rgb2gray(image_rgb)
print("canny...")
edges = canny(gray_img, sigma=2.0,
              low_threshold=30, high_threshold=50)

edgescv2 = color.gray2rgb(img_as_ubyte(edges))

# edgescv2 = cv2.ellipse(edgescv2, (350,210), (250,150),-4,0,360,(0,0,0),-1)



# canvas = np.zeros((edges.shape[0],edges.shape[1], 1), np.uint8)
# Fill image with red color(set each pixel to red)
# canvas[:] = 255
# edgescv2 = canvas[edges]
# edgescv2 = cv2.cvtColor(edgescv2, cv2.COLOR_GRAY2RGB)
cv2.imshow('image', edgescv2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
print("hough...")
result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=100, max_size=120)
result.sort(order='accumulator')

# Estimated parameters for the ellipse
print("estimate parameters...")
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
img[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(img)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
