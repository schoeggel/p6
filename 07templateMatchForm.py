import cv2
import numpy as np
from matplotlib import pyplot as plt

# Als template wir eine binäre Ellipse verwendet
# das sbb-Bild wird so verarbeitet:
#     1. möglichst nur die kanten des runden Gitters angezeigt werden.
#     2. versucht wird, die kontur zu schliessen
#     3. Kontur füllen
#     4. template matchen

# Abgebrochen !!!


# Bild vom Zug laden
gray_img = cv2.imread("data/NurGitterL.png",  cv2.IMREAD_GRAYSCALE)
gray_img = cv2.imread("sbb/4-OK1L.png",  cv2.IMREAD_GRAYSCALE)
# sbb_gray = cv2.imread("data/testDoG1.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# Vorverarbeitung, nur interessante Frequenzen behalten (8 pixel breite gitter)
# DoG: Difference of Gauss
filterband = 13
filterpoint = 36
g1 = cv2.getGaussianKernel(49, filterpoint + 0.5*filterband)
g2 = cv2.getGaussianKernel(49, filterpoint - 0.5*filterband)
g3 = (g1-g2)*10
img = cv2.sepFilter2D(img, -1, g3, g3)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.equalizeHist(img)

#threshold
old = img
th, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

# dilate
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
img = cv2.erode(img, kernel, iterations=2)
img = cv2.dilate(img, kernel, iterations=5)


#Grösstes partikel



cv2.namedWindow('DoG', cv2.WINDOW_NORMAL)
cv2.imshow("DoG", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)


img = cv2.Canny(img, threshold1=60, threshold2=110, )

# Objekt für Template matching laden
template = cv2.imread("data/05GitterLCanny1.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("data/05GitterBinary05.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]   # w,h und r,c sind vertauscht

# Blur
k = 50
template = cv2.blur(template, (k, k))
img = cv2.blur(img, (k*2, k*2))

# Template suchen
res_CCOEFF = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
res_CCORR = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
res_SQDIFF = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)


# Box für eine Methode einzeichnen
w, h = template.shape[::-1]
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_CCOEFF)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, 255, 15)


# Alles plotten
plt.subplot(221), plt.imshow(res_CCOEFF, cmap='gray')
plt.title('template matching score (TM_CCOEFF_NORMED)'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(res_CCORR, cmap='gray')
plt.title('template matching score (TM_CCORR_NORMED)'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(res_SQDIFF, cmap='gray')
plt.title('template matching score (TM_SQDIFF)'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(img, cmap='gray')
plt.title('Img'), plt.xticks([]), plt.yticks([])

plt.show()


# Zentrum einzeichnen
# Pixelpos des Gittermittelpunkts im Template:
reltx, relty  = 1104, 644
gittermitte = (top_left[0]+reltx, top_left[1]+relty)

print(gittermitte)


veri = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
veri = cv2.circle(veri, gittermitte, 10, (255, 255, 0), 1)

# roi = im[y1:y2, x1:x2]
s = -450
roi = veri[top_left[1]-s:bottom_right[1]+s,top_left[0]-s:bottom_right[0]+s]

# Anzeigen
cv2.namedWindow('verification', cv2.WINDOW_NORMAL)
cv2.imshow('verification', roi)
# cv2.resizeWindow('image', int(w/1.5), int(h/1.5))

cv2.namedWindow('src', cv2.WINDOW_NORMAL)
cv2.imshow('src', img)
cv2.resizeWindow('src', int(w/1.5), int(h/1.5))

cv2.waitKey(0)
cv2.destroyAllWindows()


# Quelle für select_largest_obj:
# https://github.com/lishen/dream2016_dm/blob/master/dm_preprocess.py
def select_largest_obj(self, img_bin, lab_val=255, fill_holes=False,
                       smooth_boundary=False, kernel_size=15):
    '''Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin (2D array): 2D numpy array of binary image.
        lab_val ([int]): integer value used for the label of the largest
                object. Default is 255.
        fill_holes ([boolean]): whether fill the holes inside the largest
                object or not. Default is false.
        smooth_boundary ([boolean]): whether smooth the boundary of the
                largest object using morphological opening or not. Default
                is false.
        kernel_size ([int]): the size of the kernel used for morphological
                operation. Default is 15.
    Returns:
        a binary image as a mask for the largest object.
    '''
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8,
                                         ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    # import pdb; pdb.set_trace()
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                      newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN,
                                        kernel_)
    return largest_mask