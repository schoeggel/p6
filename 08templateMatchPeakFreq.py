import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# Als template wir eine binäre Ellipse verwendet
# das sbb-Bild wird so verarbeitet:
#     1. möglichst nur die kanten des runden Gitters angezeigt werden: DoG Filter
#     2. Maskieren: nur der Diagonale Streifen, auf dem des Gitter erwartet wird verwenden
#     3. In der Diagonalen summieren
#     4. Glätten mit der richtigen Grösse, damit das Gitter zu EINEM Peak wird
#
#   Funktioniert grundsätzlich aber langsam. Sauberer wäre eine eprspektivische korrektur


cv2.namedWindow('debug', cv2.WINDOW_NORMAL)


def culumline(image, m=0, xtdist=-1):
    zcount = image[image > 128].sum()
    hh, ww = img.shape
    f'ww: {ww}'
    f'hh: {hh}'
    i = image.copy()
    if xtdist < 0:
        xtdist = hh // 4
    hits = np.zeros(hh + xtdist)
    ydelta = int(m * ww)

    for y1 in tqdm(range(0, hh + xtdist, 30)):
        y2 = y1 + ydelta
        i = cv2.line(i, (0, y1), (ww, y2), 0, 32)
        count = i[i > 128].sum()
        hits[y1] = zcount - count
        zcount = count
        #cv2.imshow("debug", i)
        #cv2.waitKey(1)

    return hits



# Bild vom Zug laden
gray_img = cv2.imread("data/NurGitterL.png",  cv2.IMREAD_GRAYSCALE)
gray_img = cv2.imread("sbb/3-OK1L.png",  cv2.IMREAD_GRAYSCALE)
# gray_img = cv2.imread("data/testDoG1.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

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

#threshold
old = img
th, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)


# nur diagonale verwenden
tri1 = np.array([[[0, 0], [0, 3000], [1600, 3000], [280, 0]]], dtype=np.int32)
tri2 = np.array([[[2200, 0], [4096, 2850], [4096, 0]]], dtype=np.int32)
img = cv2.fillPoly(img, tri1, 0)
img = cv2.fillPoly(img, tri2, 0)


test = culumline(img, m=-0.23)
plt.plot(test)


#cumuline glätten
k = cv2.getGaussianKernel(999, 100)
test2 = cv2.filter2D(test, -1, k)

plt.plot(test2*32)


plt.show()






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


