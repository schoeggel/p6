import cv2
from matplotlib import pyplot as plt

# Kleineres Template suchen
# Funktioniert dann aber nicht bei allen Bildern, bspw den neuen von Jan nicht.

# Bild vom Zug laden
# gray_img = cv2.imread("SBB/13L.png",  cv2.IMREAD_GRAYSCALE)
gray_img = cv2.imread("SBB/4-OK1L.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
img = cv2.Canny(img, threshold1=60, threshold2=110,)

# Objekt für Template matching laden
template = cv2.imread("data/05templateSmall1.png", cv2.IMREAD_GRAYSCALE)
template = cv2.Canny(template, threshold1=60, threshold2=110,)
w, h = template.shape[::-1]   # w,h und r,c sind vertauscht

# Blur
# k = 50
# template = cv2.blur(template, (k, k))
# img = cv2.blur(img, (k*2, k*2))

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
reltx, relty  = 78, 55
gittermitte = (top_left[0]+reltx, top_left[1]+relty)

print(gittermitte)


veri = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
veri = cv2.circle(veri, gittermitte, 10, (255, 255, 0), 1)

# roi = im[y1:y2, x1:x2]
s = 100
roi = veri[top_left[1]-s:bottom_right[1]+s,top_left[0]-s:bottom_right[0]+s]

# Anzeigen
cv2.namedWindow('verification', cv2.WINDOW_NORMAL)
cv2.imshow('verification', roi)
# cv2.resizeWindow('image', int(w/1.5), int(h/1.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
