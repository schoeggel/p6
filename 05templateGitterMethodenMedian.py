import cv2
from matplotlib import pyplot as plt

# Problem: matching result bildet statt ein Punkt-Maximum ein Ring an Maxima
# Test: Lässt sich vielleicht lösen, wenn die Bilder geglättet werden ?

# Bild vom Zug laden
gray_img = cv2.imread("SBB/14L.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
img = cv2.Canny(img, threshold1=60, threshold2=110,)


# Objekt für Template matching laden
template = cv2.imread("data/05GitterBinary05.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]   # w,h und r,c sind vertauscht

# Blur
km = 5
kb = 100
kb = (kb, kb)

# template = cv2.blur(template, kb)
# template = cv2.medianBlur(template, km)
img = cv2.blur(img, kb)
img = cv2.medianBlur(img, km)


res_CCOEFF = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
res_CCORR = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
res_SQDIFF = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)



plt.subplot(221), plt.imshow(res_CCOEFF, cmap='gray')
plt.title('template matching score (TM_CCOEFF_NORMED)'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(res_CCORR, cmap='gray')
plt.title('template matching score (TM_CCORR_NORMED)'), plt.xticks([]), plt.yticks([])

if 1==1:
    plt.subplot(223), plt.imshow(res_SQDIFF, cmap='gray')
    plt.title('template matching score (TM_SQDIFF)'), plt.xticks([]), plt.yticks([])
else:
    plt.subplot(223), plt.imshow(template, cmap='gray')
    plt.title('tmpl'), plt.xticks([]), plt.yticks([])


plt.subplot(224), plt.imshow(img, cmap='gray')
plt.title('img'), plt.xticks([]), plt.yticks([])


plt.show()