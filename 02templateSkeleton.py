import cv2
import numpy as np
import cvaux

# Bild vom Zug laden
gray_img = cv2.imread("data/canny3.png",  cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# Objekt für Template matching laden
template = cv2.imread("data/sktemplate2.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]   # w,h und r,c sind vertauscht

# Nur den besten Match behalten
result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF)

print(result.shape)
loc = np.where(result >= result.max())


# bounding Box mit Diagonalen zeichnen
pt = None
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
    cv2.line(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
    cv2.line(img, (pt[0]+w, pt[1]), (pt[0], pt[1] + h), (0, 255, 0), 1)

print(pt[0])  # 1055
print(pt[1])  # 811
print(w)      # 1818
print(h)      # 1094

# roi  = img[811  :     811+1094, 1055    :   1055+1818]
roi = img[pt[1]: pt[1] + h, pt[0]: pt[0] + w]


# test
cv2.imshow('test12', img)
cv2.imshow('test1242342', roi)


# gefärbter Ausschnitt vom Original (inkl Rechteck) überlagern mit Template
overlay = cvaux.mixtochannel(roi, template)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', overlay)
# cv2.resizeWindow('image', int(w/1.5), int(h/1.5))


cv2.waitKey(0)
cv2.destroyAllWindows()
