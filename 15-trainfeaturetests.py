# Tests mit Trainfeature Objekten
# Patch laden, Eckpunkte rechnen etc.

import cv2
import numpy as np
from trainfeature import Trainfeature
import findGrid


# Gitter
# Bilder laden
sbbL_gray = cv2.imread("sbb/13L.png", cv2.IMREAD_GRAYSCALE)
sbbR_gray = cv2.imread("sbb/13R.png", cv2.IMREAD_GRAYSCALE)

# Bilder kopieren, umwandeln #
sbbL_rgb = cv2.cvtColor(sbbL_gray, cv2.COLOR_GRAY2BGR)
sbbR_rgb = cv2.cvtColor(sbbR_gray, cv2.COLOR_GRAY2BGR)

# Test Kanalmanipulationen
# print(sbbL_rgb)
# print(sbbL_rgb.shape)
# sbbL_rgb[:,:,1]= 1# 80
# cv2.namedWindow('Unit test L', cv2.WINDOW_NORMAL)
# cv2.imshow("Unit test L", sbbL_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()


# Gitter suchen
gitterL, gitterR = findGrid.find(sbbL_gray, sbbR_gray, verbose=False)

# auf Original Foto einzeichnen
sbbL = cv2.circle(sbbL_rgb, (gitterL[0][0][0], gitterL[0][0][1]), 25, (255, 0, 255), -1)
sbbR = cv2.circle(sbbR_rgb, (gitterR[0][0][0], gitterR[0][0][1]), 25, (255, 0, 255), -1)

# Feature Tests
Trainfeature.loadmatrixp()
Trainfeature.approxreference(gitterL, gitterR)

xyz = np.array([-240, 240, 0])
s1 = Trainfeature("Gitterschraube1", xyz, 32)
print(s1)

#left, right = s1.reprojectedges()
#left = left.reshape((1,5,2)).astype(int)
#right = right.reshape((1,5,2)).astype(int)
# sbbL = cv2.polylines(sbbL, left, True, (0,255,255), 5, 1)
# sbbR = cv2.polylines(sbbR, right, True, (0,255,255), 5, 1)
# cv2.namedWindow('Unit test L', cv2.WINDOW_NORMAL)
# cv2.imshow("Unit test L", sbbL)
# cv2.namedWindow('Unit test R', cv2.WINDOW_NORMAL)
# cv2.imshow("Unit test R", sbbR)#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Schraube oben links
imgL, imgR = s1.warp()
center, val = s1.find(sbbL, sbbR, verbose=False)
print(f'center: {center}\nvalue: {val}\n')
sbbL = cv2.drawMarker(sbbL, center, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbL = s1.drawBasis(sbbL, sideLR=0, show=False, thickness=5)

# weitere Teile
xyz = np.array([-240, -240, 0])
s2 = Trainfeature("Gitterschraube1", xyz, 32)
print(s2)
s2.warp()
center, val = s2.find(sbbL, sbbR, verbose=False)
sbbL = cv2.drawMarker(sbbL, center, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)

xyz = np.array([+240, +240, 0])
s3 = Trainfeature("Gitterschraube1", xyz, 32)
print(s3)
s3.warp()
center, val = s3.find(sbbL, sbbR, verbose=False)
sbbL = cv2.drawMarker(sbbL, center, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)

xyz = np.array([+240, -240, 0])
s4 = Trainfeature("Gitterschraube1", xyz, 32)
print(s4)
s4.warp()
center, val = s4.find(sbbL, sbbR, verbose=False, extend=30)
sbbL = cv2.drawMarker(sbbL, center, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)


Trainfeature.referenceObjects(s1,s2,s3,s4)

sbbL = s1.drawBasis(sbbL, sideLR=0, show=False, length= 66, thickness=20)


cv2.namedWindow('Unit test marker L', cv2.WINDOW_NORMAL)
cv2.imshow('Unit test marker L', sbbL)
cv2.namedWindow('Unit test L', cv2.WINDOW_NORMAL)
cv2.imshow("Unit test L", imgL)
cv2.namedWindow('Unit test R', cv2.WINDOW_NORMAL)
cv2.imshow("Unit test R", imgR)
cv2.waitKey(0)
cv2.destroyAllWindows()


