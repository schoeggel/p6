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

# Gitter suchen
gitterL, gitterR = findGrid.find(sbbL_gray, sbbR_gray, verbose=False)

# auf Original Foto einzeichnen
sbbL = cv2.circle(sbbL_rgb, (gitterL[0][0][0], gitterL[0][0][1]), 25, (255, 0, 255), -1)
sbbR = cv2.circle(sbbR_rgb, (gitterR[0][0][0], gitterR[0][0][1]), 25, (255, 0, 255), -1)

cv2.namedWindow('Unit test L', cv2.WINDOW_NORMAL)
cv2.imshow("Unit test L", sbbL)
cv2.namedWindow('Unit test R', cv2.WINDOW_NORMAL)
cv2.imshow("Unit test R", sbbR)




xyz = np.array([100, 101, 102])
s1 = Trainfeature("Gitterschraube1", xyz, 80)
Trainfeature.loadmatrixp()

s1.approxreference(gitterL, gitterR)
print(s1)

cv2.waitKey(0)
cv2.destroyAllWindows()
