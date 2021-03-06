# Tests mit Trainfeature Objekten
# Patch laden, Eckpunkte rechnen etc.

import datetime
from archive.trainfeature import *
from archive import findGrid

# Gitter
# Bilder laden 10 bis 15
sbbL_gray = cv2.imread("sbb/15L.png", cv2.IMREAD_GRAYSCALE)
sbbR_gray = cv2.imread("sbb/15R.png", cv2.IMREAD_GRAYSCALE)

#sbbL_gray = cv2.imread("sbb/3-OK1L.png", cv2.IMREAD_GRAYSCALE)
#sbbR_gray = cv2.imread("sbb/3-OK1R.png", cv2.IMREAD_GRAYSCALE)

# Bilder kopieren, umwandeln #
sbbL_rgb = cv2.cvtColor(sbbL_gray, cv2.COLOR_GRAY2BGR)
sbbR_rgb = cv2.cvtColor(sbbR_gray, cv2.COLOR_GRAY2BGR)

# Gitter suchen
gitterL, gitterR = findGrid.find(sbbL_gray, sbbR_gray, verbose=False)

# auf Original Foto einzeichnen
sbbL = cv2.circle(sbbL_rgb, (gitterL[0][0][0], gitterL[0][0][1]), 25, (255, 0, 255), -1)
sbbR = cv2.circle(sbbR_rgb, (gitterR[0][0][0], gitterR[0][0][1]), 25, (255, 0, 255), -1)

# Feature Tests
Trainfeature.loadmatrixp()
Trainfeature.approxreference(gitterL, gitterR)


patchNameSchraube = "test1"
patchNameSchraube = "test2"
patchNameSchraube = "Gitterschraube1"
patchNameSchraube = "gitterschraube_or_L1"
patchName = "data/patches/GitterschraubeSet1/"
realsize = 38


# Schraube oben links
xyz = np.array([-240, 240, 0])
s1 = Trainfeature(patchName+"ol", xyz, (realsize,realsize)) #32 für kleiner patch , ca 45 für grosse
print(s1)
s1.warp()
centerL, val, centerR, _ = s1.find(sbbL, sbbR, verbose=True)
print(f'center: {centerL}\nvalue: {val}\n')
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbL = s1.drawBasis(sbbL, sideLR=0, show=False, thickness=5)

# Schraube oben rechts
xyz = np.array([+240, +240, 0])
s3 = Trainfeature(patchName+"or", xyz, (realsize,realsize))
print(s3)
s3.warp()
centerL, _ , centerR, _ = s3.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)


# Schraube unten links
xyz = np.array([-240, -240, 0])
s2 = Trainfeature(patchName+"ul", xyz, (realsize,realsize))
print(s2)
s2.warp()
centerL, _, centerR, _ = s2.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)


# Schraube unten rechts
xyz = np.array([+240, -240, 0])
s4 = Trainfeature(patchName+"ur", xyz, (realsize,realsize))
print(s4)
s4.warp()
centerL, _, centerR, _  = s4.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)




Trainfeature.referenceViaObjects(s1,s2,s3,s4)

sbbL = s1.drawBasis(sbbL, sideLR=0, show=False, length= 66, thickness=20)

s1.drawMarker(sbbL, sbbR, show=False)
s2.drawMarker(sbbL, sbbR, show=False)
s3.drawMarker(sbbL, sbbR, show=False)
expL, expR = s4.drawMarker(sbbL, sbbR, show=True)

# Bilder anzeigen bis Taste gedrückt.
now = None
key = cv2.waitKey(0) & 0xFF
print(f'Key: {key}')

# Falls Taste "s" --> Bild speichern
if key == ord("s") or key == ord("S"):
    now = datetime.datetime.now()
    filenameTrunk = f'tmp/reprojection-{now.year}-{now.month}-{now.day}--{now.hour}{now.minute}{now.second}'
    print(f'Export: {filenameTrunk}-X.jpg ...')
    cv2.imwrite(f'{filenameTrunk}-L.jpg', expL, [cv2.IMWRITE_JPEG_QUALITY, 50])
    cv2.imwrite(f'{filenameTrunk}-R.jpg', expR, [cv2.IMWRITE_JPEG_QUALITY, 50])

cv2.destroyAllWindows()


