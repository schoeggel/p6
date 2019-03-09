# Tests mit Trainfeature Objekten
# Patch laden, Eckpunkte rechnen etc.
# Version

import datetime
from archive.trainfeature import *
from archive import findGrid

# Gitter
# Bilder laden 10 bis 15
sbbL_gray = cv2.imread("sbb/12L.png", cv2.IMREAD_GRAYSCALE)
sbbR_gray = cv2.imread("sbb/12R.png", cv2.IMREAD_GRAYSCALE)

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

#
patchNameSchraube = "test1"
patchNameSchraube = "test2"
patchNameSchraube = "Gitterschraube1"
patchNameSchraube = "gitterschraube_or_L1"
patchName = "data/patches/3dcreatorSet1/"


# Teil 1
xyz = np.array([-312, 128, 4])
realsize = (36,36)
s1 = Trainfeature(patchName+"tcr3dschraubeKleinGanzLinkeSeite", xyz, realsize) #32 für kleiner patch , ca 45 für grosse
print(s1)
s1.warp()
centerL, val, centerR, _ = s1.find(sbbL, sbbR, verbose=True,extend=150)
print(f'center: {centerL}\nvalue: {val}\n')
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbL = s1.drawBasis(sbbL, sideLR=0, show=False, thickness=5)


# Teil 2
xyz = np.array([-596, +278, -67])
xyz = np.array([-596, -80, -67])
realsize = (30, 32)
vrot = [0, 0.4886921905584123, 0]
s2 = Trainfeature(patchName + "tcr3dschraubeMiniLinkeSeite", xyz, realsize, vrot)
print(s2)
s2.warp()
centerL, _ , centerR, _ = s2.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)


#Teil 3
xyz = np.array([-230, +550, 0])
realsize = (52, 181)
vrot = [0, 0, 0]
s3 = Trainfeature(patchName + "tcr3dscharnierLinkeSeite", xyz, realsize, vrot)
print(s3)
s3.warp()
centerL, _ , centerR, _ = s3.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)


# Teil 4
xyz = np.array([-250, -380, -100])
realsize = (70, 70)
vrot = [0, 0, 0]
s4 = Trainfeature(patchName + "tcr3dtestDeckelTiefLinkeSeite", xyz, realsize)
print(s4)
s4.warp()
centerL, _ , centerR, _ = s4.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)



# Teil 5
xyz = np.array([+250, -380, -100])
realsize = (70, 70)
vrot = [0, 0, 0]
s5 = Trainfeature(patchName + "tcr3dtestDeckelTiefLinkeSeite", xyz, realsize)
print(s5)
s5.warp()
centerL, _ , centerR, _ = s5.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)

# Teil 6
xyz = np.array([-312, 250, 4])
realsize = (36,36)
s6 = Trainfeature(patchName + "tcr3dschraubeKleinGanzLinkeSeite", xyz, realsize)
print(s6)
s6.warp()
centerL, _ , centerR, _ = s6.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)

# Teil 7
xyz = np.array([-120, 280, 4])
realsize = (36,36)
s7 = Trainfeature(patchName + "tcr3dschraubeKleinGanzLinkeSeite", xyz, realsize)
print(s7)
s7.warp()
centerL, _ , centerR, _ = s7.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)

# Teil 8
xyz = np.array([-420, -180, 0])
realsize = (66,62)
vrot = [0,0.157079632679489660,0]
s8 = Trainfeature(patchName + "tcr3ddeckelSeitlichLinks", xyz, realsize, vrot)
print(s8)
s8.warp()
centerL, _ , centerR, _ = s8.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)

# Teil 9
xyz = np.array([-420, +180, 0])
realsize = (66,62)
vrot = [0,0.157079632679489660,0]
s9 = Trainfeature(patchName + "tcr3ddeckelSeitlichLinks", xyz, realsize, vrot)
print(s9)
s9.warp()
centerL, _ , centerR, _ = s9.find(sbbL, sbbR, verbose=True)
sbbL = cv2.drawMarker(sbbL, centerL, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)
sbbR = cv2.drawMarker(sbbR, centerR, (0, 90, 255), cv2.MARKER_CROSS, 70, 2)





sbbL = s1.drawBasis(sbbL, sideLR=0, show=False, length= 66, thickness=20)

s1.drawMarker(sbbL, sbbR, show=False)
s2.drawMarker(sbbL, sbbR, show=False)
s3.drawMarker(sbbL, sbbR, show=False)
s4.drawMarker(sbbL, sbbR, show=False)
s5.drawMarker(sbbL, sbbR, show=False)
s6.drawMarker(sbbL, sbbR, show=False)
s7.drawMarker(sbbL, sbbR, show=False)
s8.drawMarker(sbbL, sbbR, show=False)
expL, expR = s9.drawMarker(sbbL, sbbR, show=True)

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


