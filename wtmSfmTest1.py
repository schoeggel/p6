# WTM Warped Tempate Matching
import cv2
import numpy as np
import wtmCalib
import reproFilter

def sfm(imgL1, imgR1, imgL2, imgR2, calib, verbose=False):
    """ Bestimmt die räumliche Translation zwischen zwei Bildpaaren
        von allen vier Bildern werden Keypoints erstellt und einerseits
        paarweise gematcht, andernseits sequenzweise:
            match12 = Match zwischen imgL1 und imgR1
            match34 = Match zwischen imgL2 und imgR2
            match13 = Match zwischen imgL1 und imgL2
            match24 = Match zwischen imgR1 und imgR2

         :param calib: Kamera Kalibrierdaten"""

    # Erstelle eine Maske: Zug=255, Hintergrund = 0
    w, h = 4096, 3000
    mask1 = np.ones((h,w), dtype=np.uint8) * 255            # numpy array : shape(y,x)
    mask2 = np.ones((h,w), dtype=np.uint8) * 255            # numpy array : shape(y,x)
    pt1 = np.array([[3500, 0], [4095, 770], [4095, 0]])
    pt2 = np.array([[0, 0], [0, 1870], [1000, 0]])
    cv2.fillConvexPoly(mask1, pt1, 0)
    cv2.fillConvexPoly(mask2, pt2, 0)


    ################################### MATCHING ZWISCHEN L UND R Bilder     ###################

    # Detector / descriptor
    orb:cv2.ORB = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(imgL1, mask1)
    k2, d2 = orb.detectAndCompute(imgR1, mask2)

    # Bruteforcematcher
    bf: cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)

    # Ausrichten der Punkte (sortedPts1[i] entspricht sortedPts2[i])
    sortedPts1 = []
    sortedPts2 = []
    for match in matches:
        sortedPts1.append(k1[match.queryIdx].pt)        # queryIdx = Linkes Bild
        sortedPts2.append(k2[match.trainIdx].pt)        # trainIdx = Rechtes Bild

    # Infos anzeigen min/max/mittelwert
    if verbose:
        print(f'There are {len(sortedPts1)} sorted Points')
        print(f'#matches: {len(matches)}')
        dist = [m.distance for m in matches]
        print(f'distance: min: {min(dist)}')
        print(f'distance: mean: {sum(dist) / len(dist)}')
        print(f'distance: max: {max(dist)}')

    # repro error filter ist besser bei L+R kombi
    sel_matches, msg  = reproFilter.filterReprojectionError(matches, cal.f, np.int32(sortedPts1), np.int32(sortedPts2), 4 )
    print(msg)

    # Matches anzeigen in den beiden Bildern
    if verbose:
        n =  50 # max soviele matches zeichnen
        mimg = cv2.drawMatches(imgL1, k1, imgR1, k2, sel_matches[:n], None, flags=2)
        cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
        cv2.imshow("matches", mimg)
        cv2.resizeWindow("matches", w // 10, h // 10)
        cv2.waitKey()
        cv2.destroyWindow("matcher")

    ################################### MATCHING ZWISCHEN UNTERSCHIEDLICHEN ZEITPUNKTEN  ###################

    k3, d3 = orb.detectAndCompute(imgL2, mask2)
    k4, d4 = orb.detectAndCompute(imgR2, mask2)


    R = -1
    t = -1
    return R, t

if __name__ == '__main__':
    print("unit test.")
    img13L = cv2.imread("SBB/13L.png")
    img14L = cv2.imread("SBB/14L.png")
    img15L = cv2.imread("SBB/15L.png")
    img13R = cv2.imread("SBB/13R.png")
    img14R = cv2.imread("SBB/14R.png")
    img15R = cv2.imread("SBB/15R.png")
    cal = wtmCalib.CalibData()
    print(cal)

    R, t = sfm(img13L, img13R, img14L,img14R, cal, verbose=True)

    print(f'\n{45*"-"}\n\t\t\timage 1 -> 2 (Cam1)\nR:\n{R}\nt:\n{t}')

 # brisk = cv2.cv2.BRISK_create(thresh=70)
    # k1 = brisk.detect(imgL1)
    # k2 = brisk.detect(imgR1)
    # k3 = brisk.detect(imgL2)
    # k4 = brisk.detect(imgR2)

    # freak = cv2.xfeatures2d_FREAK.create()
    # fk1, d1 = freak.compute(imgL1, k1)
    # fk2, d2 = freak.compute(imgR1, k2)
    # fk3, d3 = freak.compute(imgL2, k3)
    # fk4, d4 = freak.compute(imgR2, k4)

    # matches = sorted(matches, key=lambda x: x.distance)

    # threshold: half the mean
    #thres_dist = (sum(dist) / len(dist)) * 0.5
    # keep only the reasonable matches TODO: besser erst am Ende die T-Vektoren filtern?
    #sel_matches = [m for m in matches if m.distance < thres_dist]




