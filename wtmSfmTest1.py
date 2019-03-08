# WTM Warped Tempate Matching
import cv2
import numpy as np
import wtmCalib
import reproFilter

def sfm(img1, img2, img3, img4, calib, verbose=False):
    """ Bestimmt die räumliche Translation zwischen zwei Bildpaaren
        von allen vier Bildern werden Keypoints erstellt und einerseits
        paarweise gematcht, andernseits sequenzweise:
            match12 = Match zwischen img1 und img2
            match34 = Match zwischen img3 und img4
            match13 = Match zwischen img1 und img3
            match24 = Match zwischen img2 und img4
         :param calib: Kamera Kalibrierdaten"""
    #
    #    Definiton der Bildnummerierungen
    #
    #     left    right
    #    +-------------+
    #    |  1   |   2  |   time t
    #    +------+------+
    #    |  3   |   4  |   time t+1
    #    +------+------+


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
    akaze:cv2.AKAZE = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB)  # KAZE Descriptor geht nicht, weil float64 type...
    akazeU:cv2.AKAZE = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)
    latch:cv2.xfeatures2d_LATCH = cv2.xfeatures2d_LATCH.create()

    print("detect and compute... (~ 30 sec)")
    k1, d1 = akaze.detectAndCompute(img1, mask1)
    k2, d2 = akaze.detectAndCompute(img2, mask2)
    k3, d3 = akaze.detectAndCompute(img3, mask1)
    k4, d4 = akaze.detectAndCompute(img4, mask2)


    # Bruteforcematcher, Matches nach distance sortieren
    print("bruteforce matcher...")
    bf: cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches12 = bf.match(d1, d2)
    matches34 = bf.match(d3, d4)
    matches12 = sorted(matches12, key=lambda x: x.distance)
    matches34 = sorted(matches34, key=lambda x: x.distance)


    # Ausrichten der Punkte (sortedPts1[i] entspricht sortedPts2[i])
    pt1sort12 = []
    pt2sort12 = []
    for match in matches12:
        pt1sort12.append(k1[match.queryIdx].pt)        # queryIdx = Linkes Bild
        pt2sort12.append(k2[match.trainIdx].pt)        # trainIdx = Rechtes Bild

    pt3sort34 = []
    pt4sort34 = []
    for match in matches34:
        pt3sort34.append(k3[match.queryIdx].pt)        # queryIdx = Linkes Bild
        pt4sort34.append(k4[match.trainIdx].pt)        # trainIdx = Rechtes Bild

    # Infos anzeigen min/max/mittelwert
    if verbose:
        print(f'There are {len(pt1sort12)} sorted Points')
        print(f'#matches: {len(matches12)}')
        dist = [m.distance for m in matches12]
        print(f'distance: min: {min(dist)}')
        print(f'distance: mean: {sum(dist) / len(dist)}')
        print(f'distance: max: {max(dist)}')

    # repro error filter liefert sehr gute Ergebnisse bei L+R kombi
    sel_matches12, msg12  = reproFilter.filterReprojectionError(matches12, cal.f, np.int32(pt1sort12), np.int32(pt2sort12), 4 )
    sel_matches34, msg34  = reproFilter.filterReprojectionError(matches34, cal.f, np.int32(pt3sort34), np.int32(pt4sort34), 4 )
    print(msg12)
    print(msg34)





    # Matches anzeigen in den beiden LR Paaren
    if verbose:
        n =  150 # max soviele matches zeichnen
        wname1, wname2 = "matches 1-2", "matches 3-4"
        mimg12 = cv2.drawMatches(img1, k1, img2, k2, sel_matches12[:n], None, flags=2)
        mimg34 = cv2.drawMatches(img3, k3, img4, k4, sel_matches34[:n], None, flags=2)
        cv2.namedWindow(wname1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(wname2, cv2.WINDOW_NORMAL)
        cv2.imshow(wname1, mimg12)
        cv2.imshow(wname2, mimg34)
        cv2.resizeWindow(wname1, w // 10, h // 10)
        cv2.resizeWindow(wname2, w // 10, h // 10)
        cv2.waitKey()
        cv2.destroyWindow(wname1)
        cv2.destroyWindow(wname2)


    ################################### MATCHING ZWISCHEN UNTERSCHIEDLICHEN ZEITPUNKTEN  ###################

    # neue Matches Bilden auf Basis der gefilterten L-R Matches
    # Nur die Keypoints aus Bild 1 übernehmen, die einen erfolgreichen Match mit Bild 2 hatten
    sel_k1 = []
    for match in sel_matches12:
        sel_k1.append(k1[match.queryIdx])  # ohne .pt: ganzes keypoint objekt verwenden, queryIdx = Linkes Bild

    # Es sollen zu diesem Keypoints Matches gefunden werden im Bild 3. Da Bild 1-->3 ohne Rotation ist, werden neue
    # Deskriptoren erstellt für: Ausgewählte k in Bild 1, alle k in Bild 3

    uk1, ud1 = akazeU.compute(img1, sel_k1)  # u für UPRIGHT
    uk3, ud3 = akazeU.compute(img3, k3)      # u für UPRIGHT

    # mit den ausgewählten Keypoints neue Matches bilden (L-L und R-R)
    matches13 = bf.match(ud1, ud3)
    #matches24 = bf.match(sel_d2, sel_d4)
    matches13 = sorted(matches13, key=lambda x: x.distance)
    #matches24 = sorted(matches24, key=lambda x: x.distance)

    # Ausrichten der Punkte (sortedPts1[i] entspricht sortedPts2[i])
    pt1sort13 = []
    pt3sort13 = []
    for match in matches13:
        pt1sort13.append(uk1[match.queryIdx].pt)  # queryIdx = Linkes Bild
        pt3sort13.append(uk3[match.trainIdx].pt)  # trainIdx = Rechtes Bild


    n = 150  # max soviele matches zeichnen
    wname3, wname4 = "matches 1-3", "matches 2-4"
    mimg13 = cv2.drawMatches(img1, uk1, img3, uk3, matches13[:n], None, flags=2)
    cv2.namedWindow(wname3, cv2.WINDOW_NORMAL)
    cv2.imshow(wname3, mimg13)
    cv2.resizeWindow(wname3, w // 10, h // 10)
    cv2.waitKey()
    cv2.destroyWindow(wname3)


    R = -1
    t = -1
    return R, t

if __name__ == '__main__':
    print("unit test.")
    img13L = cv2.imread("SBB/13L.png", cv2.IMREAD_GRAYSCALE)
    img14L = cv2.imread("SBB/14L.png", cv2.IMREAD_GRAYSCALE)
    img15L = cv2.imread("SBB/15L.png", cv2.IMREAD_GRAYSCALE)
    img13R = cv2.imread("SBB/13R.png", cv2.IMREAD_GRAYSCALE)
    img14R = cv2.imread("SBB/14R.png", cv2.IMREAD_GRAYSCALE)
    img15R = cv2.imread("SBB/15R.png", cv2.IMREAD_GRAYSCALE)
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


   # sel_k4 = []
    # sel_d3 = []
    # sel_d4 = []
    # for match in sel_matches34:
    #     sel_k3.append(k3[match.queryIdx])  # ohne .pt: ganzes keypoint objekt verwenden, queryIdx = Linkes Bild
    #     sel_k4.append(k4[match.trainIdx])  # ohne .pt: ganzes keypoint objekt verwenden, trainIdx = Rechtes Bild
    #     sel_d3.append(d3[match.queryIdx])  # ohne .pt: ganzes keypoint objekt verwenden, queryIdx = Linkes Bild
    #     sel_d4.append(d4[match.trainIdx])  # ohne .pt: ganzes keypoint objekt verwenden, trainIdx = Rechtes Bild

    # Die descriptoren sind noch als liste
    # sel_d1 = np.asarray(sel_d1, dtype=np.uint8)
    # sel_d2 = np.asarray(sel_d1, dtype=np.uint8)
    # sel_d3 = np.asarray(sel_d1, dtype=np.uint8)
    # sel_d4 = np.asarray(sel_d1, dtype=np.uint8)


