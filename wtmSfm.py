# WTM Warped Tempate Matching
import cv2
import numpy as np
import wtmCalib
import reproFilter
import copy


def getLinkedMatch(kpidx, otherMatch):
    for other in otherMatch:
        if other.queryIdx == kpidx:
            return other.trainIdx
    print("No linking match found.")
    return -1

def sortKeypoints(kp1, kp2, matches) -> (list, list):
    """ Gibt die Punkte aus den Keypoints sortiert
        zurück, entsprechend der Zuordnung in 'matches'.
        (sortedPts1[i] entspricht sortedPts2[i]) """
    pt1sort, pt2sort = [], []
    for match in matches:
        pt1sort.append(kp1[match.queryIdx].pt)  # queryIdx = Linkes Bild
        pt2sort.append(kp2[match.trainIdx].pt)  # trainIdx = Rechtes Bild
    return pt1sort, pt2sort


def copyMatch(match:cv2.DMatch) -> cv2.DMatch:
    # deepcopy geht nicht -> can't pickle cv2.DMatch objects
    # es muss eine kopie sein, weil anschliessend wert überschrieben werden.
    newmatch = cv2.DMatch()
    newmatch.imgIdx = copy.copy(match.imgIdx)
    newmatch.distance = copy.copy(match.distance)
    newmatch.queryIdx = copy.copy(match.queryIdx)
    newmatch.trainIdx = copy.copy(match.trainIdx)
    return newmatch



def cleanup(matches, key1, key2) -> (list, list, list, list, list):
    """ Räumt keypoints auf, nachdem matches entfernt wurden.
        Gibt die korrigierten Matches zurück.
        Gibt eine Liste nur mit den in 'matches' vorkommenden Keypoints zurück.
        Gibt nebst den matches die geordneten Koordinaten der Keypoints zurück"""
    xy1, xy2 = [], []
    newkeys1, newkeys2 = [], []
    newmatches = [] # funktioniert nicht, error : copy.deepcopy(matches)  --> can't pickle cv2.DMatch objects   # kopie machen von matches
    for i, match in enumerate(matches):
        newmatches.append(copyMatch(match))
        newmatches[i].queryIdx = i
        newmatches[i].trainIdx = i
        newkeys1.append(key1[match.queryIdx])  # queryIdx = Linkes Bild
        newkeys2.append(key2[match.trainIdx])  # trainIdx = Rechtes Bild
        xy1 = newkeys1[-1].pt
        xy2 = newkeys2[-1].pt
    return newmatches, newkeys1, newkeys2, xy1, xy2


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

    ################################################################################################################
    ######################   M A T C H I N G   Z W I S C H E N      L      U N D      R       ######################
    ################################################################################################################

    # Detector / descriptor
    # TODO: gleich gut geeigneter detector wie AKAZA finden, der aber schneller ist in der opencv implementierung.
    # Einmal akaze für detect und compute (rotation-invariant): für L-R Match
    # Einmal latch für erneutes compute der selben punkte, aber diesmal upright: für L-L, resp. R-R Match
    # akaza unterstützt ebenfalls upright, ist aber 10x langsamer als latch.
    akaze:cv2.AKAZE = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB)  # MLDB wählen, weil der enthält int-Werte
    latch:cv2.xfeatures2d_LATCH = cv2.xfeatures2d_LATCH.create(rotationInvariance=False)

    print("detect and compute... (~ 30 sec)")
    k1, d1 = akaze.detectAndCompute(img1, mask1)
    k2, d2 = akaze.detectAndCompute(img2, mask2)
    k3, d3 = akaze.detectAndCompute(img3, mask1)
    k4, d4 = akaze.detectAndCompute(img4, mask2)
    print("detect and compute... DONE.")

    # Bruteforcematcher, Matches nach distance sortieren
    print("bruteforce matcher...")
    bf: cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches12 = bf.match(d1, d2)
    matches34 = bf.match(d3, d4)

    # Reihenfolge der Punkte gleichsetzen (sortedPts1[i] entspricht sortedPts2[i])
    pt1sort12, pt2sort12 = sortKeypoints(k1, k2, matches12)
    pt3sort34, pt4sort34 = sortKeypoints(k3, k4, matches34)

    # repro error Filter liefert sehr gute Ergebnisse bei L+R kombi
    sel_matches12, msg12  = reproFilter.filterReprojectionError(matches12, cal.f, np.int32(pt1sort12), np.int32(pt2sort12), 4 )
    sel_matches34, msg34  = reproFilter.filterReprojectionError(matches34, cal.f, np.int32(pt3sort34), np.int32(pt4sort34), 4 )
    print(msg12)
    print(msg34)

    # Verkleinerte Kopie erstellen, die nur die verbleibenden Matches und davon betroffenen Keypoints enthält
    cleanMatches12, sel_k1, sel_k2, pt1sort12, pt2sort12 = cleanup(sel_matches12, k1, k2)
    cleanMatches34, sel_k3, sel_k4, pt3sort34, pt4sort34 = cleanup(sel_matches34, k3, k4)

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






    ################################################################################################################
    ######## M A T C H I N G   Z W I S C H E N   U N T E R S C H I E D L I C H E N   Z E I T P U N K T E N  ########
    ################################################################################################################

    # neue Matches bilden auf Basis der gefilterten L-R Matches
    # Zu den ausgewählten Keypoints sollen  Matches gefunden werden im Bild 3 (matches13), resp. Bild 4 (matches24)
    # Da Bild 1<-->3 und Bild 2<-->4 ohne Rotation ist (Der Zug fährt geradeaus), werden neue Deskriptoren
    # erstellt für die ausgewählten Keypoints in Bild 1 und 2, sowie zu allen Keypoints in den Bilder 3 und 4.
    uk1, ud1 = latch.compute(img1, sel_k1)  # u für UPRIGHT
    uk2, ud2 = latch.compute(img2, sel_k2)
    uk3, ud3 = latch.compute(img3, sel_k3)
    uk4, ud4 = latch.compute(img4, sel_k4)

    # mit den ausgewählten Keypoints neue Matches bilden (L-L und R-R)
    matches13 = bf.match(ud1, ud3)
    matches24 = bf.match(ud2, ud4)

    if verbose:
        n = 100  # max soviele matches zeichnen
        wname3, wname4 = "matches 1-3", "matches 2-4"
        mimg13 = cv2.drawMatches(img1, uk1, img3, uk3, matches13[:n], None, flags=2)
        mimg24 = cv2.drawMatches(img2, uk2, img4, uk4, matches24[:n], None, flags=2)
        cv2.namedWindow(wname3, cv2.WINDOW_NORMAL)
        cv2.namedWindow(wname4, cv2.WINDOW_NORMAL)
        cv2.imshow(wname3, mimg13)
        cv2.imshow(wname4, mimg24)
        cv2.resizeWindow(wname3, w // 10, h // 10)
        cv2.resizeWindow(wname4, w // 10, h // 10)
        cv2.waitKey()
        cv2.destroyWindow(wname1)
        cv2.destroyWindow(wname2)
        cv2.destroyWindow(wname3)
        cv2.destroyWindow(wname4)

    # Matches nach distance sortieren und Reihenfolge der Punkte gleichsetzen (sortedPts1[i] entspricht sortedPts2[i])
    pt1sort13, pt3sort13 = sortKeypoints(uk1, uk3, matches13)
    pt2sort24, pt4sort24 = sortKeypoints(uk2, uk4, matches24)



    ##################################### Keypoints ordnen, k2,k3,k4 an k1 ausrichten ###########################

    # Hier liegen die reduzierten Listen mit keypoints vor: uk1 bis uk4.  sel_matches12 und sel_matches34 beinhalten
    # alle keypoints uk1 und uk2, resp uk3 und uk4. (die keypoints sind ja anhand dieser Matchess in die Liste kopiert
    # worden). Die matches13 und matches 24 referenzieren (in der Regel) nur eine Teilmenge der Keypoints, da nicht
    # für alle Keypoints auch Matches gefunden werden können.
    # Nun müssen die Punktquartette erstellt werden. Nur wenn zum Punkt aus uk1 über die Matches Verbindungen zu uk2,
    # uk3 und uk4 bestehen, darf das Quartett erstellt werden.

    quadro= []
    pt1, pt2, pt3, pt4 = [],[],[],[]

    for mastermatch in cleanMatches12:
        kpidx1 = mastermatch.queryIdx
        kpidx2 = mastermatch.trainIdx

        # finde dazugehörigen kp auf Bild 3
        kpidx3 = getLinkedMatch(mastermatch.queryIdx, matches13)

        # finde dazugehörigen kp auf Bild 4
        kpidx4 = getLinkedMatch(mastermatch.trainIdx, matches24)

        # Wenn gültiges Quartett: keypoints in Liste aufnehmen
        if kpidx1 >= 0 and kpidx2 >= 0 and kpidx3 >= 0 and kpidx4 >= 0:
            quadro.append([uk1[kpidx1], uk2[kpidx2], uk3[kpidx3], uk4[kpidx4]])
            pt1.append(uk1[kpidx1].pt)
            pt2.append(uk2[kpidx2].pt)
            pt3.append(uk3[kpidx3].pt)
            pt4.append(uk4[kpidx4].pt)

    # Triangulieren, Punkte in Form "2xN" : [[x1,x2, ...], [y1,y2, ...]]
    a =  np.float64(pt1).T
    b =  np.float64(pt2).T
    c =  np.float64(pt3).T
    d =  np.float64(pt4).T

    # koordinaten trangulieren und umformen homogen --> kathesisch
    pt3d12 = cv2.triangulatePoints(calib.pl[:3], calib.pr[:3], a[:2], b[:2])
    pt3d34 = cv2.triangulatePoints(calib.pl[:3], calib.pr[:3], c[:2], d[:2])
    pt3d12 /= pt3d12[3]
    pt3d34 /= pt3d34[3]

    print(f'triangulated pt 3d:\n{pt3d12}')


    tvec = pt3d12 - pt3d34






    R = -1
    t = -1
    return R, t

if __name__ == '__main__':
    print("unit test.")
    print(f'opencv version: {cv2.getVersionString()}')
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

    # print("detect ... ")
    # k1 = akaze.detect(img1, mask1)
    # k2 = akaze.detect(img2, mask2)
    # k3 = akaze.detect(img3, mask1)
    # k4 = akaze.detect(img4, mask2)

    # print("compute ... ")
    # k1, d1 = latch.compute(img1, k1)
    # k2, d2 = latch.compute(img2, k2)
    # k3, d3 = latch.compute(img3, k3)
    # k4, d4 = latch.compute(img4, k4)
    # print(" ---------------  DONE --------------")

