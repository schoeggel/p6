# WTM Warped Tempate Matching
import cv2
import numpy as np


def sfm(img1, img2, k, verbose=False):
    """ Bestimmt den Versatz zwischen zwei Bildern einer Serie
         Wenn die Referenz nicht anhand der Gitterposition bestimmt werden kann.
         Die R und t Matrizen werden jeweils für beide Bildpaare L und R ermittelt.
         Verwendet wird der Mittelwert
         :param k: Kameramatrix"""

    # Erstelle eine Maske: Zug=255, Hintergrund = 0
    w, h = 4096, 3000
    mask1 = np.ones((h,w), dtype=np.uint8) * 255            # numpy array : shape(y,x)
    mask2 = np.ones((h,w), dtype=np.uint8) * 255            # numpy array : shape(y,x)
    pt1 = np.array([[3500, 0], [4095, 770], [4095, 0]])
    pt2 = np.array([[0, 0], [0, 1870], [1000, 0]])
    cv2.fillConvexPoly(mask1, pt1, 0)
    cv2.fillConvexPoly(mask2, pt2, 0)

    if verbose:
        cv2.namedWindow("m1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("m2", cv2.WINDOW_NORMAL)
        cv2.imshow("m1", mask1)
        cv2.imshow("m2", mask2)
        cv2.resizeWindow("m1", w//10, h//10)
        cv2.resizeWindow("m2", w//10, h//10)
        cv2.waitKey(0)
        cv2.destroyWindow("m1")
        cv2.destroyWindow("m2")


    orb:cv2.ORB = cv2.ORB_create()
    k1, d1 = orb.detectAndCompute(img1, mask1)
    k2, d1 = orb.detectAndCompute(img2, mask2)

    # Bruteforcematcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d1)
    matches = sorted(matches, key=lambda x: x.distance)

    #Ausrichten der Punkte
    sortedPts1=[]
    sortedPts2=[]
    for match  in matches:
        ##queryIdx is the "left" image
        sortedPts1.append(k1[match.queryIdx].pt)

        # trainIdx is the "right" image
        sortedPts2.append(k2[match.trainIdx].pt)


    # Essential Matrix finden. d1 und d2 dürfen keine Listen mehr sein.
    sortedPts1=np.array(sortedPts1)
    sortedPts2=np.array(sortedPts2)
    E, mask = cv2.findEssentialMat(sortedPts1,sortedPts2, k , method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, sortedPts1, sortedPts2, k, mask=mask)
    return R, t

if __name__ == '__main__':
    print("unit test.")
    i1 = cv2.imread("SBB/13L.png")
    i2 = cv2.imread("SBB/14L.png")
    i3 = cv2.imread("SBB/15L.png")
    i11 = cv2.imread("SBB/13R.png")
    i12 = cv2.imread("SBB/14R.png")
    i13 = cv2.imread("SBB/15R.png")
    kameramatrix1 = np.array([[2.20421168e+04, 0.00000000e+00, 1.98666895e+03],
                              [0.00000000e+00, 2.20443322e+04, 1.54111484e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    print(kameramatrix1)
    R, t = sfm(i1, i2, kameramatrix1)
    print(f'\n{45*"-"}\n\t\t\timage 1 -> 2 (Cam1)\nR:\n{R}\nt:\n{t}')

    R, t = sfm(i2, i3, kameramatrix1)
    print(f'\n{45*"-"}\n\t\t\timage 2 -> 3 (Cam1)\nR:\n{R}\nt:\n{t}')

    R, t = sfm(i11, i12, kameramatrix1)
    print(f'\n{45*"-"}\n\t\t\timage 11 -> 12 (Cam2)\nR:\n{R}\nt:\n{t}')

    R, t = sfm(i12, i13, kameramatrix1)
    print(f'\n{45*"-"}\n\t\t\timage 12 -> 13 (Cam2)\nR:\n{R}\nt:\n{t}')
