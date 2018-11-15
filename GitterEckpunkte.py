import numpy as np
import random
import cv2


def eckpunkte(x, seite=0):
    # In : die x-Bildkoordinate des Gitter Mittelpunkts
    #      Seite 0 für Links oder 1 für Rechts
    # Out: ein Vektor mit den vier Eckpunkten (Schrauben) oben links uhrzeigersinn
    # Formeln basieren auf Ausmessungen vom 14.11.2018

    if not (seite == 1 or seite == 0):
        Exception("Seite muss 1 oder 0 sein")

    if seite == 1:
        # TODO
        pass

    if seite == 0:
        # olx = oben links x-koordinate
        olx = 0.8948 * x - 649.48
        oly = 1.6193 * x - 2071.7
        orx = 1.0102 * x + 390.29
        ory = 1.5680 * x - 2271
        ulx = 0.9858 * x - 407.78
        uly = 1.7839 * x - 1624.2
        urx = 1.1138 * x + 661.68
        ury = 1.7236 * x - 1827.9
        pointsl = np.float32([[olx, oly], [orx, ory], [ulx, uly], [urx, ury]])
        return pointsl


if __name__ == '__main__':
    # Testet die Funktion "eckpunkte"

    img = cv2.imread("sbb/16L.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    centerx = random.randint(1200, 2800)
    center = (centerx, int(1.6703 * centerx - 1939))
    print(center)
    # center = (1445, 458)

    pt = eckpunkte(center[0])

    for i in range(0, 4):
        img = cv2.drawMarker(img, center, (255, 0, 255), cv2.MARKER_CROSS, 100, 5)
        img = cv2.drawMarker(img, (pt[i][0], pt[i][1]), (0, 255, 255), cv2.MARKER_CROSS, 100, 5)

    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
