import wtmAux
import cv2

img = cv2.imread("SBB/13L.png")
img = wtmAux.putBetterText(img, f'Rt-Status: BLAAABLLAAA', (30, 150), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 5, 0)
img = wtmAux.putBetterText(img, f'Rt-Status: BLAAABLLAAA', (30, 350), cv2.FONT_HERSHEY_DUPLEX, 5, (0,0,0), 5, 0)
img = wtmAux.putBetterText(img, f'Rt-Status: BLAAABLLAAA', (30, 550), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 2, 2)
img = wtmAux.putBetterText(img, f'Rt-Status: BLAAABLLAAA', (30, 750), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 4, 3)
cv2.namedWindow('Basis', cv2.WINDOW_NORMAL)
cv2.imshow("Basis", img)
cv2.waitKey(0)