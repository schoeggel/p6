import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('data/testDoG1.png')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = np.random.random((800,800))
dst *= 200
print(dst)
dst = cv2.warpPerspective(src=img,M=M,dsize=dst.shape, borderMode=cv2.BORDER_CONSTANT, borderValue=200.002)

mask = dst == 200.002
print(mask)
print(dst)

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()