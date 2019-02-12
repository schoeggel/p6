import numpy as np


import cv2

img = cv2.imread('data/17-testCLAHE.png', 0)
  # create a CLAHE object (Arguments are optional).


for exp in range(1,8):
    grid = (2**exp)+1

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid,grid))

    cl1 = clahe.apply(img)
    grid3 = str(grid).zfill(3)

    print(f'Exp, Grid, Grid3: {exp}, {grid}, {grid3}')
    cv2.imwrite(f'tmp/clahe-Grid_{grid3}.png', cl1)