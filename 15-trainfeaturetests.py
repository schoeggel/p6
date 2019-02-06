# Tests mit Trainfeature Objekten
# Patch laden, Eckpunkte rechnen etc.

import cv2
import numpy as np
from trainfeature import Trainfeature

xyz = np.array([100, 101, 102])
s1 = Trainfeature("Gitterschraube1", xyz, 80)
print(s1)

