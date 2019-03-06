# WTM : Warped Template Matching
import numpy as np
import cv2
from rigid_transform_3d import rigid_transform_3D, rmserror
import calibMatrix
from enum import Enum
from cvaux import imgMergerV, imgMergerH
import math


class tm(Enum):
    MASK3CH = 1         # Template in einem bildkanal, masken in anderen kanälen. Versuch TM mit transparen
    NOISE = 2           # Template auf Rauschen.
    AVERAGE = 3         # TODO Template auf Hintergrund, der dem mittelwert des Templates entspricht
    CANNY = 4           # Template und Suchbereich durch canny edge detector laufen lassen
    TRANSPARENT = 5     # Verwenden der von opencv unterstützeten Transparenz mit Maske
    CANNYBLUR = 7       # Wie CANNY aber unscharf
    CANNYBLUR2 = 9      # Wie CANNY aber unscharf und INVERS
    NOISEBLUR = 8       # Wie NOISE aber unscharf
    ELSD = 6            # TODO: statt canny die ellipsen und linien erkennen.


class Composition:
    _trainIsReversed = False             #TODO Der Zug könnte auch verkehrt herum einfahren?
    _imagePairs = []                     # eine Liste mit Bildpaaren in der Form [[01L,01R], [02L, 02R] , ... ]
    _calib = calibMatrix.CalibData()
    _refObj = [None] * 4
    _scenes = []
    _p1 = None  # P Matrix für Triangulation
    _p2 = None  # P Matrix für Triangulation
    _tmmode = tm.CANNYBLUR  # Standard TM Mode

    _PRE_TM_K_SIZE = 5
    _PRE_TM_PX_PER_K = 15  # warpedTemplate hat Abmessung 50x50 px --> K = (5,5)
    _SCOREFILTER_K_SIZE = 5  # Kernelgrösse für die Glättung des TM Resultats (Score)
    _PIXEL_PER_CLAHE_BLOCK = 50  # Anzahl Blocks ist abhängig von der Bildgrösse


    def __init__(self):


    def findObjects(self, objects_list:list):
        pass

    # auf allen scenes das objekt suchen, vermessen
    def findObject(self, oneObject):
        pass


