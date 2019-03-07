""" Objekt, das in den Modulen wtmComposition und wtmScene verarbeitet werden kann.

 Definition: Bezugssystem Kamera (camera) <---> Bezugssystem Zug (machine) = cam <---> mac

 Klasse MachineObject:
 Ein Objekt ist ein Teil/Obejekt/Feature auf dem Zugdach
 Von einem Objekt bestehen zwei Patches, von jeder Kamera eine Aufnahme, verzerrt in eine flache Draufsicht.

 Klasse Positionen:
 Darin sind alle gemessenen Positionen aufgeführt. Erweiterung einer Liste, vom Typ "Position"

 Klasse Position:
 eine einzelne Messung. Repräsntiert die 3d Koordinaten in beiden Bezugssystemen cam und mac (Kamera und Zug)
 Die Bildnamen zeigen, in welchem Bildpaar die Messung vorgenommen wurde. """
   

import numpy as np
import cv2, math
from wtmAux import clahe, imgMergerV, imgMergerH
import wtmComposition

class Position:
    # Eine gemessene Position eines Objekts. Jeweils in beiden unterschiedliechen Bezugssystemen.
    # Die Bildernamen zeigen, in welchem Bildpaar die Messung erfolgte.

    _mac = np.zeros(3)
    _cam = np.zeros(3)
    _imgNames = ["",""]

    def __str__(self):
        return f"Position in images '{self._imgNames[0]}' and '{self._imgNames[1]}'" \
            f":\tSystem camera (cam): {self._cam}\t\tSystem machine (mac): {self._mac}\n"

    @property
    def mac(self):
        return self._mac

    @mac.setter
    def mac(self, mac):
        assert len(mac) == 3
        self._mac = mac

    @property
    def cam(self):
        return self._cam

    @cam.setter
    def cam(self, cam):
        assert len(cam) == 3
        self._cam = cam

    @property
    def imgNames(self):
        return self._imgNames

    @imgNames.setter
    def imgNames(self, imgNames: list):
        assert len(imgNames) == 2 and len(imgNames[0]) > 0 and len(imgNames[1]) > 0
        self._imgNames = imgNames


class Positions(list):
    # Liste mit den gemessenen Positionen vom Typ 'Position'
    def __str__(self) -> str:
        s = "All measured Positions:\n"
        for e in self:
            s += e.__str__()
        return s
    # TODO: messwerte mitteln und gemittelte Koordinaten zurückgeben







class MachineObject:
    """ Ein Objekt ist ein Teil/Obejekt/Feature auf dem Zugdach

         Von einem Objekt bestehen zwei Patches, von jeder Kamera eine
         Aufnahme, verzerrt in eine flache Draufsicht."""

    patchfilenameL = None
    patchfilenameR = None
    patchimageOriginalL = None  # Das Bild im Originalzustand
    patchimageOriginalR = None  # Das Bild im Originalzustand
    patchimageL = None  # Das Bild (Kontrastverbessert), wird erst später befüllt
    patchimageR = None  # Das Bild (Kontrastverbessert), wird erst später befüllt
    _patchCenter3d = [0, 0, 0]  # Vektor 3d zum Patch Mitelpunkt im sys_zug
    _realsizex = 0  # Kantenlänge 3d des Patches.
    _realsizey = 0  # Kantenlänge 3d des Patches.
    corners3d = np.zeros((5, 3))  # Vier Eckpunkte plus Mittelpunkt des Objekts als 3d Koord.
    _positions = Positions()  # Die gemessene Position im System Zug
    _measuredposition3d_cam = None  # Die gemessene Position im System Kamera

    def __init__(self, filename, center3d, realsize, rotation3d=None, name=None):
        self._patchCenter3d = center3d.astype(float)  # Vektor 3d zum Patch Mitelpunkt im sys_zug
        self._realsizex = realsize[0]  # Kantenlänge 3d des Patches.
        self._realsizey = realsize[1]  # Kantenlänge 3d des Patches.

        assert (len(filename) > 0) and (center3d.shape == (3,))
        assert (realsize[0] > 1) and realsize[1] > 1

        if rotation3d is None:
            self._rotation = None
        else:
            assert len(rotation3d) == 3
            self._rotation = rotation3d

        if name is None:
            self._name = filename  # Objektname
        else:
            self._name = name

        self.loadpatch(filename)  # default-Patch laden
        self.calculatePatchCorners3d()

    @property
    def positions(self):
        return self._positions.__str__()

    def addPosition(self, mac, cam, imgNames):
        p = Position()
        p.mac = mac
        p.cam = cam
        p.imgNames = imgNames
        self._positions.append(p)

    def loadpatch(self, filename):
        # muss .png sein !
        self.patchfilenameL = filename + "_L.png"
        self.patchfilenameR = filename + "_R.png"

        print(f'Lade: {self.patchfilenameL} und {self.patchfilenameR}')
        self.patchimageOriginalL = cv2.imread(self.patchfilenameL, cv2.IMREAD_GRAYSCALE)
        self.patchimageOriginalR = cv2.imread(self.patchfilenameR, cv2.IMREAD_GRAYSCALE)

        if self.patchimageOriginalR is None:
            self.patchimageOriginalR = self.patchimageOriginalL

        # Kontrasoptimierte Kopie erstellen
        self.patchimageL = clahe(self.patchimageOriginalL, wtmComposition.Composition.PIXEL_PER_CLAHE_BLOCK)
        self.patchimageR = clahe(self.patchimageOriginalR, wtmComposition.Composition.PIXEL_PER_CLAHE_BLOCK)

        assert (self.patchimageOriginalL.size > 0)
        assert (self.patchimageOriginalL.shape == self.patchimageOriginalR.shape)

    def calculatePatchCorners3d(self):
        # Ausgehend von der Grösse des quadratischen Patchs und dessen Zentrumskoordinaten
        # werden die Koordinaten der vier Eckpunkte berechnet. Bezugssystem: mac
        # Ohne Rotation liegt der Patch auf xy Ebene mit dem Zentrum des Quadrats bei patchCenter3d

        # Patchmitte bis Patch Rand (in x oder y Richtung)
        dx = self._realsizex / 2
        dy = self._realsizey / 2

        # Alle Ecken erhalten vorerst den Mittelpunkt als Koordinaten
        self.corners3d = np.tile(self._patchCenter3d, (5, 1))

        # Patchmitte bis Patch Ecken, die Differenz vom Mittelpunkt zur Ecke
        d = np.array([[-dx, +dy, 0],  # oben links
                      [+dx, +dy, 0],  # oben rechts
                      [-dx, -dy, 0],  # unten links
                      [+dx, -dy, 0],  # unten rechts
                      [0, 0, 0]])  # Mitte

        # Ecken erstellen
        self.corners3d = self.corners3d + d

        # Rotieren
        if self._rotation is not None:
            self.rotatePoints()

    def rotatePoints(self):
        pt = self.corners3d

        # Schwerpunkt auf Ursprung setzen
        t = np.average(pt, 0)
        pt = pt - t

        # Rotationsmatrizen mit den Winkeln in [rad] erstellen
        a, b, c = self._rotation[0], self._rotation[1], self._rotation[2]
        Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])

        # Multiplizere Matrizen (@ statt np.matmul)
        pt = pt @ Rx @ Ry @ Rz

        # Translation wieder rückgängig machen
        self.corners3d = pt + t

    def show(self):
        upper = imgMergerH([self.patchimageOriginalL, self.patchimageOriginalR])
        lower = imgMergerH([self.patchimageL, self.patchimageR])
        quadro = imgMergerV([upper, lower])
        wname = f'Patch object: original + CLAHE'
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.imshow(wname, quadro)
        cv2.resizeWindow(wname, quadro.shape[0], quadro.shape[1])
        cv2.waitKey(0)
        cv2.destroyWindow(wname)

    def __str__(self):
        s = f'wtmObject:\n Name:{self._name}\n'
        s += f' PatchfilenameL: {self.patchfilenameL}\n PatchfilenameR: {self.patchfilenameR}\n'
        s += f' Real position center:\n {self._patchCenter3d}\n'
        s += f' Real position corners:\n{self.corners3d}\n'
        return s







if __name__ == '__main__':
    # Erstellt diverse Objekte und Testet die Funktionen
    # Benötigt ein vorhandenes Patch Paar.

    patchName = "data/patches/3dcreatorSet1/"

    # Erstellt ein einfaches Objekt
    xyz = np.array([-312, 128, 4])
    size = (36, 37)
    s1 = MachineObject(patchName + "tcr3dschraubeKleinGanzLinkeSeite", xyz, size)
    print(s1)

    # Erstellt ein Objekt mit Rotation und Custom Name
    xyz = np.array([-10, 20, -30])
    rot = [1, 0.5, 0.25]
    size = (25, 40)
    s2 = MachineObject(patchName + "tcr3dschraubeKleinGanzLinkeSeite", xyz, size, rotation3d=rot, name="customName")
    print(s2)
    print(s2.positions)

    s2.addPosition([1,2,3],[4,5,6],["LinkesBild","RechtesBild"])
    print(s2.positions)

    s2.addPosition([11,12,13],[14,15,16],["13-L.png","13-R.png"])
    s2.addPosition([51,52,53],[54,55,56],["14-L.png","14-R.png"])
    s2.addPosition([81,82,83],[84,85,86],["15-L.png","15-R.png"])
    print(s2.positions)
    s2.show()






