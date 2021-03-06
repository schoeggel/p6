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
from wtmAux import clahe, imgMergerV, imgMergerH, putBetterText
import wtmComposition
import wtmEnum
from rigid_transform_3d import rmserror

class Position:
    # Eine gemessene Position eines Objekts. Jeweils in beiden unterschiedliechen Bezugssystemen.
    # Die Bildernamen zeigen, in welchem Bildpaar die Messung erfolgte.

    def __init__(self):
        self.mac = np.zeros(3)
        self.cam = np.zeros(3)
        self.imgNames = ["",""]
        self.sceneName = "unknownScene"
        self.snapshotL = None
        self.snapshotR = None      # ein schnappschuss aus der Detektion
        self.reproError = -1
        self.tmMode: wtmEnum.tm = 0
        self.cannyPxPerK = -1

    def __str__(self):
        return f"Position in scene <{self.sceneName}>:\tcamera: {self.cam}\t\tmachine: {self.mac}\n"


class Positions(list):
    # Liste mit den gemessenen Positionen vom Typ 'Position'
    def __str__(self) -> str:
        s = ""
        with np.printoptions(precision=4, suppress=True):
            for e in self:
                s += e.__str__()
        return s



class MachineObject:
    """ Ein Objekt ist ein Teil/Obejekt/Feature auf dem Zugdach

         Von einem Objekt bestehen zwei Patches, von jeder Kamera eine
         Aufnahme, verzerrt in eine flache Draufsicht."""

    maxReproError = 4

    def __init__(self, filename, center3d, realsize, rotation3d=None, name=None):
        self._patchCenter3d = center3d.astype(float)  # Vektor 3d zum Patch Mitelpunkt im sys_zug
        self._realsizex = realsize[0]  # Kantenlänge 3d des Patches.
        self._realsizey = realsize[1]  # Kantenlänge 3d des Patches.
        self.patchfilenameL = None
        self.patchfilenameR = None
        self.patchimageOriginalL = None  # Das Bild im Originalzustand
        self.patchimageOriginalR = None  # Das Bild im Originalzustand
        self.patchimageL = None  # Das Bild (Kontrastverbessert), wird erst später befüllt
        self.patchimageR = None  # Das Bild (Kontrastverbessert), wird erst später befüllt
        self.corners3d = np.zeros((5, 3))  # Vier Eckpunkte plus Mittelpunkt des Objekts als 3d Koord.
        self._positions = Positions()  # Die gültigen, gemessene Position im System Zug
        self._rejectedPositions = Positions()  # Die UNgültigen, Positionen (bspw. repojektionsfehler zu gross)
        self._measuredposition3d_cam = None  # Die gemessene Position im System Kamera

        assert (len(filename) > 0) and (center3d.shape == (3,))
        assert (realsize[0] > 1) and realsize[1] > 1

        if rotation3d is None:
            self._rotation = None
        else:
            assert len(rotation3d) == 3
            self._rotation = rotation3d

        if name is None:
            self.name = filename  # Objektname
        else:
            self.name = name

        self.loadpatch(filename)  # default-Patch laden
        self.calculatePatchCorners3d()

    @property
    def positions(self):
        if len(self._positions):
            return self._positions.__str__()
        else:
            return "empty list"

    @property
    def rejectedPositions(self):
        if len(self._positions):
            return self._rejectedPositions.__str__()
        else:
            return "empty list"

    @property
    def avgPosMac(self):
        # Rechnet den Mittelwert der Positionen (System mac) und gibt sie zurück
        if self._positions.__len__() == 0: return np.nan
        return np.average(self._positionsAsNumpyArray()[0], 0)

    @property
    def avgPosCam(self):
        # Rechnet den Mittelwert der Positionen (System cam) und gibt sie zurück
        if self._positions.__len__() == 0: return np.nan
        return np.average(self._positionsAsNumpyArray()[1], 0)

    @property
    def rmserror(self):
        """Liefert den rms Error der Messungen. Da noch kein echter SOLL - Wert vorhanden: Mittelwert verwenden"""
        if self._positions.__len__() == 0: return np.nan
        A = self.avgPosMac
        A = np.tile(A, (len(self._positions),1))
        B = self._positionsAsNumpyArray()[0]
        e = rmserror(A, B)
        return e

    def _positionsAsNumpyArray(self):
        mac = np.zeros((len(self._positions), 3))
        cam = np.zeros((len(self._positions), 3))
        for idx, p in enumerate(self._positions):
            mac[idx] = p.mac
            cam[idx] = p.cam
        return mac, cam


    def addPosition(self, p:Position):
        # Erstellt eine neue Positionsmessung, fügt sie zur Messliste hinzu
        assert len(p.cam) == 3 and len(p.mac) == 3
        assert len(p.imgNames) == 2 and len(p.imgNames[0]) > 0 and len(p.imgNames[1]) > 0
        if p.reproError <= self.maxReproError:
            self._positions.append(p)
        else:
            self._rejectedPositions.append(p)

    def loadpatch(self, filename):
        # muss .png sein !
        self.patchfilenameL = filename + "_L.png"
        self.patchfilenameR = filename + "_R.png"

        print(f'read: {self.patchfilenameL} und {self.patchfilenameR}')
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
        wname = f'Machine object: original + CLAHE'
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.imshow(wname, quadro)
        cv2.resizeWindow(wname, quadro.shape[0], quadro.shape[1])
        cv2.waitKey(0)
        cv2.destroyWindow(wname)


    def showGoodSnapshots(self):
        if len(self._positions) == 0: return
        self._snapshots(self._positions, "Good", show=True, save=False)

    def showBadSnapshots(self):
        if len(self._rejectedPositions) == 0: return
        self._snapshots(self._rejectedPositions, "Bad", show=True, save=False)

    def exportGoodSnapshots(self):
        if len(self._positions) == 0: return
        self._snapshots(self._positions, "Good", show=False, save=True)

    def exportBadSnapshots(self):
        if len(self._rejectedPositions) == 0: return
        self._snapshots(self._rejectedPositions, "Bad", show=False, save=True)

    def _snapshots(self, positionlist, txt, show=False, save=False):
        snapshots = []
        tmm = 0
        for e in positionlist:
            tmm = e.tmMode
            image = imgMergerH([e.snapshotL, e.snapshotR])
            image = putBetterText(image, f'reproError={e.reproError}', (5,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255),1,1)
            image = putBetterText(image, f'tmMode={e.tmMode}', (5,60), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255),1,1)
            if tmm in [wtmEnum.tm.CANNYBLUR, wtmEnum.tm.NOISEBLUR, wtmEnum.tm.CANNYBLUR2]:
                image = putBetterText(image, f'px per K={e.cannyPxPerK}', (5, 90), cv2.FONT_HERSHEY_DUPLEX, 1,
                                      (255, 255, 255), 1, 1)

            snapshots.append(image)

        bigpic = imgMergerV(snapshots)
        if show:
            wname = "snapshots - " + txt
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
            cv2.imshow(wname, bigpic)
            cv2.waitKey(0)
            cv2.destroyWindow(wname)
        if save:
            print(f'snapshot export: {self.name} ...')
            cv2.imwrite(f'tmp/snapshots-{self.name}-{txt}-{tmm}.jpg', bigpic, [cv2.IMWRITE_JPEG_QUALITY, 45])




    def __str__(self):
        return f'{self.__class__.__name__ } <{self.name}>\nPatchfilenameL: {self.patchfilenameL}\nPatchfilenameR: ' \
               f'{self.patchfilenameR}\nvalid position entries: {len(self._positions)}' \
               f' | rejected position entries: {len(self._rejectedPositions)}' \
               f'\nGood Positions:\n{self.positions}' \
               f'\nRejected Positions:\n{self.rejectedPositions}' \
               f'\nmean position: {self.avgPosMac}' \
               f'\nrms error: {self.rmserror}'




if __name__ == '__main__':
    # Erstellt diverse Objekte und Testet die Funktionen
    # Benötigt ein vorhandenes Patch Paar.
    print(f'unit test.\nopencv version: {cv2.getVersionString()}')

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

    s2.addPosition([1,2,3],[4,5,6],["LinkesBild","RechtesBild"],"testScene", [[],[]],3 )
    print(s2.positions)

    s2.addPosition([11,12,13],[14,15,16],["13-L.png","13-R.png"],"testScene", [[],[]],3 )
    s2.addPosition([51,52,53],[54,55,56],["14-L.png","14-R.png"],"testScene", [[],[]],3 )
    s2.addPosition([81,82,83],[84,85,86],["15-L.png","15-R.png"],"testScene", [[],[]],3 )
    print(s2.positions)
    s2.show()
