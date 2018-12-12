# Klasse für ein zu findendes / messendes Objekt oder Feature am Zug
import numpy as np
import cv2
from rigid_transform_3d import rigid_transform_3D


class Trainfeature:
    # Die Koordinatentransformation ist für alle Instanzen gleich
    R = np.diag([1,1,1])                            # Init Wert
    t = np.zeros(3)                                 # Init Wert

    def __init__(self):
        self.name = "new Feature"                   # Objektname
        self.patchfilename = None                   # zum laden des patchbilds
        self.patchsize = (-1, -1)                   # Grösse des geladenen patchs
        self.realpos = np.zeros(3)                  # Vektor 3d Position Patch Mitelpunkt (im Train system)
        self.realsize = -1                          # Kantenlänge 3d des Patches.
        self.patchimage = None                      # Das Bild
        self.tilt = None                            # später: TODO : objekt kann auch schräg sein
        self.corners = None                         # Die Vier Eckpunkte als 3d Koordinaten (im Train system)


    def reference(self, refpts):
        """
        Referenz festlegen für das Koordinatensystem "Zug"
        :param refpts: Die auf einer Ebene quadratisch angeordneten Schrauben (numpy.shape = (4,3))
        :return: None
        """

        # Refpts sind die 3d Koordinaten der vier Gitter Schrauben punkte (ol, or, ur, ul)
        # Damit wird sowohl der Ursprung (gleich deren Mittelwert) als auch die Orientierung
        # des Zug-Koordinatensystems gegenüber dem Kamera-Welt Koordinatensystems bekannt.
        # Koordinaten sind nicht homogen, sondern karthesisch.

        assert refpts.shape == (4, 3)

        # Der Mittelwert der vier Eckpunkte ist der Ursprung des Zugskoordinatensystems
        m = np.average(refpts)

        # Die Vektoren zu den vier Eckpunkten
        vma = refpts[0] - m
        vmb = refpts[1] - m
        vmc = refpts[2] - m
        vmd = refpts[3] - m

        # Die Ausrichtung anhand der Ebene bestimmen
        # Einheitsvektor ex
        vme = vmb + vmc - vmd - vma
        ex = vme / abs(vme)

        # Einheitsvektor ey
        vme = vma + vmb - vmc - vmd
        ey = vme / abs(vme)

        # Einheitsvektor ez (steht senkrecht auf die anderen zwei)
        ez = np.cross(ex, ey) / abs(np.cross(ex, ey))

        # Rotation und Translation berechnen
        # [[x1, y1, z1] , [x2 ... ]]
        systemzug = np.diag(ex,ey,ez)               # kanonische Einheitsvektoren
        systemcam = np.diag([1, 1 , 1])             # kanonische Einheitsvektoren
        R, t = rigid_transform_3D(systemcam, systemzug)
        print("R\n", R)
        print("t\n", t)


    def createfeature(self, name, realpos, realsize):
        self.name = name                            # Objektname
        self.patchfilename = name + ".png"          # zum laden des patchbilds
        self.patchsize = (-1, -1)                   # Grösse des geladenen patchs
        self.realpos = realpos                      # Vektor 3d Position Patch Mitelpunkt (im Train system)
        self.realsize = realsize                    # Kantenlänge 3d des Patches.
        self.tilt = None                            # später: TODO : objekt kann auch schräg sein
        self.corners = None                         # Die Vier Eckpunkte als 3d Koordinaten (im Train system)
        self.loadpatch()                            # default-Patch laden

        return self





    def loadpatch(self, filename=None):
        if filename is not None:                    # optional kann ein anderes als das standardbild geladen werden
            self.patchfilename = filename

        self.patchimage = cv2.imread("patches/" + self.patchfilename + ".png", cv2.IMREAD_GRAYSCALE)
        self.patchsize = self.patchimage.size()

