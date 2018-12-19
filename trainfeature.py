# Klasse für ein zu findendes / messendes Objekt oder Feature am Zug
import numpy as np
import cv2
from rigid_transform_3d import rigid_transform_3D


class Trainfeature:
    # Die Koordinatentransformation <cam1 --> zug> ist für alle Instanzen gleich
    # Die Werte werden einmalig pro Bildpaar gesetzt mit der Methode "reference"
    R = np.diag([1,1,1])                            # Init Wert
    t = np.zeros(3)                                 # Init Wert


    def __init__(self, name, realpos, realsize):
        self.name = name                            # Objektname
        self.patchfilename = "data/patches/" + name + ".png"          # zum laden des patchbilds
        self.patchsize = ()                   # Grösse des geladenen patchs
        self.patchimage = None                      # Das Bild
        self.realpos = realpos                      # Vektor 3d Position Patch Mitelpunkt (im Train system)
        self.realsize = realsize                    # Kantenlänge 3d des Patches.
        self.tilt = None                            # später: TODO : objekt kann auch schräg sein
        self.corners = None                         # Die Vier Eckpunkte als 3d Koordinaten (im Train system)
        self.loadpatch()                            # default-Patch laden

    @staticmethod
    def reference(refpts):
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
        # Mittelwert Achse 0
        m = np.average(refpts,0)

        # Die Vektoren zu den vier Eckpunkten abcd
        vma = refpts[0] - m
        vmb = refpts[1] - m
        vmc = refpts[2] - m
        vmd = refpts[3] - m

        # Die Ausrichtung anhand der Ebene bestimmen
        # Einheitsvektor ex
        vme = vmb + vmc - vmd - vma
        ex = vme / np.linalg.norm(vme)

        # Einheitsvektor ey
        vme = vma + vmb - vmc - vmd
        ey = vme / np.linalg.norm(vme)

        # Einheitsvektor ez (steht senkrecht auf die anderen zwei)
        ez = np.cross(ex, ey) / np.linalg.norm(np.cross(ex, ey))

        # Rotation und Translation berechnen und in Klassenvariablen schreiben
        # [[x1, y1, z1] , [x2 ... ]]
        systemcam = np.diag(np.float32([1, 1 , 1]))                             # kanonische Einheitsvektoren
        #systemcam = np.append([np.zeros(3)], systemcam, axis=0)     # erste Zeile = Ursprung
        systemzug = np.stack((ex+m,ey+m,ez+m))                   # Usprung und kanonische Einheitsvektoren
        print("sysCam\n", systemcam)
        print("sysTrain\n", systemzug)
        print("\n\n")

        Trainfeature.R, Trainfeature.t = rigid_transform_3D(systemcam, systemzug)

        print("R\n", Trainfeature.R)
        print("t\n", Trainfeature.t)



    def loadpatch(self, filename=None):
        if filename is not None:                    # optional kann ein anderes als das standardbild geladen werden
            self.patchfilename = filename
        print("Lade: ", self.patchfilename )
        self.patchimage = cv2.imread(self.patchfilename, cv2.IMREAD_GRAYSCALE)
        self.patchsize = self.patchimage.shape



if __name__ == '__main__':
# Tests

    ref = np.float32([[-3.1058179e+02, -1.5248854e+02, 7.8082729e+03],
                      [ 1.3992499e+02, -2.6128412e+02, 7.9217915e+03],
                      [ 2.9599524e+02,  5.3110981e+00, 7.5579175e+03],
                      [-1.5471242e+02,  1.1419590e+02, 7.4451899e+03]])

    Trainfeature.reference(ref)



    A2 = (Trainfeature.R @ ref.T) + np.tile(Trainfeature.t, (1, 4))
    A2 = A2.T
    print("Reconstruct abcd Test\n", A2)
    #obj1 = Trainfeature("test1", 232, 2311)
    #obj2 = Trainfeature("test1", 232, 2311)


