# Klasse für ein zu findendes / messendes Objekt oder Feature am Zug
import numpy as np
import cv2
from rigid_transform_3d import rigid_transform_3D, rmserror
import calibMatrix


class Trainfeature:
    # Die Koordinatentransformation <cam1 <--> zug> ist für alle Instanzen gleich
    # Die Werte werden einmalig pro Bildpaar gesetzt mit der Methode "reference"
    # oder ungefähr gesetzt mit der Methode "approxreference"

    p1 = None                                       # P Matrix für Triangulation
    p2 = None                                       # P Matrix für Triangulation
    __R_exact = np.diag([0, 0, 0])                  # Init Wert
    __t_exact = np.zeros(3)                         # Init Wert
    __R_approx = np.diag([0, 0, 0])                 # Init Wert
    __t_approx = np.zeros(3)                        # Init Wert
    __rtstatus = -1                                 # -1: keine, 0: approx, 1:exakte vorhanden


    def __init__(self, name, realpos, realsize):
        assert (len(name) > 0) and (realpos.shape == (3,)) and (realsize > 1)

        self.name = name  # Objektname
        self.patchfilename = "data/patches/" + name + ".png"          # zum laden des patchbilds
        self.patchimage = None                      # Das Bild
        self.patchsize = (0,0)                      # Bildgrösse des geladenen Patchs
        self.realpos = realpos.astype(float)        # Vektor 3d Position Patch Mitelpunkt im sys_zug
        self.realsize = realsize                    # Kantenlänge 3d des Patches.
        self.rotation = None                        # später: TODO : objekt kann auch schräg sein
        self.edges3d = np.empty((4,3))              # Die Vier Eckpunkte des Objekts als 3d Koordinaten im sys_zug
        self.edges2dleft = np.empty((1,4,2))        # Eckpunkte auf dem Bild.
        self.edges2dright = np.empty((1,4,2))       # Eckpunkte auf dem Bild.
        self.warpedpatch = None                     # das gewarpte template
        self.measuredposition = None                # Die gemessene Position
        self.loadpatch()                            # default-Patch laden



    def warp(self):
        # der Patch wird perspektivisch verzerrt, damit er so aussieht wie auf dem Bild erwartet

        # Eckpunkte (x,y) für beide Bilder L,R berechnen
        self.reprojectedges()

        # Die Leinwand für das transformierte Bild wird so gross
        minxl, minyl = self.edges2dleft.min(0)[0]
        maxxl, maxyl = self.edges2dleft.max(0)[0]

        minxr, minyr = self.edges2dright.min(0)[0]
        maxxr, maxyr = self.edges2dright.max(0)[0]

        canvasL = (int(maxxl-minxl), int(maxyl-minyl))
        canvasR = (int(maxxr-minxr), int(maxyr-minyr))

        # patch eckpunkte auf neue Leinwand umrechen (x und y minima pro Seite von pixelkoordinate subtrahieren)
        ofsl = np.float32([minxl, minyl])
        ofsr = np.float32([minxr, minyr])
        edgesL = self.edges2dleft - np.tile(ofsl, (4, 1, 1))
        edgesR = self.edges2dright - np.tile(ofsr, (4, 1, 1))
        edgesL = edgesL[:,0].astype(np.float32)
        edgesR = edgesR[:,0].astype(np.float32)


        # Eckpunkte des quadratischen Patchs:
        d = self.patchsize[0]
        quadrat = np.float32([[0, 0], [d, 0], [0, d], [d, d]])


        # Transformation am Bild durchführen. Datentyp muss float32 sein, sonst b(l)ockt open cv
        ML = cv2.getPerspectiveTransform(quadrat, edgesL)
        ML_inv = np.linalg.inv(ML)
        imgL = cv2.warpPerspective(self.patchimage, ML, canvasL, borderMode=cv2.BORDER_TRANSPARENT)

        MR = cv2.getPerspectiveTransform(quadrat, edgesR)
        MR_inv = np.linalg.inv(MR)
        imgR = cv2.warpPerspective(self.patchimage, MR, canvasR, borderMode=cv2.BORDER_TRANSPARENT)

        return imgL, imgR

    def rt(self):
        """
        Liefert die beste verfügbare Umrechnung der Bezugsysteme zurück.
        Exakte Umrechnung vor Approximation vor keine Umrechnung.

        :return: R, t (sys_cam --> sys_zug)
        """
        if self.__rtstatus == 2:
            return self.__R_exact, self.__t_exact
        else:
            return self.__R_approx, self.__t_approx




    def reprojectedges(self):
        # rechnet die Patch Ecken in x,y Pixelkoordinaten um
        # und speicher diese in der Instanz

        # Koordinaten der Patch Ecken rechnen (sys_zug)
        self.calculatedges3d()

        # ecken umrechnen sys_zug --> sys_cam (direction = 0)
        edges3d_cam = self.transformsys(self.edges3d, 0)

        # reprojection der Punkte in Bildpixelkoordinaten
        cal = calibMatrix.CalibData()
        left, jcb = cv2.projectPoints(edges3d_cam, cal.rl, cal.tl, cal.kl, cal.drl)
        right, jcb = cv2.projectPoints(edges3d_cam, cal.rr, cal.tr, cal.kr, cal.drr)
        print(f'projection LEFT:\n{left}\n\nprojection RIGHT:\n{right}')

        # opencv liefert die punkte im shape (n,1,3) zurück. L und R zusammenführen --> (n,2,3)
        self.edges2dleft = left
        self.edges2dright = right
        return left, right




    def transformsys(self, pts, direction):
        # rechnet punkte von einem Bezugssystem ins andere um
        # dir == 0: sys_zug --> sys_cam
        # dir == 1: sys_cam --> sys_zug
        # pts müssen in shape (n,3) sein, bspw: [[x,y,z]] oder [[x,y,z],[x2,y2,z2]]

        R, t = self.rt()    # beste verfügbare R|t Matrix
        n = pts.shape[0]    # wieviele punkte ?

        if direction == 0:
            A = pts
            B2 = (R @ A.T) + np.tile(t, (1, n))
            B2 = B2.T
            return B2
        elif direction == 1:
            B = pts
            A2 = (B - np.tile(t, (1, n)).T) @ R
            return A2

        else:
            assert False


    @classmethod
    # lädt die P-Matrizen für die Triangulationen
    def loadmatrixp(cls, customp1=None, customp2 = None):
        if customp1 is not None and customp2 is not None:
            cls.p1 = customp1
            cls.p2 = customp2
        else:
            cal = calibMatrix.CalibData()
            cls.p1 = cal.pl
            cls.p2 = cal.pr


    def calculatedges3d(self):
        # Ausgehend von der Grösse des quadratischen Patchs und dessen Zentrumskoordinaten
        # werden die Koordinaten der vier Eckpunkte berechnet. Bezugssystem: sys_zug
        # Ohne Rotation liegt der Patch auf xy Ebene mit dem Zentrum des Quadrats bei realpos
        if self.rotation is not None:
            print("Warnung, Rotation des Templates ist nicht nicht implementiert.") # TODO

        # Patchmitte bis Patch Rand
        d = self.realsize / 2

        # Alle Ecken erhalten vorerst den Mittelpunkt als Koordinaten
        self.edges3d = np.tile(self.realpos, (4, 1))

        # Patchmitte bis Patch Ecken, die Differenz vom Mittelpunkt zur Ecke
        d = np.array([[-d, +d, 0],
                      [+d, +d, 0],
                      [-d, -d, 0],
                      [+d, -d, 0]])
        self.edges3d += d


    @staticmethod
    def bisect(v1, v2):
        """
        Erstellt eine Winkelhalbierende zwischen v1, Ursprung, v2

        :param v1: erster Vektor ab Ursprung
        :param v2: zweiter Vektor ab Ursprung
        :return: Winkelhalbierende (Vektor ab Ursprung)
        """
        s = np.linalg.norm(v1) / np.linalg.norm(v2)    # beide Vektoren gleich lang machen
        return (v1 + s * v2) / 2



    @classmethod
    def approxreference(cls, gitterposL, gitterposR):
        # ungefähre Koordinatenbasis auf die Mitte des Gitter stellen. Pose ist Standard, stimmt nur ungefähr.

        # die Kanonischen Einheitsvektoren des sys_zug, aber mit dem Ursprung noch bei [0,0,0] von sys_cam
        systemzug = np.array([[0, 0, 0],
                                   [0.94423342, -0.2282705, 0.23731],
                                   [-0.32667794, -0.5590511, 0.76207],
                                   [-0.0412888, -0.7970912, -0.60245]])

        # Raumpunkt Gitter triangulieren
        # Bildkoordinaten  Gitter
        a3xN = np.float64([[gitterposL[0][0][0]],
                           [gitterposL[0][0][1]]])

        b3xN = np.float64([[gitterposR[0][0][0]],
                           [gitterposR[0][0][1]]])

        gitter = cv2.triangulatePoints(cls.p1[:3], cls.p2[:3], a3xN[:2], b3xN[:2])

        # homogen --> karthesisch
        gitter /= gitter[3]
        gitter = gitter[:3]
        print(f'\nGitter Raumpunkt:\n{gitter}')

        # Einheitsvektoren an die richtige Position verschieben
        gitter = np.tile(gitter.T, (4, 1))                          # zeilen vervielfachen
        systemzug = systemzug + gitter                              # Translation

        # Rotation und Translation berechnen und in Klassenvariablen schreiben
        systemcam = np.diag(np.float64([1, 1, 1]))                  # kanonische Einheitsvektoren
        systemcam = np.append([np.zeros(3)], systemcam, axis=0)     # erste Zeile = Ursprung

        # Rotation und Translation zwischen den beiden Bezugssystem berechnen
        cls.__R_approx, cls.__t_approx = rigid_transform_3D(systemcam, systemzug)
        cls.__rtstatus = 0

    @classmethod
    def reference(cls, refpts):
        """
        Referenz festlegen für das Koordinatensystem "Zug"

        :param refpts: Die auf einer Ebene quadratisch angeordneten Schrauben (numpy.shape = (4,3))
        :return: None
        """

        # Refpts sind die 3d Koordinaten der vier Gitter Schrauben punkte (ol, or, ur, ul)
        # Damit wird sowohl der Ursprung (gleich deren Mittelwert) als auch die Orientierung
        # des Zug-Koordinatensystems gegenüber dem Kamera-Welt Koordinatensystems bekannt.
        # Koordinaten sind karthesisch.

        assert refpts.shape == (4, 3)

        # Der Mittelwert der vier Eckpunkte ist der Ursprung des Zugskoordinatensystems
        # Mittelwert Achse 0
        m = np.average(refpts,0)

        # Die Vektoren zu den vier Eckpunkten abcd
        vma = refpts[0] - m
        vmb = refpts[1] - m
        vmc = refpts[2] - m
        vmd = refpts[3] - m
        print("Debug vma, vmb, vmc, vmd:")
        print(vma, vmb, vmc, vmb)

        # Die Ausrichtung anhand der Ebene bestimmen
        # ungefähr deshalb, weil der winkel zwischen x und y nicht in jedem Fall 90° beträgt
        # ungefähre x richtung
        x_wrong = vmb + vmc - vmd - vma

        # ungefähre y richtung
        y_wrong = vma + vmb - vmc - vmd

        #Z achse steht senkrecht darauf:
        z_ok = np.cross(x_wrong, y_wrong)

        #Winkelhalbierende zwischen den ungefähren x und y achsen
        xym = cls.bisect(x_wrong, y_wrong)

        #Achsen x und y mit den geforderten 90° Winkel erstellen
        tmp1 = np.cross(xym, z_ok)  # Hilfsvektoren
        x_ok = cls.bisect(tmp1, xym)
        y_ok = cls.bisect(-tmp1, xym)

        #Normieren und verschieben
        ex = x_ok / np.linalg.norm(x_ok) + m
        ey = y_ok / np.linalg.norm(y_ok) + m
        ez = z_ok / np.linalg.norm(z_ok) + m

        # Rotation und Translation berechnen und in Klassenvariablen schreiben
        systemcam = np.diag(np.float64([1, 1 , 1]))                 # kanonische Einheitsvektoren
        systemcam = np.append([np.zeros(3)], systemcam, axis=0)     # erste Zeile = Ursprung
        systemzug = np.stack((m, ex,ey,ez))                         # Usprung und kanonische Einheitsvektoren

        # Rotation und Translation zwischen den beiden Bezugssystem berechnen
        cls.__R_exact, cls.__t_exact = rigid_transform_3D(systemcam, systemzug)
        cls.__rtstatus = 1

        print("sysCam\n", systemcam)
        print("sysTrain\n", systemzug)
        print("R\n", Trainfeature.__R_exact)
        print("t\n", Trainfeature.__t_exact)


    def loadpatch(self, filename=None):
        if filename is not None:                    # optional kann ein anderes als das standardbild geladen werden
            self.patchfilename = filename
        print("Lade: ", self.patchfilename )
        self.patchimage = cv2.imread(self.patchfilename, cv2.IMREAD_GRAYSCALE)
        self.patchsize = self.patchimage.shape

    def __str__(self):
        s = f'\nClass Info:\n rt status: {self.__rtstatus}'
        s += f'\n R (exact):\n{self.__R_exact}\n t (exact):\n{self.__t_exact}\n'
        s += f' R (approx.):\n{self.__R_approx}\n t (approx.):\n{self.__t_approx}\n'
        s += f'\nObject info:\n Name: {self.name}\n Patchfilename: {self.patchfilename}\n'
        s += f' Real position center:\n{self.realpos}\n'
        s += f' Real position corners:\n{self.edges3d}\n'
        return s

if __name__ == '__main__':
# Tests zur Umrechnung zwischen den Bezugssystemen.

    # Triangulierte Schraubenmittelpunkte der Gitterschrauben.
    ref = np.array([[  -3.1058179e+02, -1.5248854e+02, 7.8082729e+03],
                      [ 1.3992499e+02, -2.6128412e+02, 7.9217915e+03],
                      [ 2.9599524e+02,  5.3110981e+00, 7.5579175e+03],
                      [-1.5471242e+02,  1.1419590e+02, 7.4451899e+03]])

    # A und B aus einem Unittest von RigidTransform:
    A = np.array([[0.19347454, 0.62539694, 0.47472073],
                  [0.73361547, 0.44604185, 0.440494],
                  [0.06122058, 0.51811031, 0.68447433],
                  [0.11634119, 0.91875987, 0.79474321],
                  [0.41273859, 0.37991864, 0.21993256],
                  [0.44945468, 0.87862294, 0.28273395],
                  [0.77345643, 0.52144629, 0.99183976],
                  [0.47187748, 0.90766783, 0.75452366],
                  [0.61793414, 0.33182565, 0.56279906],
                  [0.46996138, 0.84868544, 0.54318338]])

    B = np.array([[0.46791738, 1.71527172, 0.41685238],
                  [0.98888301, 1.49656549, 0.493366],
                  [0.30929093, 1.68465264, 0.63341236],
                  [0.39431055, 2.09509612, 0.62837356],
                  [0.6819297, 1.39049579, 0.26862063],
                  [0.76213075, 1.88150193, 0.18812946],
                  [0.99127965, 1.72937223, 1.00036753],
                  [0.74913118, 2.04774457, 0.63099096],
                  [0.85307579, 1.432212, 0.63048298],
                  [0.75851369, 1.92892881, 0.44675717]])

    if (1==1):
        # Testfall mit Kamera, Einheitsvektoren Zugsystem vorberechnet
        # camera system:
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        #zugsystem (Ursprung, ex, ey, ez)
        B = np.array([[-7.34349500, -73.5664155, 7683.29295],
                      [-6.39926158, -73.7946860, 7683.53026],
                      [-7.67017294, -74.1254666, 7684.05502],
                      [-7.38478380, -74.3635067, 7682.69050]])

    Trainfeature.__R_exact, Trainfeature.__t_exact = rigid_transform_3D(A, B)
    print("Rotation = \n", Trainfeature.__R_exact)
    print("Translation = \n", Trainfeature.__t_exact)

    # A --> rT --> B2 funktioniert
    B2 = (Trainfeature.__R_exact @ A.T) + np.tile(Trainfeature.__t_exact, (1, 4))
    B2 = B2.T

    # Test für den umgekehrten Weg B ---> A2
    A2 = (B - np.tile(Trainfeature.__t_exact, (1, 4)).T) @ Trainfeature.__R_exact

    print("\nReconstruct A2\n", A2)
    print("Reconstruct B2\n", B2)
    err1 = rmserror(A2, A)
    err2 = rmserror(B2, B)
    print("Error for each direction\n", err1,"\n", err2)


