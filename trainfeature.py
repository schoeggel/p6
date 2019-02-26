# Klasse für ein zu findendes / messendes Objekt oder Feature am Zug
import numpy as np
import cv2
from rigid_transform_3d import rigid_transform_3D, rmserror
import calibMatrix
from enum import Enum


class tm(Enum):
    IMAGE = 1           # Template in einem bildkanal, masken in anderen kanälen. Versuch TM mit transparen
    NOISE = 2           # Template auf Rauschen.
    AVERAGE = 3         # Template auf Hintergrund, der dem mittelwert des Templates entspricht
    CANNY = 4           # Template und Suchbereich durch canny edge detector laufen lassen
    ELSD = 5            # TODO: statt canny die ellipsen und linien erkennen.


class Trainfeature:
    # Die Koordinatentransformation <cam1 <--> zug> ist für alle Instanzen gleich
    # Die Werte werden einmalig pro Bildpaar gesetzt mit der Methode "reference"
    # oder ungefähr gesetzt mit der Methode "approxreference"

    PIXEL_PER_CLAHE_BLOCK = 50                      # Anzahl Blocks ist abhängig von der Bildgrösse

    p1 = None                                       # P Matrix für Triangulation
    p2 = None                                       # P Matrix für Triangulation
    __R_exact = np.diag([0, 0, 0])                  # Init Wert
    __t_exact = np.zeros(3)                         # Init Wert
    __R_approx = np.diag([0, 0, 0])                 # Init Wert
    __t_approx = np.zeros(3)                        # Init Wert
    __rtstatus = -1                                 # -1: keine, 0: approx, 1:exakte vorhanden


    def __init__(self, name, center3d, realsize):
        assert (len(name) > 0) and (center3d.shape == (3,)) and (realsize > 1)

        self.name = name  # Objektname
        self.tmmode = tm.IMAGE                      # In welcher Form werden die Bilder beim Matching verwendet.
        self.patchfilename = "data/patches/" + name + ".png"          # zum laden des patchbilds
        self.patchimageRaw = None                   # Das Bild im Originalzustand
        self.patchimage = None                      # Das Bild (Kontrastverbessert)
        self.patchCenter3d = center3d.astype(float) # Vektor 3d zum Patch Mitelpunkt im sys_zug
        self.realsize = realsize                    # Kantenlänge 3d des Patches.
        self.rotation = None                        # später: TODO : objekt kann auch schräg sein
        self.corners3Dtrain = np.zeros((5, 3))        # Vier Eckpunkte plus Mittelpunkt des Objekts als 3d Koord.
        self.corners2DimgL = np.zeros((1, 5, 2))      # Eckpunkte auf dem Bild. (plus Mitte)
        self.corners2DimgR = np.zeros((1, 5, 2))      # Eckpunkte auf dem Bild. (plus Mitte)
        self.corners2DtemplateL = np.zeros((1, 5, 2)) # Eckpunkte auf dem Template (plus Mitte)
        self.corners2DtemplateR = np.zeros((1, 5, 2)) # Eckpunkte auf dem Template (plus Mitte)
        self.warpedpatchL = None                    # das gewarpte template
        self.warpedpatchR = None                    # das gewarpte template
        self.wpShapeL = (-1,-1)                     # Grösse des verzerrten Templates.
        self.wpShapeR = (-1,-1)                     # Grösse des verzerrten Templates.
        self.wpMaskNormL = None                     # Maske für verzerrtes Template.
        self.wpMaskNormR = None                     # Maske für verzerrtes Template.
        self.wpMaskExtL = None                      # Maske für verzerrtes Template. Hintergrund leicht überlappend
        self.wpMaskExtR = None                      # Maske für verzerrtes Template. Hintergrund leicht überlappend
        self.measuredposition3d_zug = None          # Die gemessene Position im System Zug
        self.measuredposition3d_cam = None          # Die gemessene Position im System Kamera
        self.loadpatch()                            # default-Patch laden


    #liefert die eckpunkte für den suchbereich.
    # im Format [oly, ury, olx, urx]
    def getROIptsL(self, extend = 100):
        return self.getROIsingleSide(self.corners2DimgL, extend)

    def getROIptsR(self, extend = 100):
        return self.getROIsingleSide(self.corners2DimgR, extend)

    def getROIsingleSide(self, corners, extend):
        # Liefert den Suchbereich einer Seite
        minx, miny = corners.min(0)[0]
        maxx, maxy = corners.max(0)[0]
        olx, oly = minx - extend, miny - extend
        urx, ury = maxx + extend, maxy + extend
        return list(map(int, [oly, ury, olx, urx]))

    @staticmethod
    def separateRGB(img_in, vertical=False):
        # separiert die 3 Kanäle und legt sie nebeneinander oder übereinander
        r, g, b = img_in[:,:,0], img_in[:,:,1], img_in[:,:,2]
        if vertical:
            res = np.vstack((r,g,b))
        else:
            res = np.hstack((r,g,b))

        return res


    @staticmethod
    def filterScore(score_in):
        # macht gauss filter über map, um eindeutig maximum zu erhalten.
        # mappt die bild info auf einen range 0..1
        # TODO: die Auflösung könnte auch hochskaliert werden vor dem gauss filter, um
        # den peak sub-pixel-"genau" zu lokalisieren.

        # glätten (Filter Parameter wurden experimentell bestimmt)
        scoreSmooth = cv2.GaussianBlur(score_in, (9, 9), 2)

        # Kontrast verbessern: Das bild ist nicht uint8, cv2.equalizeHist() funktioniert nicht
        imin, imax = scoreSmooth.min(),scoreSmooth.max()
        scoreSmooth = np.interp(scoreSmooth, [imin, imax] , [0, 1])

        return scoreSmooth


    def drawBasis(self, img_in, sideLR = 0, length=100, thickness=20, show=False):
        # zeichnet die Basis des Zugssystems auf das Bild ein
        # RGB == XYZ (opencv draw: BGR)
        # SideLR = 0 : Links   |  SideLR = 1 : Rechts
        img = img_in.copy()

        # Basis aufstellen im system zug
        basis3d = np.diag(np.float64([length, length, length]))  # kanonische Einheitsvektoren
        basis3d = np.append([np.zeros(3)], basis3d, axis=0)  # erste Zeile = Ursprung

        # umrechnen ins Kamerasystem
        pts_cam = self.transformsys(basis3d, direction=0)

        # Projektion der Punkte in Bildpixelkoordinaten
        cal = calibMatrix.CalibData()
        if sideLR == 0:
            pts, jcb = cv2.projectPoints(pts_cam, cal.rl, cal.tl, cal.kl, cal.drl)
        elif sideLR == 1:
            pts, jcb = cv2.projectPoints(pts_cam, cal.rr, cal.tr, cal.kr, cal.drr)

        #Basis einzeichnen
        # pts im shape (4,1,2)
        pts = pts.astype(int)
        img = cv2.line(img, (pts[0][0][0], pts[0][0][1]), (pts[1][0][0], pts[1][0][1]), (0,0,255), thickness)
        img = cv2.line(img, (pts[0][0][0], pts[0][0][1]), (pts[2][0][0], pts[2][0][1]), (0,255,0), thickness)
        img = cv2.line(img, (pts[0][0][0], pts[0][0][1]), (pts[3][0][0], pts[3][0][1]), (255,0,0), thickness)

        if show:
            cv2.namedWindow('Basis', cv2.WINDOW_NORMAL)
            cv2.imshow("Basis", img)
            cv2.waitKey(0)

        return img



    def find(self, imageL, imageR, verbose=False, extend=100):
        # sucht das objekt im angegebenen Bild
        # Liefert die gemessene Position zurück (2d,3d), speichert gemessene 3d pos in Instanz

        assert (self.corners3Dtrain.sum != 0)

        ROIL = self.getROIptsL(extend)
        ROIR = self.getROIptsR(extend)

        # Kanal 0 erhält die Bildinfo
        # Nur Region of interest ausschneiden
        channel_0L = imageL[ROIL[0]:ROIL[1],ROIL[2]:ROIL[3],0]
        channel_0R = imageR[ROIR[0]:ROIR[1],ROIR[2]:ROIR[3],0]

        # Kanal 1 bleibt leer
        channel_1L = np.zeros(channel_0L.shape, dtype= np.uint8)
        channel_1R = np.zeros(channel_0R.shape, dtype= np.uint8)

        # Kanäle zu RGB Bild stapeln
        rgbL = np.dstack((channel_0L, channel_1L, channel_1L))
        rgbR = np.dstack((channel_0R, channel_1R, channel_1R))

        # CLAHE
        rgbL = self.clahe(rgbL, 0)
        rgbR = self.clahe(rgbR, 0)


        ####################  JE NACH METHODE SIND SPEZIFISCHE TEMPLATE/IMAGE VORBEREITUNGEN NÖTIG ##################

        if self.tmmode == tm.CANNY:
            # Kanten finden und template Hintergrund auf 0 setzen, inkl dem Rand zum Template (daher MaskEXT statt NORM)
            patchL = cv2.Canny(self.warpedpatchL, 80, 160)
            patchR = cv2.Canny(self.warpedpatchR, 80, 160)
            patchL[self.wpMaskExtL==False] = 0
            patchR[self.wpMaskExtR==False] = 0
            imageL = cv2.Canny(rgbL, 80, 160)
            imageR = cv2.Canny(rgbR, 80, 160)

        elif self.tmmode == tm.IMAGE:
            # Kanal 0 auf alle RGB erweitern
            patchL = np.dstack((self.warpedpatchL[:,:,0],self.warpedpatchL[:,:,0],self.warpedpatchL[:,:,0]))
            patchR = np.dstack((self.warpedpatchR[:,:,0],self.warpedpatchR[:,:,0],self.warpedpatchR[:,:,0]))
            imageL = np.dstack((rgbL[:,:,0],rgbL[:,:,0],rgbL[:,:,0]))
            imageR = np.dstack((rgbR[:,:,0],rgbR[:,:,0],rgbR[:,:,0]))

        elif self.tmmode == tm.NOISE:
            # Kanal 0 auf alle RGB erweitern, template hintergrund mit rauschen füllen
            imageL = np.dstack((rgbL[:, :, 0], rgbL[:, :, 0], rgbL[:, :, 0]))
            imageR = np.dstack((rgbR[:, :, 0], rgbR[:, :, 0], rgbR[:, :, 0]))
            valueL =  2 * rgbL[:, :, 0].mean()                                      # Mittlere Helligkeit im suchbereich
            valueR =  2 * rgbR[:, :, 0].mean()                                      # Mittlere Helligkeit im Suchbereich
            noiseL = (np.random.random(self.wpShapeL) * valueL).astype(np.uint8)
            noiseR = (np.random.random(self.wpShapeR) * valueR).astype(np.uint8)

            patchGreyL = self.warpedpatchL[:, :, 0]
            noiseL[self.wpMaskExtL] =  patchGreyL[self.wpMaskExtL]
            patchGreyR = self.warpedpatchR[:, :, 0]
            noiseR[self.wpMaskExtR] =  patchGreyR[self.wpMaskExtR]

            patchL = np.dstack((noiseL, noiseL, noiseL))
            patchR = np.dstack((noiseR, noiseR, noiseR))

        else:
            patchL = self.warpedpatchL
            patchR = self.warpedpatchR
            imageL = rgbL
            imageR = rgbR


        if verbose:
            cv2.namedWindow('rgbL', cv2.WINDOW_NORMAL)
            cv2.imshow("imageL", imageL)
            cv2.waitKey(0)

        # match L
        (centerx,centery), val = self.match(imageL, patchL, self.corners2DtemplateL[4], verbose=verbose)
        # center ist messpunkt relativ zum linken oberen Ecke der ROI
        # Umrechnen: center = (y,x), ROI : [oly, ury, olx, urx]
        centerxyL = (centerx + ROIL[2], centery + ROIL[0])

        # Match R
        (centerx,centery), val = self.match(imageR, patchR, self.corners2DtemplateR[4], verbose=verbose)
        centerxyR = (centerx + ROIR[2], centery + ROIR[0])


        # Triangulieren
        print(f'Trianguliere diese beiden Punkte: {centerxyL} und {centerxyR}')
        # Bild pixel koordinaten der Objekt Zentren
        a3xN = np.float64([[centerxyL[0]],
                           [centerxyL[1]]])

        b3xN = np.float64([[centerxyR[0]],
                           [centerxyR[1]]])

        # koordinaten trangulieren und umformen homogen --> kathesisch
        self.measuredposition3d_cam = cv2.triangulatePoints(self.p1[:3], self.p2[:3], a3xN[:2], b3xN[:2])
        self.measuredposition3d_cam /= self.measuredposition3d_cam[3]

        # System Cam --> System Zug
        self.measuredposition3d_zug = self.transformsys(self.measuredposition3d_cam[:3].T, direction=1)

        return centerxyL, val



    def clahe(self, img, channel=None):
        # verbessert den kontrast im angegebenen Kanal

        # Kanal aus RGB separieren, falls RGB Bild (Kanalnr als Arg. vorhanden)
        if channel is not None:
            ch = img[:,:,channel]
        else:
            ch = img.copy()

        # grid ist abhängig von der bildgrösse aber immer mind. 2x2
        px = 0.5 * (ch.shape[0] + ch.shape[1])
        grid  = round(px / self.PIXEL_PER_CLAHE_BLOCK)
        if grid < 2: grid = 2

        # Kontrast verbessern
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(grid,grid))
        ch = clahe.apply(ch)

        # Verbesserter Kanal wieder in RGB-Bild einfügen, falls RGB
        if img.ndim == 3:
            res = img.copy()
            res[:,:,channel] = ch
        else:
            res = ch

        return res



    def match(self, img_in, template_in, patchcenter, verbose=False):
        # Rückgabewerte: beste Position und Konfidenz
        # code kopiert aus opencv tutorial
        # patchcenter = Mitte des Patchs. TM Resultat bezieht sich auf die Ecke

        # Gemäss Versuchsauswertung die am besten geeignet bei multikanal mit maske: CCORR_NORMED
        #method = cv2.TM_CCORR_NORMED
        if self.tmmode in [tm.NOISE, tm.IMAGE, tm.AVERAGE]:
            method = cv2.TM_CCOEFF_NORMED
        else:
            method = cv2.TM_CCORR_NORMED

        template = template_in.copy()
        img = img_in.copy()

        # falls rgb und greyscale gemischt kommen
        if img.ndim == 3 and template.ndim == 2:
            template = np.dstack((template, template, template))
        elif img.ndim == 2 and template.ndim == 3:
            img = np.dstack((img, img, img))

        # Apply template Matching
        # "location" ist im Format (x,y), wie auch "offset" und "center"
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            val = min_val
        else:
            top_left = max_loc
            val = max_val

        # Versatz von der Ecke des Templates zur Mitte des Templates berücksichtigen
        location = top_left + patchcenter
        location = tuple(location.astype(int))

        if verbose:
            print(f'Template.shape: {template.shape}')
            print(f'image.shape: {img.shape}')
            cv2.imshow("matchingImage", cv2.drawMarker(img, location, (30,30,255), cv2.MARKER_CROSS, 10,1))
            tmp = cv2.drawMarker(template, tuple(patchcenter.astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 8, 1)
            cv2.namedWindow('matchingTemplate', cv2.WINDOW_NORMAL)
            cv2.imshow("matchingTemplate",tmp)
            cv2.imwrite('tmp/templatedebug.png', template)
            cv2.imwrite('tmp/templateImgdebug.png', img)
            print(f'Top-Left: {top_left} ; Offset PatchCenter: {patchcenter} ; Template Center: {location}')
            res = self.filterScore(res)
            res = cv2.circle(res, top_left,8, 0, 1)
            res = cv2.circle(res, top_left, 9, 255, 1)
            cv2.namedWindow('scoremap', cv2.WINDOW_NORMAL)
            cv2.imshow("scoremap", res)
            print(res.shape)
            cv2.waitKey(0)

        return location, val

    def createMasks(self):
        # NEU ANDERS HERUM: WEISSES POLYGON AUF SCHWARZ
        # 0 = Hintergrund, 255 = Bildinformation verzerrtes Template

        # LINKE SEITE
        pt = self.polygonpoints(self.corners2DtemplateL)
        mask = (np.zeros(self.wpShapeL)).astype(np.uint8)                      # weil gilt : y, x = a.shape
        mask = cv2.fillConvexPoly(mask, pt, 255)
        self.wpMaskNormL = mask == 255
        mask = cv2.polylines(mask, pt, isClosed=True, color=0, thickness=3)       # Maske leicht überlappend machen
        self.wpMaskExtL = mask == 255

        # RECHTE SEITE
        pt = self.polygonpoints(self.corners2DtemplateR)
        mask = (np.zeros(self.wpShapeR)).astype(np.uint8)                      # weil gilt : y, x = a.shape
        mask = cv2.fillConvexPoly(mask, pt, 255)
        self.wpMaskNormR = mask == 255
        mask = cv2.polylines(mask, pt, isClosed=True, color=0, thickness=3)       # Maske leicht überlappend machen
        self.wpMaskExtR = mask == 255


    def warp(self):
        # der Patch wird perspektivisch verzerrt, damit er so aussieht wie auf dem Bild erwartet
        # Eckpunkte des quadratischen Patchs (wie gespeichert, Vogelperspektive, quadratisch)
        d = self.patchimage.shape[0]
        quadrat = np.float32([[0, 0], [d, 0], [0, d], [d, d]])

        # Eckpunkte Pixelkoordinaten (x,y) für beide Bilder L,R berechnen
        self.reprojectedges()

        # Die Leinwand für das transformierte Bild wird so gross, dass der verzerrte Patch exakt hineinpasst
        minxl, minyl = self.corners2DimgL.min(0)[0]
        maxxl, maxyl = self.corners2DimgL.max(0)[0]
        minxr, minyr = self.corners2DimgR.min(0)[0]
        maxxr, maxyr = self.corners2DimgR.max(0)[0]
        self.wpShapeL = (int(maxyl-minyl), int(maxxl-minxl))
        self.wpShapeR = (int(maxyr-minyr), int(maxxr-minxr))

        # patch eckpunkte auf neue Leinwand umrechen (x und y minima pro Seite von pixelkoordinate subtrahieren)
        ofsl = np.float32([minxl, minyl])   # Offset Linkes Bild
        ofsr = np.float32([minxr, minyr])   # Offset Rechtes Bild
        self.corners2DtemplateL = self.corners2DimgL - np.tile(ofsl, (5, 1, 1))
        self.corners2DtemplateR = self.corners2DimgR - np.tile(ofsr, (5, 1, 1))
        self.corners2DtemplateL = self.corners2DtemplateL[:, 0].astype(np.float32)
        self.corners2DtemplateR = self.corners2DtemplateR[:, 0].astype(np.float32)

        # Masken erstellen
        self.createMasks()

        # Transformation am Bild durchführen. Datentyp der Punkte muss float32 sein, sonst b(l)ockt open cv
        # für die Ermittlung von M nur die 4 Ecken ohne Zentrum verwenden
        ML = cv2.getPerspectiveTransform(quadrat, self.corners2DtemplateL[:4])
        MR = cv2.getPerspectiveTransform(quadrat, self.corners2DtemplateR[:4])
        imgL = cv2.warpPerspective(self.patchimage, ML, (self.wpShapeL[1], self.wpShapeL[0]))   # dSize ist (x,y)
        imgR = cv2.warpPerspective(self.patchimage, MR, (self.wpShapeR[1], self.wpShapeR[0]))   # dSize ist (x,y)

        # template ist noch greyscale. rgb Version erstellen mit unterschiedlichen Informationen pro Kanal. Siehe Doku
        # grey in den Kanal 0 kopieren.
        channel_0L = imgL.copy()
        channel_0R = imgR.copy()

        # Kanal 1: Maske Template (0 wenn es sich um den zu ignorierenden Bereich des Templates handelt)
        channel_1L = np.zeros(self.wpShapeL, dtype=np.uint8)
        channel_1R = np.zeros(self.wpShapeR, dtype=np.uint8)
        channel_1L[self.wpMaskNormL] = 255
        channel_1R[self.wpMaskNormR] = 255

        # Kanal 2: Bleibt Nuller (wird nur im zu durchsuchenden Bild verwendet, nicht im Template)
        channel_2L = np.zeros(self.wpShapeL, dtype=np.uint8)
        channel_2R = np.zeros(self.wpShapeR, dtype=np.uint8)

        # Kanäle stapeln
        rgbL = np.dstack((channel_0L, channel_1L, channel_2L))
        rgbR = np.dstack((channel_0R, channel_1R, channel_2R))

        self.warpedpatchL = rgbL
        self.warpedpatchR = rgbR
        return rgbL, rgbR


    def showpatch(self):
        # zeigt den geladenen und fall vorhanden die gewarpten patches an
        cv2.namedWindow(f'patch as loaded ({self.patchfilename})', cv2.WINDOW_NORMAL)
        cv2.imshow(f'patch as loaded ({self.patchfilename})', self.patchimage)
        if self.warpedpatchL is not None:
            cv2.namedWindow('patch warp L', cv2.WINDOW_NORMAL)
            cv2.imshow("patch warp L", self.warpedpatchL)
        if self.warpedpatchR is not None:
            cv2.namedWindow('patch warp R', cv2.WINDOW_NORMAL)
            cv2.imshow("patch warp R", self.warpedpatchR)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    @staticmethod
    def polygonpoints(edges):
        # oben links oben rechts unten links unten rechts --> punkte für fillConvexPoly
        # fillConvexpoly vertausch die Punkt x und y --> hier korrigieren ( [::-1] tauscht)
        poly = np.zeros((1, 4, 2), dtype=np.int32)
        poly[0][0] = edges[0]
        poly[0][1] = edges[1]
        poly[0][2] = edges[3]
        poly[0][3] = edges[2]
        return poly

    def reprojectedges(self):
        # rechnet die Patch Ecken in x,y Pixelkoordinaten um
        # und speicher diese in der Instanz

        # Koordinaten der Patch Ecken rechnen (sys_zug)
        self.calculatePatchCorners3d()

        # ecken umrechnen sys_zug --> sys_cam (direction = 0)
        edges3d_cam = self.transformsys(self.corners3Dtrain, 0)

        # reprojection der Punkte in Bildpixelkoordinaten
        cal = calibMatrix.CalibData()
        left, jcb = cv2.projectPoints(edges3d_cam, cal.rl, cal.tl, cal.kl, cal.drl)
        right, jcb = cv2.projectPoints(edges3d_cam, cal.rr, cal.tr, cal.kr, cal.drr)
        print(f'projection LEFT:\n{left}\n\nprojection RIGHT:\n{right}')

        # opencv liefert die punkte im shape (n,1,3) zurück.
        self.corners2DimgL = left
        self.corners2DimgR = right
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


    def calculatePatchCorners3d(self):
        # Ausgehend von der Grösse des quadratischen Patchs und dessen Zentrumskoordinaten
        # werden die Koordinaten der vier Eckpunkte berechnet. Bezugssystem: sys_zug
        # Ohne Rotation liegt der Patch auf xy Ebene mit dem Zentrum des Quadrats bei patchCenter3d
        if self.rotation is not None:
            print("Warnung, Rotation des Templates ist nicht nicht implementiert.") # TODO

        # Patchmitte bis Patch Rand (in x oder y Richtung)
        d = self.realsize / 2

        # Alle Ecken erhalten vorerst den Mittelpunkt als Koordinaten
        self.corners3Dtrain = np.tile(self.patchCenter3d, (5, 1))

        # Patchmitte bis Patch Ecken, die Differenz vom Mittelpunkt zur Ecke
        d = np.array([[-d, +d, 0],          # oben links
                      [+d, +d, 0],          # oben rechts
                      [-d, -d, 0],          # unten links
                      [+d, -d, 0],          # unten rechts
                      [ 0,  0, 0]])         # Mitte
        self.corners3Dtrain += d


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
    def referenceViaObjects(cls, tfol, tfor, tfur, tful):
        """
        Referenz festlegen für das Koordinatensystem "Zug"

        :param objects: Die auf einer Ebene quadratisch angeordneten Schrauben (als Trainfeature Objekte)
        :return: None
        """
        refpts = np.zeros((4,3))
        refpts[0] = tfol.measuredposition3d_cam[:3].flatten()
        refpts[1] = tfor.measuredposition3d_cam[:3].flatten()
        refpts[2] = tfur.measuredposition3d_cam[:3].flatten()
        refpts[3] = tful.measuredposition3d_cam[:3].flatten()
        cls.reference(refpts)




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
        print(vma, vmb, vmc, vmd)

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

        # Rotation und Translation zwischen den beiden Bezugssystem berechnen und in Klassenvariablen schreiben
        systemcam = np.diag(np.float64([1, 1 , 1]))                 # kanonische Einheitsvektoren
        systemcam = np.append([np.zeros(3)], systemcam, axis=0)     # erste Zeile = Ursprung
        systemzug = np.stack((m, ex,ey,ez))                         # Usprung und kanonische Einheitsvektoren
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
        self.patchimageRaw = cv2.imread(self.patchfilename, cv2.IMREAD_GRAYSCALE)

        # Kontrastverbesserte Variante
        self.patchimage = self.clahe(self.patchimageRaw)


    def __str__(self):
        s = f'\nClass Info:\n rt status: {self.__rtstatus}'
        s += f'\n R (exact):\n{self.__R_exact}\n t (exact):\n{self.__t_exact}\n'
        s += f' R (approx.):\n{self.__R_approx}\n t (approx.):\n{self.__t_approx}\n'
        s += f'\nObject info:\n Name: {self.name}\n Patchfilename: {self.patchfilename}\n'
        s += f' Real position center:\n{self.patchCenter3d}\n'
        s += f' Real position corners:\n{self.corners3Dtrain}\n'
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


