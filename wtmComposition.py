# WTM : Warped Template Matching
from cvaux import imgMergerV, imgMergerH, putBetterText, separateRGB
import calibMatrix
import wtmObject
import wtmScene
import wtmEnum

class Composition:
    _trainIsReversed = False             #TODO Der Zug könnte auch verkehrt herum einfahren?
    _scenes = []

    imagePairs = []                     # eine Liste mit Bildpaaren in der Form [[01L,01R], [02L, 02R] , ... ]
    calib = calibMatrix.CalibData()
    refObj = [None] * 4
    p1 = None  # P Matrix für Triangulation
    p2 = None  # P Matrix für Triangulation
    tmmode = wtmEnum.tm.CANNYBLUR  # Standard TM Mode

    PRE_TM_K_SIZE = 5
    PRE_TM_PX_PER_K = 15  # warpedTemplate hat Abmessung 50x50 px --> K = (5,5)
    SCOREFILTER_K_SIZE = 5  # Kernelgrösse für die Glättung des TM Resultats (Score)
    PIXEL_PER_CLAHE_BLOCK = 50  # Anzahl Blocks ist abhängig von der Bildgrösse



    def __init__(self, imagePairs:list, refObj:list, tmmode:wtmEnum.tm = None):
        """Lädt die Bildpaare und Referenzobjekte.

            Die Referenzobjekt sind die vier Gitterschrauben.
            Deren Reihenfolge innerhalb der Liste ist vorgegeben:
            Oben links, oben rechts, unten rechts, unten links.
            """
        # todo: prüfen der Liste, ob die Bilder existieren etc..
        if tmmode is not None:
            self._tmmode = tmmode
        self.imagePairs = imagePairs
        self.refObj = refObj

    # Alle angegebenen Objekte messen
    def measureObjects(self, objects_list:list):
        pass

    # auf allen scenes das objekt suchen, vermessen
    def measureObject(self, oneObject:wtmObject.MachineObject):
        pass

    # Ein Objekt in einer Scene lokalisieren
    def measureObjectInScene(self,oneObject:wtmObject.MachineObject, scene:wtmScene.Scene):
        pass
