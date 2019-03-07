# WTM : Warped Template Matching
from cvaux import imgMergerV, imgMergerH, putBetterText, separateRGB
import cv2
import wtmCalib
import wtmObject
import wtmScene
import wtmEnum

class Composition:
    _trainIsReversed = False             #TODO Der Zug könnte auch verkehrt herum einfahren?
    _scenes = []

    imagePairs = []                     # eine Liste mit Bildpaaren in der Form [[01L,01R], [02L, 02R] , ... ]
    calib = wtmCalib.CalibData()
    refObj = [None] * 4                 # Die Gitterschrauben Objekte werden separat geführt.
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
        assert imagePairs is not None and refObj is not None
        # todo: prüfen der Liste, ob die Bilder existieren etc..
        if tmmode is not None:
            self._tmmode = tmmode
        self.imagePairs = imagePairs
        self.refObj = refObj
        self.createScenes()

    def createScenes(self):
        # Für jedes Bildpaar wird eine scene erstellt und in der Liste angefügt
        for pair in self.imagePairs:
            imgL = cv2.imread(pair[0])
            imgR = cv2.imread(pair[1])
            newScene = wtmScene.Scene(self,imgL, imgR)
            self._scenes.append(newScene)

    # Alle angegebenen Objekte messen
    def measureObjects(self, objects_list: list):
        pass

    # auf allen scenes das objekt suchen, vermessen
    def measureObject(self, oneObject):
        oneObject: wtmObject.MachineObject = oneObject
        pass

    # Ein Objekt in einer Scene lokalisieren
    def measureObjectInScene(self, oneObject, scene):
        oneObject: wtmObject.MachineObject = oneObject
        scene: wtmScene.Scene = scene
        res = scene.locate(oneObject, self)

    def sceneinfo(self, n:int = None):
        if n is None:
            n = range(len(self._scenes))
        else:
            n = [n]

        for i in n:
            print(f'Scene [{i}]:')
            try:
                scene:wtmScene.Scene = self._scenes[i]
                print(scene)
            except (TypeError, IndexError):
                print(f'Scene not found.')

    def __str__(self):
        return f"""
        {self.calib}\n
        composition containg {len(self.imagePairs)} image pair(s) 
        and {len(self.refObj)} reference objects. 
        active template matching mode (tmmode) is {self.tmmode}
        use .sceneinfo() for scene details.
        """


if __name__ == '__main__':
    fn1 = "SBB/13L.png"
    fn2 = "SBB/13R.png"
    test = Composition([[fn1, fn2], [fn1, fn2]], [])
    print(test)
    test.sceneinfo()
    test.sceneinfo(9999.323)
    test.sceneinfo(20000)



