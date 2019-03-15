# WTM : Warped Template Matching
import wtmCalib
import wtmObject
import wtmScene
import wtmEnum
import wtmSfm
import copy

class Composition:
    _trainIsReversed = False             #TODO Der Zug könnte auch verkehrt herum einfahren?
    _scenes = []

    imagePairs = []                     # eine Liste mit Bildpaaren in der Form [[01L,01R], [02L, 02R] , ... ]
    calib = wtmCalib.CalibData()
    refObj = [None] * 4                 # Die Gitterschrauben Objekte werden separat geführt.
    tmmode = wtmEnum.tm.CANNYBLUR       # Standard TM Mode

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
        self.sceneWithExaktRt = None            # Eine Scene, bei der die Transformatione zwischen cam und mac bekannt ist.
        if tmmode is not None:
            self._tmmode = tmmode
        self.imagePairs = imagePairs
        self.refObj = refObj
        self.createScenes()
        if self.sceneWithExaktRt is not None:
            self.sfmScenes()            # Transformation versuchen über sfm zu erhalten

    @property
    def status(self):
        """Zeigt den Status der scenes an: Ob die Gitterposition in Ordnung ist und ob
            eine exakte Referenz gesetzt werden konnte."""
        return ...

    def createScenes(self):
        """Für jedes Bildpaar wird eine scene erstellt und in der Liste angefügt
            Konnte beim Initialisieren der scene ein Gitter gefunden werden,
            so kann die scene ab jetzt Objekte lokalisieren."""
        # Die scenes werde untereinander verlinkt mit .next und .prev.
        # .sceneWithExaktRt enthält eine gültige Referenz fürs System machine.
        newScene = None
        for imgL,imgR in self.imagePairs:
            newScene = wtmScene.Scene(self,imgL, imgR, prevScene = newScene)
            self._scenes.append(newScene)

            if len(self.refObj) == 4 and newScene.rtstatus == wtmEnum.rtref.APPROX:
                singleUseRefObjects = copy.deepcopy(self.refObj)    # auf diesen Objekten nur einmal messen
                self.locateObjects(singleUseRefObjects, newScene, verbose= False)
                newScene.referenceViaObjects(singleUseRefObjects)   # exakte Transformation Rt finden

            if newScene.rtstatus in [wtmEnum.rtref.BYOBJECT]:
                self.sceneWithExaktRt = newScene

    def locateObjects(self, oneOrMoreObjects, scenes =None, roiScale=1.25, verbose=False):
        """"Alle angegebenen Objekte auf allen angegbenen scenes messen
        :param oneOrMoreObjects: Ein einzelnes MachineObjekt oder eine Liste
        :param scenes: Optional eine einzelne Scene oder eine Liste. Ohne Angabe werden alle scenes verwendet"""

        if type(oneOrMoreObjects) is not list:
            oneOrMoreObjects = [oneOrMoreObjects]

        if scenes is None:
            scenes = self._scenes
        elif type(scenes) is not list:
            scenes = [scenes]

        for oneObject in oneOrMoreObjects:
            for s in scenes:
                try:
                    s.locate(oneObject, roiScale=roiScale, verbose=verbose)
                    print(f'locate: successfully located <{oneObject.name}> in scene <{s.name}>')
                except:
                    print(f'locate: failed to locate <{oneObject.name}> in scene <{s.name}>')


    def sceneinfo(self, n:int = None):
        """Informationen zu einer bestimmten scene anzeigen"""
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

    def sfmScenes(self):
        """findet die scene, bei der die Zug Referenz nicht mehr über das Gitter gesetzt werden konnte
            und setzt die Referenz anhand der smf mit dem bildpaar mit der letzten gültigen referenz."""
        # einmal vorwärts:
        sc: wtmScene.Scene = self.sceneWithExaktRt
        while not sc.isLast:
            if sc.next.rtstatus in [wtmEnum.rtref.NONE, wtmEnum.rtref.APPROX]:
                print(f'\nsfm: calculate t:  {sc.name} ------> {sc.next.name}')
                _, dt = wtmSfm.sfm(sc.photoL,sc.photoR,sc.next.photoL,sc.next.photoR, self.calib, verbose= False)
                sc.next.R_exact = sc.R_exact
                sc.next.t_exact = sc.t_exact - dt
                sc.next.rtstatus = wtmEnum.rtref.BYSFM
            sc = sc.next

        # einmal rückwärts
        sc: wtmScene.Scene = self.sceneWithExaktRt
        while not sc.isFirst:
            if sc.prev.rtstatus in [wtmEnum.rtref.NONE, wtmEnum.rtref.APPROX]:
                print(f'\nsfm: calculate t:  {sc.name} ------> {sc.prev.name}')
                _, dt = wtmSfm.sfm(sc.photoL, sc.photoR, sc.prev.photoL, sc.prev.photoR, self.calib, verbose=False)
                sc.prev.R_exact = sc.R_exact
                sc.prev.t_exact = sc.t_exact - dt
                sc.prev.rtstatus = wtmEnum.rtref.BYSFM
            sc = sc.prev


    def __str__(self):
        return f"""
        {self.calib}\n
        composition contains {len(self.imagePairs)} image pair(s) 
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



