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
        self.refScene = None            # Eine Scene, bei der die Transformatione zwischen cam und mac bekannt ist.
        if tmmode is not None:
            self._tmmode = tmmode
        self.imagePairs = imagePairs
        self.refObj = refObj
        self.createScenes()
        if self.refScene is not None:
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
        # .refScene enthält eine gültige Referenz fürs System machine.
        newScene = None
        for imgL,imgR in self.imagePairs:
            newScene = wtmScene.Scene(self,imgL, imgR, prevScene = newScene)
            self._scenes.append(newScene)

            if len(self.refObj) == 4 and newScene.rtstatus == wtmEnum.rtref.APPROX:
                singleUseRefObjects = copy.deepcopy(self.refObj)    # auf diesen Objekten nur einmal messen
                self.locateObjectInScene(singleUseRefObjects, newScene, verbose= False)
                newScene.referenceViaObjects(singleUseRefObjects)   # exakte Transformation Rt finden

            if newScene.rtstatus in [wtmEnum.rtref.BYOBJECT]:
                self.refScene = newScene

    def locateObjects(self, objects_list: list, verbose=False):
        """"Alle angegebenen Objekte auf allen scenes messen"""
        for obj in objects_list:
            print(obj)
            self.locateObject(obj, verbose=verbose)

    def locateObject(self, oneObject, verbose=False):
        """Auf allen scenes das eine Objekt lokalisieren"""
        oneObject: wtmObject.MachineObject = oneObject
        for s in self._scenes:
            s:wtmScene.Scene = s
            try:
                s.locate(oneObject, verbose=verbose)
            except:
                print(f'Could not locate Object {oneObject.patchfilenameL} in scene {s.photoNameL}' )


    def locateObjectInScene(self, oneOrMoreObjects, scene, verbose=False):
        """ Ein einzelnes Objekt (oder mehrere) in einer einzelnen Scene lokalisieren
            TODO: Fehlerhandling verbessern."""
        scene: wtmScene.Scene = scene
        if type(oneOrMoreObjects) is not list:
            oneOrMoreObjects = [oneOrMoreObjects]
        for oneObject in oneOrMoreObjects:
            oneObject:wtmObject.MachineObject = oneObject
            try:
                pos = scene.locate(oneObject, verbose=verbose)
            except:
                print(f'Could not locate Object {oneObject.patchfilenameL} in scene {scene.photoNameL}' )

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
        sc:wtmScene.Scene =  self.refScene
        while not sc.isLast:
            if sc.next.rtstatus in [wtmEnum.rtref.NONE, wtmEnum.rtref.APPROX]:
                dR, dt = wtmSfm.sfm(sc.photoL,sc.photoR,sc.next.photoL,sc.next.photoR, self.calib, verbose= False)
                sc.next.R_exact = sc.R_exact
                sc.next.t_exact = sc.t_exact - dt
                sc.next.rtstatus = wtmEnum.rtref.BYSFM
            sc = sc.next

        # einmal rückwärts
        sc: wtmScene.Scene = self.refScene
        while not sc.isFirst:
            if sc.prev.rtstatus in [wtmEnum.rtref.NONE, wtmEnum.rtref.APPROX]:
                dR, dt = wtmSfm.sfm(sc.photoL, sc.photoR, sc.prev.photoL, sc.prev.photoR, self.calib, verbose=False)
                sc.prev.R_exact = sc.R_exact
                sc.prev.t_exact = sc.t_exact - dt
                sc.prev.rtstatus = wtmEnum.rtref.BYSFM
            sc = sc.prev


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



