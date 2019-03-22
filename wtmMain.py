# WTM : Warped Template Matching

import numpy as np
from wtmComposition import Composition
from wtmObject import MachineObject
import wtmFactory
from wtmEnum import tm
import wtmScene

############################# OBJEKTE #######################################
patchName = "data/patches/3dcreatorSet1/"
# Erstellt ein einfaches Objekt
xyz = np.array([-312, 128, 4])
size = (36, 37)
s1 = MachineObject(patchName + "tcr3dschraubeKleinGanzLinkeSeite", xyz, size)
print(s1)


####################### NEU DIE OBJEKTE PER FACTORY HOLEN #####################
allRefs, allObjects =  wtmFactory.createMachineObjects()
3

############################# COMPOSITION #######################################
#fn = [["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"]]
#fn = [["SBB/13L.png", "SBB/13R.png"],["SBB/14L.png", "SBB/14R.png"],["SBB/15L.png", "SBB/15R.png"]]
#fn = [["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"],["SBB/17L.png", "SBB/17R.png"],["SBB/18L.png", "SBB/18R.png"]]

# einsetzen von "fremden" Bilder in eine Serie: sfm funktioniert nicht.
fn = [["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"],["SBB/17L.png", "SBB/17R.png"],
      ["SBB/18L.png", "SBB/18R.png"],["SBB/19L.png", "SBB/19R.png"],["SBB/20L.png", "SBB/20R.png"], ["SBB/45-OK1L.png", "SBB/45-OK1R.png"]]

fn = [["SBB/115-OK1L.png", "SBB/115-OK1R.png"]]
fn = [["SBB/123-OK1L.png", "SBB/123-OK1R.png"]]

fn = [["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"],["SBB/17L.png", "SBB/17R.png"],
      ["SBB/18L.png", "SBB/18R.png"],["SBB/19L.png", "SBB/19R.png"],["SBB/20L.png", "SBB/20R.png"]]

fn = [["SBB/8L.png", "SBB/8R.png"], ["SBB/9L.png", "SBB/9R.png"], ["SBB/10L.png", "SBB/10R.png"],
      ["SBB/11L.png", "SBB/11R.png"], ["SBB/12L.png", "SBB/12R.png"], ["SBB/13L.png", "SBB/13R.png"],
      ["SBB/14L.png", "SBB/14R.png"], ["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"],
      ["SBB/17L.png", "SBB/17R.png"],["SBB/18L.png", "SBB/18R.png"],["SBB/19L.png", "SBB/19R.png"],["SBB/20L.png", "SBB/20R.png"]]

cp = Composition(fn, allRefs, tmmode=tm.CANNYBLUR2)
s: wtmScene.Scene = cp._scenes[0]
s.drawOrigin(show=True)


print(cp)
cp.sceneinfo()

#cp._scenes[0].drawOrigin(length=90, show=True, mirror=False)
#cp._scenes[1].drawOrigin(length=150, show=True, mirror=False)
#cp._scenes[-1].drawOrigin(length=1500, show=True, mirror=True)
#cp._locateSingleObject(allObjects[0], extend=200,verbose=True)
#exit(0)

    # vieles messen
verbose = False
if 1==0:
    idx = int(input(f'welches Objekt vermessen? -1 für alle: '))
    if idx > -1:
        allObjects = [allObjects[idx]]
        verbose = True


cp.locateObjects(allObjects, verbose=verbose, export=True)
cp.locateObjects(allRefs, verbose=verbose, export=True)

combi = allObjects + allRefs
for o in combi:

    o:MachineObject
    print("--------------------------------------------------------------------")
    print(o)
    #o.showBadSnapshots()
    #o.showGoodSnapshots()
    #o.exportBadSnapshots()
    #o.exportGoodSnapshots()




