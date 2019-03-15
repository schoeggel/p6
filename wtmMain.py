# WTM : Warped Template Matching

import numpy as np
from wtmComposition import Composition
from wtmObject import MachineObject
import wtmFactory
from wtmEnum import tm

############################# OBJEKTE #######################################
patchName = "data/patches/3dcreatorSet1/"
# Erstellt ein einfaches Objekt
xyz = np.array([-312, 128, 4])
size = (36, 37)
s1 = MachineObject(patchName + "tcr3dschraubeKleinGanzLinkeSeite", xyz, size)
print(s1)


####################### NEU DIE OBJEKTE PER FACTORY HOLEN #####################
allRefs, allObjects =  wtmFactory.createMachineObjects()


############################# COMPOSITION #######################################
fn = [["SBB/8L.png", "SBB/8R.png"], ["SBB/9L.png", "SBB/9R.png"], ["SBB/10L.png", "SBB/10R.png"],
      ["SBB/11L.png", "SBB/11R.png"], ["SBB/12L.png", "SBB/12R.png"], ["SBB/13L.png", "SBB/13R.png"],
      ["SBB/14L.png", "SBB/14R.png"], ["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"],
      ["SBB/17L.png", "SBB/17R.png"],["SBB/18L.png", "SBB/18R.png"]]
#fn = [["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"]]
#fn = [["SBB/13L.png", "SBB/13R.png"],["SBB/14L.png", "SBB/14R.png"],["SBB/15L.png", "SBB/15R.png"]]
#fn = [["SBB/15L.png", "SBB/15R.png"],["SBB/16L.png", "SBB/16R.png"],["SBB/17L.png", "SBB/17R.png"],["SBB/18L.png", "SBB/18R.png"]]

cp = Composition(fn, allRefs, tmmode=tm.CANNYBLUR)
print(cp)
cp.sceneinfo()

#cp._scenes[0].drawOrigin(length=90, show=True, mirror=False)
#cp._scenes[1].drawOrigin(length=150, show=True, mirror=False)
#cp._scenes[-1].drawOrigin(length=1500, show=True, mirror=True)
#cp._locateSingleObject(allObjects[0], extend=200,verbose=True)
#exit(0)

    # vieles messen
if 1==0:
    idx = int(input(f'welches Objekt vermessen? -1 für alle: '))
    if idx > -1:
        allObjects = [allObjects[idx]]

cp.locateObjects(allObjects, verbose=False)
print("ok")

for o in allObjects:

    o:MachineObject
    print("--------------------------------------------------------------------")
    print(o)
    print(f'Good Positions:\n{o.positions}')
    print(f'Rejected Positions:\n{o.rejectedPositions}')
    print(f'mean position: {o.avgPosMac}')
    print(f'rms error: {o.rmserror}')
    print(f'std dev per axis: {np.std(o._positionsAsNumpyArray()[0], 0)}')
    #o.showBadSnapshots()
    #o.showGoodSnapshots()
    o.saveBadSnapshots()
    o.saveGoodSnapshots()




