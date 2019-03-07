# WTM : Warped Template Matching

import numpy as np
from wtmComposition import Composition
from wtmObject import MachineObject
import wtmFactory

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
fn = [["SBB/11L.png", "SBB/11R.png"], ["SBB/12L.png", "SBB/12R.png"], ["SBB/13L.png", "SBB/13R.png"], ["SBB/14L.png", "SBB/14R.png"], ["SBB/15L.png", "SBB/15R.png"]]
cp = Composition(fn, [s1, s1, s1, s1])
print(cp)
cp.sceneinfo()
#cp.locateObjectInScene(s1, cp._scenes[0])
cp.locateObject(s1)
print(s1.positions)

# vieles messen
cp.locateObjects(allObjects)
for o in allObjects:
    o:MachineObject
    print(o.positions)
    print(o.avgPosMac)
    print(o.rmserror)
    print(np.std(o._positionsAsNumpyArray()[0], 0))




