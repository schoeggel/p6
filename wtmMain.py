# WTM : Warped Template Matching

import numpy as np
from wtmComposition import Composition
from wtmObject import MachineObject

patchName = "data/patches/3dcreatorSet1/"


############################# OBJEKTE #######################################
# Erstellt ein einfaches Objekt und kopiere 4x
xyz = np.array([-312, 128, 4])
size = (36, 37)
s1 = MachineObject(patchName + "tcr3dschraubeKleinGanzLinkeSeite", xyz, size)
print(s1)

############################# COMPOSITION #######################################
fn = [["SBB/11L.png", "SBB/11R.png"], ["SBB/12L.png", "SBB/12R.png"], ["SBB/13L.png", "SBB/13R.png"], ["SBB/14L.png", "SBB/14R.png"], ["SBB/15L.png", "SBB/15R.png"]]
cp = Composition(fn, [s1, s1, s1, s1])
print(cp)
cp.sceneinfo()
#cp.measureObjectInScene(s1, cp._scenes[0])
cp.measureObject(s1)
print(s1.positions)



