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
fn1 = "SBB/13L.png"
fn2 = "SBB/13R.png"
test = Composition([[fn1, fn2], [fn1, fn2]], [s1, s1, s1, s1])
print(test)

