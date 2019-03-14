# WTM : Warped Template Matching
"""Erstellt die zu lokalisierenden Objekte und gibt sie in zwei separaten Listen.

    Die erste Liste enthält die vier Referenzobjekte 'Gitterschraube'
    Die zweite Listen enthält alle anderen Objekte"""

import numpy as np
import wtmObject

def createMachineObjects(version = 2):
    if version == 1: return createMachineObjects1()
    elif version == 2: return createMachineObjects2()
    elif version == 3: return  ...

def createMachineObjects2():
    """ Erstellung der Objekte via Metafile.

        Sucht im angegebenen Verzeichnis nach allen .txt Dateien.
        Die Text Datei muss die Daten zum Objekt enthalten. Ausgehend vom
        Dateinamen werden noch die Bilddateien gesucht und geladen.
        Für ein Objekt müssen genau drei Dateien vorliegen:
        *__.txt (Metadaten), *_L.png (Template linke Kamera), *_R.png (Template rechte Kamera)
    """

    manyMachineObjects = processFolder("data/patches/3dcreatorSet1/")
    refObjects = processFolder("data/patches/3dcreatorSetRef1/")
    return refObjects, manyMachineObjects


def processFolder(folder):
    from pathlib import Path
    txtfiles = list(Path(folder).glob('**/*.txt'))
    lst = []
    for metafile in txtfiles:
        stem = metafile.stem[:-2]
        name, xyz, rot, size = parser(metafile)
        lst.append(wtmObject.MachineObject(folder + stem, xyz, size, rotation3d=rot, name=name))
    return lst


def parser(filename):
    f = open(filename, 'r')
    name = f.readline()[7:-1]    # ohne \n
    posx = int(f.readline()[7:])
    posy = int(f.readline()[7:])
    posz = int(f.readline()[7:])
    rotx = float(f.readline()[7:])
    roty = float(f.readline()[7:])
    rotz = float(f.readline()[7:])
    sizex = int(f.readline()[7:])
    sizey = int(f.readline()[7:])
    f.close()
    xyz = np.array([posx, posy, posz])
    rot = np.array([rotx, roty, rotz])
    return name, xyz , rot, (sizex, sizey)


def createMachineObjects1():
    """ Erstellung der Objekte ist hardcoded."""
    folder = "data/patches/3dcreatorSet1/"
    refObjects = []
    objects = []

    ################################# GITTERSCHRAUBEN (REFERENZ OBJEKTE) ####################################
    pass
    #################################          RESTLICHE OBJEKTE         ####################################
    xyz, size = np.array([-312, 128, 4]), (36, 37)
    objects.append(wtmObject.MachineObject(folder + "tcr3dschraubeKleinGanzLinkeSeite", xyz, size))
    return refObjects, object


if __name__ == '__main__':
    ref, objs = createMachineObjects2()
    print(ref)
    print("-"*30)
    print(objs)