# Verschiedene Hilfsfunktionen
import scipy.io.matlab
import numpy as np



# Lädt ein .mat File und liefert das angegebene Struct zurück.
# Es kann auf dem Returnobjekt direkt mit den Strukturnamen gearbeitet werden:
# obj.CameraData1.ImageSize  etc...
# siehe https://docs.scipy.org/doc/scipy/reference/tutorial/io.html
def ___loadconfigmat(filename, structname='paramStruct'):
    cfg = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return cfg[structname]


# Lädt ein YAML. Auf dem Return Objekt muss etwas umständlicher auf die Daten zugegriffen werden:
# obj['CameraData1'
def ___loadconfigyaml(filename):
    with open(filename, 'r') as ymlfile:
        yaml = YAML(typ='safe')  # default, if not specfied, is 'rt' (round-trip)
        yaml.load(filename)
        return pyaml.load(ymlfile)


# TODO: Fertigstellen !
# Lädt ein ini. Auf dem Return Objekt muss etwas umständlicher auf die Daten zugegriffen werden:
# obj['CameraData1']
# NOCH NICHT GETESTET / FERTIG !!!
def ___loadconfigini(filename):
    cfg = configparser.ConfigParser()
    cfg.read(filename)
    cfg.sections()
    return cfg



def loadconfig(filename, structname='paramStruct'):
    if '.mat' in filename:
        return ___loadconfigmat(filename, structname)
    if '.yaml' in filename or '.yml' in filename:
        return ___loadconfigyaml(filename)
    if '.ini' in filename:
        return ___loadconfigini(filename)




def mixtochannel(im1, im2=None, im3=None):
    # mischt im1 bis im3 in RGB Kanal

    # Kanäle extrahieren
    if im1.ndim < 3:
        ch1 = im1
    else:
        ch1 = im1[:, :, 0]

    if im2 is None:
        ch2 = np.empty(ch1.shape)
    elif im2.ndim < 3:
        ch2 = im2
    else:
        ch2 = im2[:, :, 0]

    if im3 is None:
        ch3 = np.empty(ch1.shape)
    elif im3.ndim < 3:
        ch3 = im3
    else:
        ch3 = im3[:, :, 0]

    rgb = np.dstack((ch1, ch2, ch3))
    return rgb



