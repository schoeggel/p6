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

    rgb = np.dstack((ch1,ch2,ch3))
    return rgb
