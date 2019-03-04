# Verschiedene Hilfsfunktionen
import scipy.io.matlab
import numpy as np
import cv2



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


def imgMergerH(list_of_img_in: list, bgcolor=(128, 128, 128)):
    # Abmessungen des Bilds mit der grössten vertikalen Abmessung finden)
    resolutions = [x.shape[0] for x in list_of_img_in]
    maxv = max(resolutions)
    list_of_img = []

    for img in list_of_img_in:
        # alles auf RGB ändern
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Alle Bilder auf diese Abmessung erweitern und in neue Liste speichern
        pixel = np.array(bgcolor, dtype=np.uint8)
        gap = maxv - img.shape[0]
        gap_upper = gap // 2
        gap_lower = gap - gap_upper
        filler_upper = np.tile(pixel, (gap_upper, img.shape[1], 1))
        filler_lower = np.tile(pixel, (gap_lower, img.shape[1], 1))
        img = np.vstack((filler_upper, img, filler_lower))
        list_of_img.append(img)

    return np.hstack(list_of_img)


def imgMergerV(list_of_img_in: list, bgcolor=(128, 128, 128)):
    # Abmessungen des Bilds mit der grössten horizontalen Abmessung finden)
    resolutions = [x.shape[1] for x in list_of_img_in]
    maxh = max(resolutions)
    list_of_img = []

    for img in list_of_img_in:
        # alles auf RGB ändern
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Alle Bilder auf diese Abmessung erweitern und in neue Liste speichern
        pixel = np.array(bgcolor, dtype=np.uint8)
        gap = maxh - img.shape[1]
        gap_upper = gap // 2
        gap_lower = gap - gap_upper
        filler_upper = np.tile(pixel, (img.shape[0], gap_upper, 1))
        filler_lower = np.tile(pixel, (img.shape[0], gap_lower, 1))
        img = np.hstack((filler_upper, img, filler_lower))
        list_of_img.append(img)

    return np.vstack(list_of_img)