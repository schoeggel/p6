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


def loadconfig(filename, structname='paramStruct'):
    if '.mat' in filename:
        return ___loadconfigmat(filename, structname)
    if '.yaml' in filename or '.yml' in filename:
        return ...  # Removed
    if '.ini' in filename:
        return ...  # Removed


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


# Wie cv putText, aber immer lesbar wegen Umrandung.
def putBetterText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None,
                  colorOutline=None):
    out = img.copy()
    if colorOutline is None:  # ohne explizite Angabe wird die Vordergrundfarbe invertiert für die Umrandungsfarbe
        colorOutline = (255 - color[0], 255 - color[1], 255 - color[2])

    # 12x Hintergrund Text
    displace = [(0, 1), (1, 0), (0, -1), (-1, 0),
                (-2, -2), (-2, 2), (2, -2), (-2, -2),
                (0, 2), (2, 0), (0, -2), (-2, 0)]
    for dx, dy in displace:
        x = max(1, org[0] + dx)
        y = max(1, org[1] + dy)
        out = cv2.putText(out, text, (x, y), fontFace, fontScale, colorOutline, thickness, lineType)

    # Vordergrund Text:
    out = cv2.putText(out, text, org, fontFace, fontScale, color, thickness, lineType)
    return out


def separateRGB(img_in, vertical=False):
    # separiert die 3 Kanäle und legt sie nebeneinander oder übereinander
    r, g, b = img_in[:, :, 0], img_in[:, :, 1], img_in[:, :, 2]
    if vertical:
        res = np.vstack((r, g, b))
    else:
        res = np.hstack((r, g, b))
    return res


def clahe(img, pxPerBlock=50, channel=None):
    # verbessert den kontrast im angegebenen Kanal

    # Kanal aus RGB separieren, falls RGB Bild (Kanalnr als Arg. vorhanden)
    if channel is not None:
        ch = img[:, :, channel]
    else:
        ch = img.copy()

    # grid ist abhängig von der bildgrösse aber immer mind. 2x2
    px = 0.5 * (ch.shape[0] + ch.shape[1])
    grid = round(px / pxPerBlock)
    if grid < 2: grid = 2

    # Kontrast verbessern
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(grid, grid))
    ch = clahe.apply(ch)

    # Verbesserter Kanal wieder in RGB-Bild einfügen, falls RGB
    if img.ndim == 3:
        res = img.copy()
        res[:, :, channel] = ch
    else:
        res = ch

    return res
