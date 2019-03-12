# WTM : Warped Template Matching
""" Die verschiedenen Methoden der Vorverarbeitung der Bilder und Tempates."""

from enum import Enum

class tm(Enum):
    MASK3CH = 1         # Template in einem bildkanal, masken in anderen kanälen. Versuch TM mit transparen
    NOISE = 2           # Template auf Rauschen.
    AVERAGE = 3         # TODO Template auf Hintergrund, der dem mittelwert des Templates entspricht
    CANNY = 4           # Template und Suchbereich durch canny edge detector laufen lassen
    TRANSPARENT = 5     # Verwenden der von opencv unterstützeten Transparenz mit Maske
    CANNYBLUR = 7       # Wie CANNY aber unscharf
    CANNYBLUR2 = 9      # Wie CANNY aber unscharf und INVERS
    NOISEBLUR = 8       # Wie NOISE aber unscharf
    ELSD = 6            # TODO: statt canny die ellipsen und linien erkennen.


class rtref(Enum):
    NONE = 0           # Auf der Scene wurde noch kein Zug Referenz gesetzt
    APPROX = 1       # Eine ungefähre Referenz anhand der Gitterstruktur
    BYOBJECT = 2     # Eine exakte Refernz gesetzt anhand der Gitterschrauben
    BYSFM = 3        # die Referenz wurde anhand Sfm  Bildvergleich bestimmt.