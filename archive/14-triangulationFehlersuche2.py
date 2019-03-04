import numpy as np
import cv2

# quelle: https://pythonpath.wordpress.com/2012/08/29/cv2-triangulatepoints/
# Die eigene Triangulation ("14-...") funktioniert nicht, Fehlersuche erfolglos.
# Rückrechnung eigene Triangulation gemäss der Quelle versucht, Ergebnisse falsch
# Vorgehen: Original Code der Quelle Schrittweise eigene Daten einfüllen,
# und laufend überprüfen, ob das Ergebnis der Rückrechnung noch stimmt.

# Änderungen:
#  - Syntaxanpassungen für python 3
#  - Bildpunkte (pixelkoordinaten) der Schrauben eingesetzt



# Camera projection matrices
P1 = np.eye(4)
P2 = np.array([[0.878, -0.01, 0.479, -1.995],
               [0.01, 1., 0.002, -0.226],
               [-0.479, 0.002, 0.878, 0.615],
               [0., 0., 0., 1.]])
# Homogeneous arrays
a3xN = np.array([[1110, 2376, 2850, 1529, 1110],
                 [1111, 814, 1557, 1881, 1111],
                 [1., 1., 1., 1., 1.]])

b3xN = np.array([[1715, 3010, 2712, 1357, 1715],
                 [820, 1004, 1795, 1591, 820],
                 [1., 1., 1., 1., 1.]])

# The cv2 method
X = cv2.triangulatePoints(P1[:3], P2[:3], a3xN[:2], b3xN[:2])
# Remember to divide out the 4th row. Make it homogeneous
X /= X[3]
# Recover the origin arrays from PX
x1 = np.dot(P1[:3], X)
x2 = np.dot(P2[:3], X)
# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]

print('X\n', X)
print('x1\n', x1)
print('x2\n', x2)

with np.printoptions(precision=2, suppress=True):
    print('(x1-origin)/origin\n', np.divide(x1-a3xN, a3xN)*100)
    print('MAX\n', np.max(np.divide(x1-a3xN, a3xN)*100), '% Delta')
