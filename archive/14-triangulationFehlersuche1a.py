import numpy as np
import cv2

# quelle: https://pythonpath.wordpress.com/2012/08/29/cv2-triangulatepoints/
# Die eigene Triangulation ("14-...") funktioniert nicht, Fehlersuche erfolglos.
# Rückrechnung eigene Triangulation gemäss der Quelle versucht, Ergebnisse falsch
# Vorgehen: Original Code der Quelle Schrittweise eigene Daten einfüllen,
# und laufend überprüfen, ob das Ergebnis der Rückrechnung noch stimmt.

#Max Delta = 0.12%

# Änderungen:
#  - Syntaxanpassungen für python 3
#  - kleine Were ändern

# Camera projection matrices
P1 = np.eye(4)
P2 = np.array([[0.878, -0.01, 0.479, -1.995],
               [0.01, 1., 0.002, -0.226],
               [-0.479, 0.002, 0.878, 0.615],
               [0., 0., 0., 1.]])
# Homogeneous arrays
a3xN = np.float32([[0.091, 0.167, 0.231, 0.083, 0.154],
                 [0.364, 0.333, 0.308, 0.333, 0.308],
                 [0.001, 0.001, 0.001, 0.001, 0.001]])/0.001
b3xN = np.float32([[0.42, 0.537, 0.645, 0.431, 0.538],
                 [0.389, 0.375, 0.362, 0.357, 0.345],
                 [0.001, 0.001, 0.001, 0.001, 0.001]])/0.001
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
