import numpy as np
import cv2
import calibMatrix

# quelle: https://pythonpath.wordpress.com/2012/08/29/cv2-triangulatepoints/
# Die eigene Triangulation ("14-...") funktioniert nicht, Fehlersuche erfolglos.
# Rückrechnung eigene Triangulation gemäss der Quelle versucht, Ergebnisse falsch
# Vorgehen: Original Code der Quelle  eigene Daten einfüllen,
# und überprüfen, ob das Ergebnis der Rückrechnung noch stimmt.


# Änderungen:
#  - Syntaxanpassungen für python 3
#  - echte Kamercalibrierungsdaten laden
#  - Neue Bildpunkte von Jans Set gmessen: 123 L+R



# Camera projection matrices
P1 = np.eye(4)
P2 = np.array([[0.878, -0.01, 0.479, -1.995],
               [0.01, 1., 0.002, -0.226],
               [-0.479, 0.002, 0.878, 0.615],
               [0., 0., 0., 1.]])


cob = calibMatrix.CalibData()
P1 = cob.pl
P2 = cob.pr


# Homogeneous arrays
a3xN = np.array([[0.1110, 0.2376, 0.2850, 0.1529, 0.1110],
                 [0.1111, 0.0814, 0.1557, 0.1881, 0.1111],
                 [1., 1., 1., 1., 1.]])

b3xN = np.array([[0.1715, 0.3010, 0.2712, 0.1357, 0.1715],
                [0.0820, 0.1004, 0.1795, 0.1591, 0.0820],
                [1., 1., 1., 1., 1.]])





# alternatives bildmaterial von Jan steger: 123 L + R
# min max: -80
a3xN = np.float64([[1686, 3018, 3572, 2182, 1686],
                   [2085, 1752, 2581, 2945, 2085],
                   [1., 1., 1., 1., 1.]])

b3xN = np.float64([[1260, 2636, 2370, 872, 1260],
                   [1805, 2010, 2906, 2678, 1805],
                   [1., 1., 1., 1., 1.]])


# alternatives bildmaterial von Jan steger: 123 L + R
# min max : -2517
a3xN = np.float64([[1686, 3018, 3572, 2182, 1686],
                   [3000-2085, 3000-1752, 3000-2581, 3000-2945, 3000-2085],
                   [1., 1., 1., 1., 1.]])

b3xN = np.float64([[1260, 2636, 2370, 872, 1260],
                   [3000-1805, 3000-2010, 3000-2906, 3000-2678, 3000-1805],
                   [1., 1., 1., 1., 1.]])


# alternatives bildmaterial von Jan steger: 123 L + R
# min max: komplett daneben > 300'000%
a3xN = np.float32([[1686/4096, 3018/4096, 3572/4096, 2182/4096, 1686/4096],
                   [2085/3000, 1752/3000, 2581/3000, 2945/3000, 2085/3000],
                   [1., 1., 1., 1., 1.]])

b3xN = np.float32([[1260/4096, 2636/4096, 2370/4096, 872/4096, 1260/4096],
                   [1805/3000, 2010/3000, 2906/3000, 2678/3000, 1805/3000],
                   [1., 1., 1., 1., 1.]])




# WESHALB GIBT ES ZWISCHEN DEN NÄCHSTEN ZWEI VARIANTEN EINE UNTERSHCIEDLICHE ABWEICHUNG ???
# alternatives bildmaterial von Jan steger: 123 L + R
# min max: -99 +120
a3xN = np.float32([[1686, 3018, 3572, 2182, 1686],
                   [2085, 1752, 2581, 2945, 2085]]).reshape(-1,1,2)

b3xN = np.float32([[1260, 2636, 2370, 872, 1260],
                   [1805, 2010, 2906, 2678, 1805]]).reshape(-1,1,2)


# alternatives bildmaterial von Jan steger: 123 L + R
# min max: -80 +60
#a3xN = np.float32([[1686, 3018, 3572, 2182, 1686],
#                   [2085, 1752, 2581, 2945, 2085],
#                   [1., 1., 1., 1., 1.]])

#b3xN = np.float32([[1260, 2636, 2370, 872, 1260],
#                   [1805, 2010, 2906, 2678, 1805],
#                  [1., 1., 1., 1., 1.]])


# VERSUCH, die punkte zu entzerren --> die änderungen sind so klein, es ändert sich nichts am ergebnis.
a3xN = cv2.undistortPoints(a3xN, cob.kl, cob.drl)
b3xN = cv2.undistortPoints(b3xN, cob.kr, cob.drr)
print('a3xN\n', a3xN)
print('b3xN\n', b3xN)



# # The cv2 method
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
    print('MAX:', np.max(np.divide(x1-a3xN, a3xN)*100), '% Delta')
    print('MIN:', np.min(np.divide(x1-a3xN, a3xN)*100), '% Delta')
print('\nP1\n', P1)
print('\nP2\n', P2)

