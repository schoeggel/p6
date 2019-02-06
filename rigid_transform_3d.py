# Quelle: http://nghiaho.com/uploads/code/rigid_transform_3D.py_
# Doc: http://nghiaho.com/?page_id=671
# Author: Nghia Ho
# Angepasst und erweitert: J. Koch


from numpy import *
from math import sqrt


# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert (len(A) == len(B))

    #beide in np.array umwandeln
    A = array(A)
    B = array(B)

    N = A.shape[0];  # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    # Korrektur: veraltet, matmul verwenden
    # H = transpose(AA) * BB
    # Seit python 3.6 : @ bei np.arrays für MAtrizenmultiplikation
    H = transpose(AA) @ BB


    U, S, Vt = linalg.svd(H)

    # Korrektur: veraltet, matmul verwenden
    #R = Vt.T * U.T
    R = Vt.T @ U.T

    # special reflection case
    if linalg.det(R) < 0:
        #assert 1==0  # TODO : entfernen nach allen Tests
        print("Reflection detected")
        # Vt[2, :] *= -1
        # R = Vt.T * U.T
        # Bessere Lösung nach Nick Lambert: multiply 3rd column of R by -1
        R[2, :] *= -1

    t = -R @ centroid_A.T + centroid_B.T

    # Manchmal kommt ein 3x3 Translationsvektor als Ergebnis.
    assert (t.shape == (3,))
    t = reshape(t,(3,1))
    print (t)

    return R, t


def rmserror(A, B):
    """
    Berechnet den Abweichungsfehler
    :param A, B: Eingangs-Array. Müssen gleiche Form haben
    :return: Fehler
    """

    assert (A.shape == B.shape)
    delta = A - B
    delta = multiply(delta, delta)
    delta = sum(delta)
    rmse = sqrt(delta / A.shape[0])
    return rmse


if __name__ == '__main__':
# Test with random data

# Random rotation and translation
    R = mat(random.rand(3, 3))
    t = mat(random.rand(3, 1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = linalg.svd(R)
    R = U @ Vt

    # remove reflection
    if linalg.det(R) < 0:
        R[2, :] *= -1

    # number of points
    n = 10

    A = mat(random.rand(n, 3));
    print("debug tile(t, ((1,n)):\n", tile(t, (1, n)))
    B = R @ A.T + tile(t, (1, n))
    B = B.T;

    # recover the transformation
    ret_t = zeros(3)
    ret_R = diag([0,0,0])
    ret_R, ret_t = rigid_transform_3D(A, B)


    A2 = (ret_R @ A.T) + tile(ret_t, (1, n))     # die translation mithilfe von "tile" auf alle punkte erweitern.
    A2 = A2.T

    # Find the error
    err = rmserror(A2, B)


    print("Points A\n", A)
    print("Points B\n", B)
    print("Rotation\n", R)
    print("Translation\n", t)
    print("RMSE\n", err)
    print("If RMSE is near zero, the function is correct!")
