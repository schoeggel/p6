# Quelle: http://nghiaho.com/uploads/code/rigid_transform_3D.py_
# Doc: http://nghiaho.com/?page_id=671
# Author: Nghia Ho
# Anmerkung: Ja nachdem, ob die Punkte als Matrix oder ndarray
# übergeben, kommt es zu unterschieden.
# Die TEST-Variante beinhaltete fixierte Punkte aus einem Beispiel mit dem Zug
# wichtig: beim umwandeln in array nicht float 32 verwenden, dann wirds ungenau, falls
# die einheitsvekoren im unit [mm] sind



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
    H = transpose(AA) @ BB


    U, S, Vt = linalg.svd(H)

    # Korrektur: veraltet, matmul verwenden
    #R = Vt.T * U.T
    R = Vt.T @ U.T

    # special reflection case
    if linalg.det(R) < 0:
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
    n = 4

    A = mat(random.rand(n, 3));
    A = array([[-3.1058179e+02, -1.5248854e+02, 7.8082729e+03],
                      [ 1.3992499e+02, -2.6128412e+02, 7.9217915e+03],
                      [ 2.9599524e+02,  5.3110981e+00, 7.5579175e+03],
                      [-1.5471242e+02,  1.1419590e+02, 7.4451899e+03]])

    print("debug tile(t, ((1,n)):\n", tile(t, (1, n)))
    B = R @ A.T + tile(t, (1, n))
    B = B.T;

    # recover the transformation
    ret_R, ret_t = rigid_transform_3D(A, B)

    A2 = (ret_R @ A.T) + tile(ret_t, (1, n))
    A2 = A2.T

    # Find the error
    err = A2 - B

    err = multiply(err, err)
    err = sum(err)
    rmse = sqrt(err / n);

    print("Points A\n", A)
    print("Points B\n", B)
    print("Rotation\n", R)
    print("Translation\n", t)
    print("RMSE\n", rmse)
    print("If RMSE is near zero, the function is correct!")
