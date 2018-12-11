# Kann ein MAtlab Struct laden mit den Cameradaten.
# MAtlab speichert die Daten jeweils transponiert.

import numpy
import cvaux


class CalibData:
    def __init__(self, cfile="cfg/cameracalib.mat", ffile="cfg/F.mat"):
        self.files = [cfile, ffile]
        self.__cal = cvaux.loadconfig(cfile)
        self.f = cvaux.loadconfig(ffile, "F")
        self.rl = numpy.eye(3)
        self.tl = numpy.zeros(3)
        self.rr = self.__cal.RotationOfCamera2.transpose()
        self.tr = self.__cal.TranslationOfCamera2.transpose()
        self.kl = self.__cal.CameraParameters1.IntrinsicMatrix.transpose()
        self.kr = self.__cal.CameraParameters2.IntrinsicMatrix.transpose()
        self.drl = self.__cal.CameraParameters1.RadialDistortion   # distortion radial left
        self.drr = self.__cal.CameraParameters2.RadialDistortion   # distortion radial left

        # this is just a 5 × 1 matrix containing k1, k2, p1, p2, and k3( in thatorder):
        self.drl = numpy.append(self.drl, [0, 0, 0])
        self.drr = numpy.append(self.drr, [0, 0, 0])

        # Projektionsmatrizen zusammensetzen/berechnen nach Zissermann 6.8
        # P = K*[T|t]
        self.pl = numpy.matmul(self.kl, numpy.column_stack((self.rl, self.tl.T)))
        self.pr = numpy.matmul(self.kr, numpy.column_stack((self.rr, self.tr.T)))


if __name__ == '__main__':
    c = CalibData()
    print(c.rl)
    print(c.tl)
    print(c.rr)
    print(c.tr)
    print(c.pl)
    print(c.pr)
    print(c.drl)
    print(c.drr)
    print(c.drl)
    print(c.drr)
    print(c.kl)
    print(c.kr)

