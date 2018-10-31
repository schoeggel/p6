import numpy
import cvaux as util


class CalibData:
    def __init__(self, cfile="cfg/cameracalib.mat", ffile="cfg/F.mat"):
        self.files = [cfile, ffile]
        self.__cal = util.loadconfig(cfile)
        self.f = util.loadconfig(ffile, "F")
        self.rl = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.tl = numpy.array([0, 0, 0])
        self.rr = self.__cal.RotationOfCamera2
        self.tr = self.__cal.TranslationOfCamera2
        self.kl = self.__cal.CameraParameters1.IntrinsicMatrix.transpose()
        self.kr = self.__cal.CameraParameters2.IntrinsicMatrix.transpose()
        self.pl = numpy.dot(self.kl, numpy.column_stack((self.rl, self.tr)))
        self.pr = numpy.dot(self.kr, numpy.column_stack((self.rr, self.tr)))
