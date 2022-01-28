import os, random, sys
import numpy as np
from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import DebyePDFCalculator

sys.path.append(os.getcwd())
random.seed(14)  # 'Random' numbers


class simPDFs_xyz:
    def __init__(self):

        # print("{} has been created.".format(self.csvname))

        # Parameters
        self._starting_parameters_xyz()  # Initiates starting parameters
        self.sim_para = ['xyz', 'Biso', 'rmin', 'rmax', 'rstep',
                         'qmin', 'qmax', 'qdamp', 'delta2']

        r = np.arange(self.rmin, self.rmax, self.rstep)  # Used to create header

        #self.genPDFs_xyz(path+'/'+file)  # Without dummy features


    def _starting_parameters_xyz(self):
        self.qmin = 1
        self.qmax = 30  # Instrument resolution
        self.qdamp = 0.04  # Instrumental dampening
        self.rmin = 0  # Smallest r value
        self.rmax = 30.1  # Can not be less then 10 AA
        self.rstep = 0.1  # Nyquist for qmax = 30
        self.Biso = 1  # Atomic vibration
        self.delta2 = 2  # Corelated vibration

        return None

    def genPDFs_xyz(self, clusterFile):
        stru = loadStructure(clusterFile)

        stru.B11 = self.Biso
        stru.B22 = self.Biso
        stru.B33 = self.Biso
        stru.B12 = 0
        stru.B13 = 0
        stru.B23 = 0

        PDFcalc = DebyePDFCalculator(rmin=self.rmin, rmax=self.rmax, rstep=self.rstep,
                                qmin=self.qmin, qmax=self.qmax, qdamp=self.qdamp, delta2=self.delta2)
        r0, g0 = PDFcalc(stru)
        self.r = r0
        self.Gr = g0

        return None

    def parameterMixer_xyz(self):
        # Add some random factor to the simulation parameters

        self.qmin = random.uniform(0.0, 2)
        self.qmax = random.uniform(14, 28)
        self.qdamp = random.uniform(0.01, 0.1)
        self.Biso = random.uniform(0.0001, 0.5)
        #self.delta2 = random.uniform(0.1, 8)

        return None

    def set_parameters_xyz(self, rmin, rmax, rstep, Qmin, Qmax, Qdamp, Biso, delta2):
        # Add some random factor to the simulation parameters

        self.qmin = Qmin
        self.qmax = Qmax
        self.qdamp = Qdamp
        self.rmin = rmin
        self.rmax = rmax
        self.rstep = rstep
        self.Biso = Biso
        self.delta2 = delta2

        return None

    def getPDF_xyz(self):
        return self.r, self.Gr
