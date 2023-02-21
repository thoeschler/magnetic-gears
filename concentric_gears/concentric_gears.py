import numpy as np

class Modulator:
    def __init__(self, n, Rm, d, w, alpha, conductivity):
        self._n = n
        self._Rm = Rm
        self._d = d
        self._w = w
        self._alpha = alpha
        self._angle = 0.
        self._sigma = conductivity

    @property
    def n(self):
        return self._n

    @property
    def Rm(self):
        return self._Rm
    
    @property
    def d(self):
        return self._d
    
    @property
    def w(self):
        return self._w
    
    @property
    def alpha(self):
        return self._alpha


class ConcentricGearsProblem:
    def __init__(self, first_gear, second_gear, modulator):
        assert np.allclose(first_gear.x_M, second_gear.x_M)