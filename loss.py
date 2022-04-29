import numpy as np


class LogisticLoss:
    @staticmethod
    def val(y, y_hat):
        # Numerically stable version, seperating the negative and positive to avoid overflow.
        t = y * y_hat
        idx = t > 0
        out = np.zeros(idx.shape)
        out[idx] = np.log(1 + np.exp(-t[idx]))
        out[~idx] = -t[~idx] + np.log(1+ np.exp(t[~idx]))   #exp_t / (1.0 + exp_t)
        return out 
    # @staticmethod
    # def val(y, y_hat): # Numerically unstable version
    #     return np.log(1 + np.exp(-y * y_hat))

    @staticmethod
    def prime(y, y_hat):
        return -y / (1 + np.exp(y * y_hat))

    @staticmethod
    def dprime(y, y_hat):
        a = np.exp(y * y_hat)
        return 1/(1+a) - 1/((1+a)**2) # Numerically stable version
        # return a / ((1 + a) ** 2)  # Numerically unstable version


class L2:
    @staticmethod
    def val(y, y_hat):
        return (y - y_hat)**2 / 2.

    @staticmethod
    def prime(y, y_hat):
        return y_hat - y

    @staticmethod
    def dprime(y, y_hat):
        return np.ones_like(y_hat)


class PseudoHuberLoss:

    def __init__(self, delta=1.0):
        self.delta = delta

    def val(self, y, y_hat):
        diff = y_hat - y
        return (self.delta ** 2) * (np.sqrt(1. + (diff / self.delta) ** 2) - 1.)

    def prime(self, y, y_hat):
        diff = y_hat - y
        return diff / np.sqrt(1. + (diff / self.delta) ** 2)

    def dprime(self, y, y_hat):
        diff = y_hat - y
        return np.power((1. + (diff / self.delta) ** 2), -1.5)
