import numpy as np
from .logger import Logger

logger = Logger()

class Activation():
    def __init__(self, activation, derivate):
        self.activation = activation
        self.derivate = derivate

########################################
#       Funzioni attivazione           #
########################################

def fun_ReLU(a):
    return np.maximum(a, 0)


def derivate_ReLU(a):
    return np.where(a > 0, 1, 0)


def LReLU(a):
    alfa = 0.2 # 0 < alfa < 1
    return np.maximum(a, alfa * a)


def derivate_LReLU(a, alfa=0.2):
    return np.where(a > 0, 1, alfa)


def fun_softmax(a):
    a = np.asarray(a, dtype=np.float64)

    # softmax per colonne (axis=0): sottrai il max di ogni colonna
    a = a - np.max(a, axis=0, keepdims=True)

    exp_a = np.exp(a)
    denom = np.sum(exp_a, axis=0, keepdims=True)

    # opzionale: protezione extra (in teoria non serve dopo max-shift)
    denom = np.maximum(denom, 1e-12)

    return exp_a / denom
    

def delta_softmax(y,t):
    probs = y-t
    return probs

########################################
#               Oggetti                #
########################################


ReLU = Activation(fun_ReLU,derivate_ReLU)

Softmax = Activation(fun_softmax,delta_softmax)

LeakyReLU = Activation(LReLU,derivate_LReLU)
