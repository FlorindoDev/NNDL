import numpy as np
from .Logger import Logger

logger = Logger()

def ReLU(a):
    return np.maximum(a, 0)


def Derivata_ReLU(a):
    return np.where(a > 0, 1, 0)


def LReLU(a):
    alfa = 0.01 # 0 < alfa < 1
    return np.maximum(a, alfa * a)

def PReLU(a):
    pass

def ELU(a):
    pass


def softmax(a):
    a = np.asarray(a)
    exp_a = np.exp(a)
    probs = exp_a / np.sum(exp_a, axis=0, keepdims=True)
    
    #col_sums = np.sum(probs, axis=0)
    #logger.print(col_sums, "softmax col sums:")

    return probs
    
