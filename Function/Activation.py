import numpy as np

def ReLU(a):
    return np.maximum(a, 0)

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
    
    col_sums = np.sum(probs, axis=0)
    print(f"softmax col sums: {col_sums}")

    return probs
    
