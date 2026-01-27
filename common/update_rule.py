import numpy as np

def standard_update_weight(weight,dW,biase,db,learning_rate):
    dW = np.atleast_2d(dW)
    weight = weight - (dW * learning_rate)
    db = np.atleast_2d(db)
    biase = biase - (db * learning_rate)
    return weight, biase


def rprop(weight,dW,biase,db):


    dW = np.atleast_2d(dW)
    weight = weight - np.sign(dW) 
    pass

def adam():
    pass

