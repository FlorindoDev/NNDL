import numpy as np
from .logger import Logger

logger = Logger()

def CrossEntropy(y, t):
    """
    y : array (N, C)  -> output della softmax per colonna
    t : array (N, C)  -> target one-hot per riga
    """
    
    loss = np.sum(-np.sum(t * np.log(y), axis=0)) # somma del vettore di loss
    #logger.print(loss,"LOSS")
    return loss
    
    

