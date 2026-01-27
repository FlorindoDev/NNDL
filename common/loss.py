import numpy as np
from .logger import Logger

logger = Logger()

def CrossEntropy(y, t, eps=1e-12):
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    y = np.clip(y, eps, 1.0)              # evita log(0)
    loss_per_col = -np.sum(t * np.log(y), axis=0)  # CE per sample (colonna)
    return np.sum(loss_per_col)
    
    

