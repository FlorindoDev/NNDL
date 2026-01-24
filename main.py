from Model.NeuralNetwork import NeuralNetwork as nn
from Function.Activation import ReLU, softmax
from Function.Weight_Init import Glorot, He

import numpy as np

rete = nn([4,3,5],ReLU,softmax,0.01,He)

rete.forward(np.atleast_2d([[3,4,5,6],[7,3,2,1]]))