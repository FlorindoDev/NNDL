from Model.NeuralNetwork import NeuralNetwork as nn
from common.activation import ReLU, Softmax
from common.weight_Init import Glorot, He
from common.loss import CrossEntropy
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test  = x_test.reshape(x_test.shape[0], -1)

rete = nn([784,10]) # creo la rete
input=np.atleast_2d(x_train[0:1]) # prendo input che è una matrice
label=np.atleast_1d(y_train[0:1]) # label di ogni input (è un vettore per ora)
rete.train(input,label,1,1)
