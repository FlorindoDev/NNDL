from Model.NeuralNetwork import NeuralNetwork as nn
from common.Activation import ReLU, softmax
from common.Weight_Init import Glorot, He
from common.Loss import CrossEntropy
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

data_set =  mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], -1)
x_test  = x_test.reshape(x_test.shape[0], -1)

print(x_train)  # (60000, 784)
#print(y_train[0:500])  # (60000, 784)

#y = np.atleast_2d([[0.4,0.7],
#                   [0.6,0.3]]
#                )
#t =  np.atleast_2d([[0,1],[1,0]])
#CrossEntropy(y,t)


rete = nn([784,784,10],ReLU,softmax,0.01,He)

#rete.forward(np.atleast_2d([[3,4,5,6],[7,3,2,1]]))
rete.train(np.atleast_2d(x_train[0:500]),y_train[0:500],1,32)