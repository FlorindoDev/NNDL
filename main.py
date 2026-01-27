from Model.NeuralNetwork import NeuralNetwork as nn
from common.activation import ReLU, Softmax
from common.weight_Init import Glorot, He
from common.loss import CrossEntropy
import numpy as np
from tensorflow.keras.datasets import mnist
from common.logger import Logger

logger = Logger()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test  = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

rete = nn([784,100,10]) # creo la rete
input=np.atleast_2d(x_train) # prendo input che è una matrice
label=np.atleast_1d(y_train) # label di ogni input (è un vettore per ora)
rete.train(input,label,30,32)

input_pre = np.atleast_2d(x_test)
output_pre = np.atleast_1d(y_test)
# print(y_test[0:20])
# logger.print_matrix(rete.forward(input_pre),"predizioni",True)
rete.evaluate(input_pre,output_pre)