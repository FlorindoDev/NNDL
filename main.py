from Model.NeuralNetwork import NeuralNetwork as nn
from common.activation import ReLU, Softmax, LeakyReLU
from common.weight_Init import Glorot, He
from common.loss import CrossEntropy
from common.update_rule import RProp, StandardUpdateRule,Adam
import numpy as np
from tensorflow.keras.datasets import mnist
from common.logger import Logger

logger = Logger()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test  = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0


rprop = RProp()
standard = StandardUpdateRule()
adam = Adam()



rete = nn(layer_sizes=[784,200,10],update_rule=rprop,activation=LeakyReLU,weight_init=Glorot) # creo la rete
input=np.atleast_2d(x_train) # prendo input che è una matrice
label=np.atleast_1d(y_train) # label di ogni input (è un vettore per ora)
rete.train(input,label,200,input.shape[0]) # alleno la rete

input_pre = np.atleast_2d(x_test)
output_pre = np.atleast_1d(y_test)
# print(y_test[0:20])
# logger.print_matrix(rete.forward(input_pre),"predizioni",True)
rete.evaluate(input_pre,output_pre)