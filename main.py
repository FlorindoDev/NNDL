from Model.manual.NeuralNetwork import NeuralNetwork as nn
from common.activation import ReLU, Softmax, LeakyReLU,Sigmoid
from common.weight_Init import Glorot, He,Contadina
from common.loss import CrossEntropy
from common.update_rule import RProp, StandardUpdateRule,Adam
import numpy as np
from tensorflow.keras.datasets import mnist
from common.logger import Logger
from common.dataset_preprocessor import DatasetPreprocessor


logger = Logger()
dataset_preprocessor = DatasetPreprocessor()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train, x_val, y_val, x_test, y_test = dataset_preprocessor.prepare_from_splits(x_train, y_train,x_test, y_test,0.30)

rprop = RProp()
standard = StandardUpdateRule()
adam = Adam()



rete = nn(layer_sizes=[784,512,256,128,64,10],update_rule=adam,activation=LeakyReLU,weight_init=He) # creo la rete


rete.train(x_train,y_train,50,32,early_stopping=True,X_validation=x_val,y_validation=y_val,patience=5) # alleno la rete

# print(y_test[0:20])
# logger.print_matrix(rete.forward(input_pre),"predizioni",True)
rete.evaluate(x_test,y_test,True)
