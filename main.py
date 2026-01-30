from Model.manual.NeuralNetwork import NeuralNetwork as nn
from common.activation import ReLU, Softmax, LeakyReLU
from common.weight_Init import Glorot, He,Contadina
from common.loss import CrossEntropy
from common.update_rule import RProp, StandardUpdateRule,Adam
import numpy as np
from tensorflow.keras.datasets import mnist
from common.logger import Logger


logger = Logger()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test  = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

split = int(0.6 * len(x_train)) # calcolo percentuale , ad es. 60% test e 40% validation

X_tr  = x_train[:split] 
y_tr  = y_train[:split]

X_val = x_train[split:]
y_val = y_train[split:]



rprop = RProp()
standard = StandardUpdateRule()
adam = Adam()



rete = nn(layer_sizes=[784,128,64,10],update_rule=adam,activation=LeakyReLU,weight_init=He) # creo la rete
input=np.atleast_2d(X_tr) # prendo input che è una matrice
label=np.atleast_1d(y_tr) # label di ogni input (è un vettore per ora)
input_val=np.atleast_2d(X_val) # prendo input che è una matrice
label_val=np.atleast_1d(y_val) # label di ogni input (è un vettore per ora)

rete.train(input,label,100,32,early_stopping=True,X_validation=input_val,y_validation=label_val,pacience=15) # alleno la rete

input_pre = np.atleast_2d(x_test)
output_pre = np.atleast_1d(y_test)
# print(y_test[0:20])
# logger.print_matrix(rete.forward(input_pre),"predizioni",True)
rete.evaluate(input_pre,output_pre,True)


rete2 = nn(layer_sizes=[784,128,64,10],update_rule=adam,activation=LeakyReLU,weight_init=He) # creo la rete
input=np.atleast_2d(X_tr) # prendo input che è una matrice
label=np.atleast_1d(y_tr) # label di ogni input (è un vettore per ora)
input=np.atleast_2d(X_val) # prendo input che è una matrice
label=np.atleast_1d(y_val) # label di ogni input (è un vettore per ora)
rete2.train(input,label,50,32) # alleno la rete

input_pre = np.atleast_2d(x_test)
output_pre = np.atleast_1d(y_test)
# print(y_test[0:20])
# logger.print_matrix(rete.forward(input_pre),"predizioni",True)
rete2.evaluate(input_pre,output_pre)