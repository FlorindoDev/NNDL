import numpy as np
from common.activation import ReLU, Softmax
from common.weight_Init import He
from common.logger import Logger
from common.loss import CrossEntropy
from common.update_rule import standard_update_weight



class NeuralNetwork():

    def __init__(self, layer_sizes, num_classes=10, activation=ReLU, output_activation=Softmax, 
             learning_rate=0.01, weight_init=He, loss=CrossEntropy, update_rule = standard_update_weight , logger=None):
        """
        Inizializza la rete neurale.
        
        Args:
            layer_sizes: lista con il numero di neuroni per layer [input, hidden1, ..., output]
            num_classes: numero di classi per la classificazione
            activation: funzione di attivazione per i layer nascosti
            output_activation: funzione di attivazione per l'output
            learning_rate: learning rate per l'ottimizzazione
            weight_init: metodo di inizializzazione pesi
            loss: funzione di loss
            logger: istanza di Logger opzionale
        """
        
        self.logger = logger if logger else Logger()
        
        
        if layer_sizes[-1] != num_classes:
            raise ValueError(f"L'ultimo layer deve avere {num_classes} neuroni, ha {layer_sizes[-1]}")
        
        self.num_classes = num_classes
        self.fun_activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.loss = loss
        self.update_rule = update_rule
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.pre_activations = []
        self.dW = [] #gradineti pesi lista di matrici
        self.db = [] #gradienti baias 
        
        self._initialize_weights()
      
    
    def _initialize_weights(self):
        """Inizializza pesi e bias per tutti i layer."""
        for i in range(self.num_layers):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            

            W = self.weight_init(output_size,input_size)
            b = np.zeros((1, output_size))
               

            self.weights.append(W)
            self.biases.append(b)

        self.logger.print_matrix(self.weights, 'matrice dei pesi')
        self.logger.print_matrix(self.biases, 'matrice dei biases') 
        
        
    def _compute_delta(self,t):
        delta = self.output_activation.derivate(self.activations[-1], t)
        deltas = [delta] # una lista di matrici che contiene i delta che generano ogni layer, ogni matrice sarà di grandezza (numero di neuroni in quel layer,numero di input)


        for i in range(self.num_layers - 1, 0, -1): # è piu naturale cosi, se sono 2 layer(1 output e 1 hidden), fa 1(ultimo layer), 0(primo layer) è come un vettore C con indice che inizia da 0 a n-1
            delta_next = deltas[-1]
            delta = self.fun_activation.derivate(self.pre_activations[i-1]) * (self.weights[i].T @ delta_next) # i - 1 pensalo come il layer corrente
            deltas.append(delta)

        deltas.reverse()

        return deltas
    
    # normalizziamo per batch_size per evitare che  il gradeinte diventi troppo grande per via del aumentare della grandezza del batch
    def _compute_gradient(self, X, deltas):
        batch_size = X.shape[0]
        self.dW.append((deltas[0] @ X) / batch_size)    
        self.db.append(np.sum(deltas[0], axis=1, keepdims=True).T / batch_size)
        
        for i in range(1,self.num_layers) : 
            self.dW.append((deltas[i] @ self.activations[i - 1].T) / batch_size)
            self.db.append(np.sum(deltas[i], axis=1, keepdims=True).T / batch_size)

    def forward(self, X): 
        """
        Forward propagation.
        
        Args:
            X: input data (batch_size, input_features)
            
        Returns:
            output: prediction della rete
        """
        self.pre_activations.append((self.weights[0] @ X.T) + self.biases[0].T)

        #Se ho un singolo layer usa l'attivazione di output
        if self.num_layers == 1:
            self.activations.append(self.output_activation.activation(self.pre_activations[0]))
        else:
            self.activations.append(self.fun_activation.activation(self.pre_activations[0]))
         
        for step in range(1,self.num_layers):

            self.pre_activations.append((self.weights[step] @ self.activations[step-1]) + self.biases[step].T)
            
            if step == self.num_layers - 1 :
                self.activations.append(self.output_activation.activation(self.pre_activations[step]))
            else:
                self.activations.append(self.fun_activation.activation(self.pre_activations[step]))
            

        self.logger.print_matrix(self.pre_activations, 'matrice dei pre_activations')
        self.logger.print_matrix(self.activations, 'matrice dei activations') 
        return self.activations[self.num_layers-1]
    

    def backward(self, X, t):
        self.dW = []
        self.db = []
        deltas = self._compute_delta(t)
        self._compute_gradient(X, deltas)
        

            
    def update_weights(self):
        for i in range(self.num_layers):
            self.weights[i], self.biases[i] = standard_update_weight(self.weights[i],self.dW[i],self.biases[i],self.db[i],self.learning_rate)

    
    def train(self, X_train, y_train, epochs=1, batch_size=32, verbose=True):
        """
        Addestra la rete.
        
        Args:
            X_train: training data
            y_train: training labels
            epochs: numero di epoche
            batch_size: dimensione del batch
            verbose: se stampare info durante training
        """
        
    
        for epoche in range(epochs):
            loss = []
            for start in range(0, len(X_train), batch_size):
                batch = np.atleast_2d(X_train[start:start+batch_size])
                target = np.atleast_1d(y_train[start:start+batch_size])
                t = np.eye(self.num_classes)[target].T 
                self.forward(batch)                
                loss.append(self.loss(self.activations[self.num_layers - 1], t)) # qui si potrà sempre calcolare il prodotto perchè l'output sarà sempre un (10,size_of_input) e t sarà sempre (10,size_of_input)
                # la loss che uscirà sarà un numero che sarà la somma delle loss sui singoli esempi del batch
                self.backward(batch,t)
                self.update_weights()
                self.activations = []  
                self.pre_activations = []
            print(f"loss in epoca {epoche}: {loss[-1]}")
            self.logger.print_matrix(self.weights, 'matrice dei pesi')
            self.logger.print_matrix(self.biases, 'matrice dei biases') 

        

        
    def evaluate(self, X_test, y_test):
        """Valuta le performance su test set."""
        pass
