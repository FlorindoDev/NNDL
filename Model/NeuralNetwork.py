import numpy as np
from common.Activation import ReLU, softmax
from common.Weight_Init import He
from common.Logger import Logger
from common.Loss import CrossEntropy



class NeuralNetwork():

    def __init__(self, layer_sizes, num_classes=10, activation=ReLU, output_activation=softmax, 
             learning_rate=0.01, weight_init=He, loss=CrossEntropy, logger=None):
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
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.loss = loss
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.pre_activations = []
        self.dW = []
        self.db = []
        
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

        #self.logger.print_matrix(self.weights, 'matrice dei pesi')
        #self.logger.print_matrix(self.biases, 'matrice dei biases') 
        
    
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
            self.activations.append(self.output_activation(self.pre_activations[0]))
        else:
            self.activations.append(self.activation(self.pre_activations[0]))
         
        for step in range(1,self.num_layers):

            self.pre_activations.append((self.weights[step] @ self.activations[step-1]) + self.biases[step].T)
            
            if step == self.num_layers - 1 :
                self.activations.append(self.output_activation(self.pre_activations[step]))
            else:
                self.activations.append(self.activation(self.pre_activations[step]))
            
        

        #self.logger.print_matrix(self.pre_activations, 'matrice dei pre_activations')
        self.logger.print_matrix(self.activations, 'matrice dei activations') 
        
    
    def backward(self, X, y):
        """
        Backward propagation - calcola i gradienti.
        
        Args:
            X: input data
            y: target labels
        """
        pass
    
    def update_weights(self):
        """Aggiorna i pesi usando i gradienti calcolati."""
        pass
    
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
        
    
        for epoch in range(epochs):
            for start in range(0, len(X_train), batch_size):
                batch = np.atleast_2d(X_train[start:start+batch_size])
                target = np.atleast_1d(y_train[start:start+batch_size])
                
                t = np.eye(self.num_classes)[target].T
                
                self.forward(batch)
                self.loss(self.activations[self.num_layers - 1], t) # qui si potrà sempre calcolare il prodotto perchè l'output sarà sempre un (10,size_of_input) e t sarà sempre (10,size_of_input)
                # Qui abbiamo l'errore del Batch e ora bisogna calcolare la derivata
                # TODO: dopo dobbiamo fare backprop e update dei pesi
                
                self.activations = []  
                self.pre_activations = []

        

        
    def evaluate(self, X_test, y_test):
        """Valuta le performance su test set."""
        pass
