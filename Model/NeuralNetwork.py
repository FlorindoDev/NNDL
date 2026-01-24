import numpy as np
from Function.Activation import ReLU, softmax
from Function.Weight_Init import He



class NeuralNetwork():
    def __init__(self, layer_sizes, activation=ReLU, output_activation=softmax, learning_rate=0.01, weight_init=He):
        """
        Inizializza la rete neurale.
        
        Args:
            layer_sizes: lista con il numero di neuroni per layer [input, hidden1, ..., output]
            activation: funzione di attivazione per i layer nascosti ('relu', 'sigmoid', 'tanh')
            output_activation: funzione di attivazione per l'output ('softmax', 'sigmoid', 'linear')
            learning_rate: learning rate per l'ottimizzazione
            weight_init: metodo di inizializzazione pesi ('he', 'xavier', 'random')
        """

        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1 # Senza input
        
        
        # Liste per pesi e bias
        self.weights = []  # W[0] = pesi tra input e primo hidden layer
        self.biases = []   # b[0] = bias del primo hidden layer
        
        # Liste per memorizzare attivazioni durante forward pass
        self.activations = []  # z[0] = input, z[1] = primo hidden, ...
        self.pre_activations = []     # pre-attivazioni
        
        # Gradienti
        self.dW = []  # gradienti per i pesi
        self.db = []  # gradienti per i bias

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

        print(f'matrice dei pesi \n{self.weights}') 
        print(f'matrice dei biases \n{self.biases}') 
        
    
    def _activation(self, z, function_name):
        """Applica la funzione di attivazione."""
        pass
    
    def _activation_derivative(self, z, function_name):
        """Calcola la derivata della funzione di attivazione."""
        pass
    
    def forward(self, X):
        """
        Forward propagation.
        
        Args:
            X: input data (batch_size, input_features)
            
        Returns:
            output: prediction della rete
        """

        self.pre_activations.append((self.weights[0] @ X.T) + self.biases[0].T)
        self.activations.append(self.activation(self.pre_activations[0]))
         
        for step in range(1,self.num_layers):

            self.pre_activations.append((self.weights[step] @ self.activations[step-1]) + self.biases[step].T)
            
            if step == self.num_layers - 1 :
                self.activations.append(self.output_activation(self.pre_activations[step]))
            else:
                self.activations.append(self.activation(self.pre_activations[step]))
            
        

        print(f'matrice dei pre_activations \n{self.pre_activations}') 
        print(f'matrice dei activations \n{self.activations}') 
        
    
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
    
    def compute_loss(self, y_true, y_pred):
        """Calcola la loss (cross-entropy o MSE)."""
        pass
    
    def train(self, X_train, y_train, epochs, batch_size=32, verbose=True):
        """
        Addestra la rete.
        
        Args:
            X_train: training data
            y_train: training labels
            epochs: numero di epoche
            batch_size: dimensione del batch
            verbose: se stampare info durante training
        """
        pass
    
    
    def evaluate(self, X_test, y_test):
        """Valuta le performance su test set."""
        pass
