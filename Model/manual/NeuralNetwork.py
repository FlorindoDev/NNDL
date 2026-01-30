import numpy as np
import matplotlib.pyplot as plt

from common.activation import ReLU, Softmax
from common.weight_Init import He
from common.logger import Logger
from common.loss import CrossEntropy
from common.update_rule import  standard



class NeuralNetwork():

    def __init__(self, layer_sizes, num_classes=10, activation=ReLU, output_activation=Softmax, 
                weight_init=He, loss=CrossEntropy, update_rule = standard , logger=None):
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
        self.weight_init = weight_init
        self.loss = loss
        self.update_rule = update_rule
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.pre_activations = []
        self.dW = []                #gradineti pesi lista di matrici
        self.db = []                #gradienti baias 

        self.validation_loss = []

        #Early stopping
        self.best_loss = float("inf")
        
        
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

        # una lista di matrici che contiene i delta che generano ogni layer, ogni matrice sarà di grandezza (numero di neuroni in quel layer,numero di input)
        deltas = [delta] 

        # è piu naturale cosi, se sono 2 layer(1 output e 1 hidden), fa 1(ultimo layer), 0(primo layer) è come un vettore C con indice che inizia da 0 a n-1
        for i in range(self.num_layers - 1, 0, -1): 
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

    def _one_hot(self,target):
        return np.eye(self.num_classes)[target].T


    def _shuffle(self, X_train, y_train):
            idx = np.random.permutation(len(X_train))
            X_train = X_train[idx]
            y_train = y_train[idx]
            return X_train, y_train


    def _compute_validation_loss(self, X,y,batch_size=32):
        local_loss = []
        for start in range(0, len(X), batch_size):
            batch = np.atleast_2d(X[start:start+batch_size])
            target = np.atleast_1d(y[start:start+batch_size])

            one_hot = self._one_hot(target)

            self.forward(batch)

            loss_val=self.loss(self.activations[self.num_layers - 1], one_hot)
            local_loss.append(loss_val)

        self.validation_loss.append(float(np.mean(local_loss)))

        
    def early_stopping(self):        
        if (self.best_loss > self.validation_loss[-1]):
            self.best_loss = self.validation_loss[-1]
            return True
        return False
           

    def forward(self, X): 
        """
        Forward propagation.
        
        Args:
            X: input data (batch_size, input_features)
            
        Returns:
            output: prediction della rete
        """

        self.activations = []  
        self.pre_activations = []
        

        pre_activation_first_layer = (self.weights[0] @ X.T) + self.biases[0].T
        self.pre_activations.append(pre_activation_first_layer)

        #Se ho un singolo layer usa l'attivazione di output
        if self.num_layers == 1:
            self.activations.append(self.output_activation.activation(pre_activation_first_layer))
        else:
            self.activations.append(self.fun_activation.activation(pre_activation_first_layer))
         
        for step in range(1,self.num_layers):

            pre_activation_layer = (self.weights[step] @ self.activations[step-1]) + self.biases[step].T

            self.pre_activations.append(pre_activation_layer)
            
            if step == self.num_layers - 1 :
                self.activations.append(self.output_activation.activation(pre_activation_layer))
            else:
                self.activations.append(self.fun_activation.activation(pre_activation_layer))
            

        self.logger.print_matrix(self.pre_activations, 'matrice dei pre_activations')
        self.logger.print_matrix(self.activations, 'matrice dei activations') 
        return self.activations[self.num_layers-1]
    
  
        
    def backward(self, X, t):
        self.dW = []
        self.db = []
        deltas = self._compute_delta(t)
        self._compute_gradient(X, deltas)
        

            
    def update_weights(self,learning_rate):
        """
        Aggiorna i pesi
        """
        for i in range(self.num_layers):
            self.weights[i],self.biases[i] = self.update_rule(
                    self.weights[i],
                    self.dW[i],
                    self.biases[i],
                    self.db[i],
                    learning_rate,
                    i
                )
            

    
    def train(self, X_train, y_train, epochs=1, batch_size=32, learning_rate=0.01, early_stopping=False, X_validation=[], y_validation=[],pacience=5):
        """
        Addestra la rete.
        
        Args:
            X_train: training data
            y_train: training labels
            epochs: numero di epoche
            batch_size: dimensione del batch
        """
        local_pacience = pacience
        self.train_losses = []
        best_weights = []
        best_biases = []
        
        for epoche in range(epochs):
            
            epoch_batch_losses = []
            X_train, y_train = self._shuffle(X_train,y_train)

            for start in range(0, len(X_train), batch_size):

                batch = np.atleast_2d(X_train[start:start+batch_size])
                target = np.atleast_1d(y_train[start:start+batch_size])

                one_hot = self._one_hot(target)

                self.forward(batch)   

                # la loss che uscirà sarà un numero che sarà la somma delle loss sui singoli esempi del batch
                #qui si potrà sempre calcolare il prodotto perchè l'output sarà sempre un (10,size_of_input) e t sarà sempre (10,size_of_input)             
                loss_val=self.loss(self.activations[self.num_layers - 1], one_hot)
                epoch_batch_losses.append(loss_val)

                self.backward(batch,one_hot)

                self.update_weights(learning_rate)


            if early_stopping:
                self._compute_validation_loss(X_validation,y_validation,batch_size)
                self.logger.print(self.validation_loss[-1], f"loss validation in epoca {epoche + 1}", True)
                
                find_best = self.early_stopping()
                
                if find_best:
                    #significa che ho trovato un minimo val_loss
                    local_pacience = pacience
                    best_weights = self.weights.copy()
                    best_biases = self.biases.copy()
                else:
                    # non abbiamo trovato uno migliore, decrementa
                    local_pacience -= 1

            
            
            epoch_loss = float(np.mean(epoch_batch_losses))
            self.train_losses.append(epoch_loss)
            self.logger.print(epoch_loss, f"loss in epoca {epoche + 1}", True)

            self.logger.print_matrix(self.weights, 'matrice dei pesi')
            self.logger.print_matrix(self.biases, 'matrice dei biases') 


            if early_stopping and local_pacience == 0:
                # Ripristina i migliori pesi e bias
                self.weights = best_weights.copy()
                self.biases = best_biases.copy()
                self.logger.print(0, "Early stopping attivato", True)
                return
        


        return    
    
            
    def evaluate(self, X_test, y_test, early_stopping=False):
        """Valuta le performance su test set."""

        X = np.atleast_2d(X_test)
        y = np.atleast_1d(y_test)
        if len(X) != len(y):
            raise ValueError("X_test e y_test devono avere la stessa lunghezza.")



        outputs = self.forward(X)
        y_pred = np.argmax(outputs, axis=0)
        accuracy = np.mean(y_pred == y)

        if early_stopping:
            # indice epoca best
            best_epoch = np.argmin(self.validation_loss)

        # Grafico
        plt.figure()
        plt.plot(self.train_losses, label="Train loss")

        if early_stopping:

            plt.plot(self.validation_loss, label="Validation loss")

            # linea verticale nera sull’epoca best
            plt.axvline(
                x=best_epoch,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Best epoch"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Error Curve")
        plt.grid(True)
        plt.legend()

        plt.show()

        print(accuracy)
        # self.logger.print(accuracy, "Accuracy sul test set: ", True)
