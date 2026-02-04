import numpy as np
from tensorflow.keras.datasets import mnist
from Model.manual.NeuralNetwork import NeuralNetwork as nn
from common.activation import ReLU, Softmax, LeakyReLU
from common.weight_Init import He
from common.logger import Logger
from common.dataset_preprocessor import DatasetPreprocessor
from common.update_rule import RProp,Adam

def main():
    """Entry point principale per Rete Neurale Manuale."""
    # Flag di esecuzione
    TRAIN_AND_SAVE = False  # Se True, allena il modello e salva i pesi
    LOAD_AND_TEST = True    # Se True, carica i pesi esistenti e testa il modello

    logger = Logger()
    dataset_preprocessor = DatasetPreprocessor()
    
    # Carica dati
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preprocessor.prepare_from_splits(x_train, y_train, x_test, y_test, 0.30)

    # Parametri rete
    layer_sizes = [784, 512,256,128,64, 10]
    update_rule = Adam()
    weight_init = He
    activation = LeakyReLU
    
    # Nome file per i pesi
    weights_path = "Model/manual/weights/fcnn_manual_weights.npz"

    rete = nn(
        layer_sizes=layer_sizes, 
        update_rule=update_rule, 
        activation=activation, 
        weight_init=weight_init, 
        logger=logger
    )

    if TRAIN_AND_SAVE:
        print(f"\n--- TRAINING MODE (Manual NN) ---")
        rete.train(
            x_train, y_train, 
            epochs=50, 
            batch_size=32, 
            early_stopping=True, 
            X_validation=x_val, 
            y_validation=y_val, 
            patience=5
        )
        
        # Salvataggio pesi
        import os
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        rete.save_weights(weights_path)
        
        print("\nEVALUATION ON TEST SET")
        rete.evaluate(x_test, y_test, True)

    if LOAD_AND_TEST:
        print(f"\n--- LOAD & TEST MODE (Manual NN) ---")
        rete.load_weights(weights_path)
        
        print("\nEVALUATION ON TEST SET")
        rete.evaluate(x_test, y_test, False,LOAD_AND_TEST) # False perché non abbiamo history se carichiamo solo i pesi

if __name__ == "__main__":
    main()
