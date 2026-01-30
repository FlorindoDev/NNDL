import numpy as np
import os
from dotenv import load_dotenv


class Logger():

    def __init__(self):
        load_dotenv() 
        self.active = os.getenv("LOGGER", "FALSE").upper() == "TRUE"



    def print_matrix(self, matrix, name=None,bypass=False):
        """
        Stampa una matrice (o lista di matrici) in modo leggibile.
        
        Args:
            matrix: numpy array o lista di numpy arrays
            name: nome opzionale da stampare prima della matrice
        """
        
        if not self.active and not bypass:
            return

        if name:
            print(f"\n{name}:")
            print("-" * (len(name) + 1))
        
        if isinstance(matrix, list):
            for i, m in enumerate(matrix):
                print(f"Layer {i}:")
            
                with np.printoptions(precision=4, suppress=True, linewidth=150):
                    print(m)
                print("")
        else:
            with np.printoptions(precision=4, suppress=True, linewidth=150):
                print(matrix)
    
    def print(self, value, name=None, bypass=False):

        if not self.active and not bypass:
            return
    
        print(f'{name}: {value}\n')

    def print_triaing_progress(self,epoch,epochs,val_loss,train_loss,current_patience, patience, early_stopping, bypass=False):

        if not self.active and not bypass:
            return

        print(f"\n\nEpoch {epoch + 1}/{epochs}\n" + "-" * 40)
        print(f"Training loss: {train_loss:.6f}")

        if early_stopping:
            print(f"Validation loss: {val_loss:.6f}")
            if(current_patience == patience):
                print(f"✓ New best model saved (loss: {val_loss:.6f})")
            elif(current_patience < patience and current_patience != 0):
                print(f"No improvement. Patience: {current_patience}/{patience}")
            else:
                print(f"No improvement. Patience: {current_patience}/{patience}")
                print("\n⚠ Early stopping triggered!")
    
    def print_test_evaluation(self,accuracy,  bypass=False):

        if not self.active and not bypass:
            return

        print("\n" + "=" * 50)
        print("FINAL TEST EVALUATION")
        print("=" * 50)
        print(f"Test Error:\n Accuracy: {100 * accuracy:>0.1f}%")