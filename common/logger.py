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
    
    def print(self, value, name=None):
        print(f'{name}:{value}\n')
    
