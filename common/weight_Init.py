import numpy as np

np.random.seed(16)

def He(output_size , input_size):
    return np.random.randn( output_size,input_size) * np.sqrt(2.0 / input_size)

def Glorot(output_size , input_size):
    limit = np.sqrt(6.0 / (input_size + output_size))
    return np.random.uniform(-limit, limit, size=(output_size, input_size))

def Contadina(output_size , input_size):
    return np.zeros((output_size,input_size))
