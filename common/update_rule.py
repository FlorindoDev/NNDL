import numpy as np

class UpdateRule:
    def __call__(weights,dW,biases,db,learning_rate,layer_index=0):
        raise NotImplementedError("UpdateRule is an abstract base class.")


class StandardUpdateRule(UpdateRule):

    def _standard_update_weight(self, weights,dW,biases,db,learning_rate):
        dW = np.atleast_2d(dW)
        weights = weights - (dW * learning_rate)
        db = np.atleast_2d(db)
        biases = biases - (db * learning_rate)
        return weights, biases

    def __call__(self, weights,dW,biases,db,learning_rate,layer_index=0):
        return self._standard_update_weight(weights,dW,biases,db,learning_rate)


class RProp(UpdateRule):
    def __init__(self, eta_plus=1.2, eta_minus=0.5, step_max=20, step_min=1e-6):
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.step_max = step_max
        self.step_min = step_min
        self.delta_b = []
        self.delta_w = []
        self.prev_dW = []
        self.prev_dB = []
        # self._state = {}

    def _init(self, dW, db, learning_rate):
        self.prev_dW.append(np.zeros_like(dW)) 
        self.prev_dB.append(np.zeros_like(db)) 
        self.delta_w.append(np.full_like(dW, learning_rate)) 
        self.delta_b.append(np.full_like(db, learning_rate))
        

    def __call__(self, weight, dW, biase, db, learning_rate, layer_index=0):
        dW = np.atleast_2d(dW)
        db = np.atleast_2d(db)

        while len(self.prev_dW) <= layer_index:
            self._init(dW , db, learning_rate)

        prod_w = dW * self.prev_dW[layer_index]
        self.delta_w[layer_index] = np.where(
            prod_w > 0,
            np.minimum(self.delta_w[layer_index] * self.eta_plus, self.step_max),
            self.delta_w[layer_index]
        )
        self.delta_w[layer_index] = np.where(
            prod_w < 0,
            np.maximum(self.delta_w[layer_index] * self.eta_minus, self.step_min),
            self.delta_w[layer_index]
        )
        dW = np.where(prod_w < 0, 0, dW)
        weight = weight - (self.delta_w[layer_index] * np.sign(dW))

        prod_b = db * self.prev_dB[layer_index]
        self.delta_b[layer_index] = np.where(
            prod_b > 0,
            np.minimum(self.delta_b[layer_index] * self.eta_plus, self.step_max),
            self.delta_b[layer_index]
        )
        self.delta_b[layer_index] = np.where(
            prod_b < 0,
            np.maximum(self.delta_b[layer_index] * self.eta_minus, self.step_min),
            self.delta_b[layer_index]
        )
        db = np.where(prod_b < 0, 0, db)
        biase = biase - (self.delta_b[layer_index] * np.sign(db))

        self.prev_dW[layer_index] = dW.copy()
        self.prev_dB[layer_index] = db.copy()
        
        return weight, biase


standard = StandardUpdateRule()



