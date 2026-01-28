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
        self.prev_dW = [] # gradienti dell'epoca precedente
        self.prev_dB = []

    def _init(self, dW, db, learning_rate):
        self.prev_dW.append(np.zeros_like(dW)) 
        self.prev_dB.append(np.zeros_like(db)) 
        self.delta_w.append(np.full_like(dW, learning_rate)) 
        self.delta_b.append(np.full_like(db, learning_rate))
        
    def _update_param(self, param, grad, prev_grad, delta):
        """
        Helper method per aggiornare un parametro (peso o bias) usando RProp.
        """
        # Prodotto tra gradiente attuale e gradiente dell'epoca precedente
        prod = grad * prev_grad
        
        # Se il segno è uguale, aumenta lo step (delta)
        delta = np.where(
            prod > 0,
            np.minimum(delta * self.eta_plus, self.step_max),
            delta
        )
        
        # Se il segno cambia, diminuisci lo step (delta)
        delta = np.where(
            prod < 0,
            np.maximum(delta * self.eta_minus, self.step_min),
            delta
        )
        
        # Se il segno è cambiato, il gradiente viene azzerato per non influenzare il prossimo passo
        grad_updated = np.where(prod < 0, 0, grad)
        
        # Aggiorna il parametro
        param_updated = param - (delta * np.sign(grad_updated))
        
        return param_updated, grad_updated, delta

    def __call__(self, weight, dW, biase, db, learning_rate, layer_index=0):
        """
            weight : matrice di pesi
            dW : matrice dei gradienti al tempo T di un layer
            biase: vettore dei bias
            db: gradiente dei bias al tempo T
            layer_index : indice del layer in cui sto aggiornando la matrice
        """
        
        dW = np.atleast_2d(dW)
        db = np.atleast_2d(db)

        # init della matrice del del layer in cui siamo dato da layer_index
        # se len(self.prev_dW) == layer_index allora abbiamo abbastanza matrici per ogni layer
        if len(self.prev_dW) <= layer_index:
            self._init(dW, db, learning_rate)

        # Aggiornamento Pesi
        # prev_dW[layer_index] sono i gradienti dell'epoca precedente per i pesi
        weight, dW, self.delta_w[layer_index] = self._update_param(
            weight, dW, self.prev_dW[layer_index], self.delta_w[layer_index]
        )

        # Aggiornamento Bias
        # prev_dB[layer_index] sono i gradienti dell'epoca precedente per i bias
        biase, db, self.delta_b[layer_index] = self._update_param(
            biase, db, self.prev_dB[layer_index], self.delta_b[layer_index]
        )

        # Aggiorno i gradienti "precedenti" con quelli attuali (modificati) per la prossima iterazione
        self.prev_dW[layer_index] = dW.copy()
        self.prev_dB[layer_index] = db.copy()
        
        return weight, biase


class Adam(UpdateRule):
    def __init__(
        self,
        betas1=0.9,        # β1
        betas2=0.999,      # β2
        eps=1e-8,          # ε
        weight_decay=0.0,  # λ (L2 inside grad, "classic Adam")
        amsgrad=False
    ):
        self.beta1 = betas1
        self.beta2 = betas2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.t = 0  

        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []

        self.v_w_max, self.v_b_max = [], []

    def _init_layer(self, w, b):
        self.m_w.append(np.zeros_like(w)) # è una matrice del primo momento
        self.v_w.append(np.zeros_like(w)) # è una matrice del secondo momento
        self.m_b.append(np.zeros_like(b))
        self.v_b.append(np.zeros_like(b))

        if self.amsgrad:
            self.v_w_max.append(np.zeros_like(w))
            self.v_b_max.append(np.zeros_like(b))

    def __call__(self, weights, dW, biases, db, learning_rate, layer_index=0):
        dW = np.asarray(dW)
        db = np.asarray(db)

        while len(self.m_w) <= layer_index:
            self._init_layer(weights, biases)

        if layer_index == 0:
            self.t += 1
        t = self.t

        if self.weight_decay != 0.0:
            dW = dW + self.weight_decay * weights
            db = db + self.weight_decay * biases 

        #update first moment
        self.m_w[layer_index] = self.beta1 * self.m_w[layer_index] + (1.0 - self.beta1) * dW
        self.m_b[layer_index] = self.beta1 * self.m_b[layer_index] + (1.0 - self.beta1) * db

        #update second moment
        self.v_w[layer_index] = self.beta2 * self.v_w[layer_index] + (1.0 - self.beta2) * (dW * dW)
        self.v_b[layer_index] = self.beta2 * self.v_b[layer_index] + (1.0 - self.beta2) * (db * db)

        m_w_hat = self.m_w[layer_index] / (1.0 - (self.beta1 ** self.t))
        m_b_hat = self.m_b[layer_index] / (1.0 - (self.beta1 ** self.t))

        if self.amsgrad:
            self.v_w_max[layer_index] = np.maximum(self.v_w_max[layer_index], self.v_w[layer_index])
            self.v_b_max[layer_index] = np.maximum(self.v_b_max[layer_index], self.v_b[layer_index])

            v_w_hat = self.v_w_max[layer_index] / (1.0 - (self.beta2 ** self.t))
            v_b_hat = self.v_b_max[layer_index] / (1.0 - (self.beta2 ** self.t))
        else:
            v_w_hat = self.v_w[layer_index] / (1.0 - (self.beta2 ** self.t))
            v_b_hat = self.v_b[layer_index] / (1.0 - (self.beta2 ** self.t))

       
        weights = weights - learning_rate * (m_w_hat / (np.sqrt(v_w_hat) + self.eps))
        biases  = biases  - learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + self.eps))

        return weights, biases



standard = StandardUpdateRule()



