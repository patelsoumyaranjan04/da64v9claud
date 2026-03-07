
import numpy as np


class Optimizer:
   
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.state = {}
    
    def update(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, layers):
        for layer in layers:
            if layer.grad_W is not None:
                layer.W -= self.learning_rate * layer.grad_W
                layer.b -= self.learning_rate * layer.grad_b

class Momentum(Optimizer):
    
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
    
    def update(self, layers):
        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                if i not in self.state:
                    self.state[i] = {
                        'v_W': np.zeros_like(layer.W),
                        'v_b': np.zeros_like(layer.b)
                    }
                
                # Update velocity
                self.state[i]['v_W'] = self.beta * self.state[i]['v_W'] - self.learning_rate * layer.grad_W
                self.state[i]['v_b'] = self.beta * self.state[i]['v_b'] - self.learning_rate * layer.grad_b
                
                # Update weights
                layer.W += self.state[i]['v_W']
                layer.b += self.state[i]['v_b']


class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):

        super().__init__(learning_rate, weight_decay)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None

    def _init_moments(self, layers):

        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]

        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):

        if self.m_W is None:
            self._init_moments(layers)

        self.t += 1

        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i, layer in enumerate(layers):

            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grad_W
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grad_W ** 2

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b ** 2

            layer.W -= lr_t * self.m_W[i] / (np.sqrt(self.v_W[i]) + self.eps)
            layer.b -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.eps)


class Nadam(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):

        super().__init__(learning_rate, weight_decay)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None

    def _init_moments(self, layers):

        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]

        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):

        if self.m_W is None:
            self._init_moments(layers)

        self.t += 1
        b1, b2 = self.beta1, self.beta2

        for i, layer in enumerate(layers):

            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.m_W[i] = b1 * self.m_W[i] + (1 - b1) * grad_W
            self.v_W[i] = b2 * self.v_W[i] + (1 - b2) * grad_W ** 2

            self.m_b[i] = b1 * self.m_b[i] + (1 - b1) * grad_b
            self.v_b[i] = b2 * self.v_b[i] + (1 - b2) * grad_b ** 2

            m_hat_W = self.m_W[i] / (1 - b1 ** self.t)
            v_hat_W = self.v_W[i] / (1 - b2 ** self.t)

            m_hat_b = self.m_b[i] / (1 - b1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - b2 ** self.t)

            nesterov_W = b1 * m_hat_W + (1 - b1) * grad_W / (1 - b1 ** self.t)
            nesterov_b = b1 * m_hat_b + (1 - b1) * grad_b / (1 - b1 ** self.t)

            layer.W -= self.lr * nesterov_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * nesterov_b / (np.sqrt(v_hat_b) + self.eps)


class NAG(Optimizer):
    
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
    
    def update(self, layers):
        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                if i not in self.state:
                    self.state[i] = {
                        'v_W': np.zeros_like(layer.W),
                        'v_b': np.zeros_like(layer.b)
                    }
                
                # Store old velocity
                v_W_old = self.state[i]['v_W'].copy()
                v_b_old = self.state[i]['v_b'].copy()
                
                # Update velocity
                self.state[i]['v_W'] = self.beta * self.state[i]['v_W'] - self.learning_rate * layer.grad_W
                self.state[i]['v_b'] = self.beta * self.state[i]['v_b'] - self.learning_rate * layer.grad_b
                
                layer.W += -self.beta * v_W_old + (1 + self.beta) * self.state[i]['v_W']
                layer.b += -self.beta * v_b_old + (1 + self.beta) * self.state[i]['v_b']


class RMSProp(Optimizer):
    
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
    
    def update(self, layers):
        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                if i not in self.state:
                    self.state[i] = {
                        's_W': np.zeros_like(layer.W),
                        's_b': np.zeros_like(layer.b)
                    }
                
                self.state[i]['s_W'] = self.beta * self.state[i]['s_W'] + (1 - self.beta) * (layer.grad_W ** 2)
                self.state[i]['s_b'] = self.beta * self.state[i]['s_b'] + (1 - self.beta) * (layer.grad_b ** 2)
                
                layer.W -= self.learning_rate * layer.grad_W / (np.sqrt(self.state[i]['s_W']) + self.epsilon)
                layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(self.state[i]['s_b']) + self.epsilon)


def get_optimizer(name, learning_rate=0.01):
    optimizers = {
        'sgd': SGD(learning_rate),
        'momentum': Momentum(learning_rate),
        'nag': NAG(learning_rate),
        'rmsprop': RMSProp(learning_rate)
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name.lower()]


