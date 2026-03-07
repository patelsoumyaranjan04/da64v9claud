import numpy as np


def _clip(x):
    return np.clip(x, -500.0, 500.0)

class Activation:
    
    def forward(self, z):
        raise NotImplementedError
    
    def backward(self, z):
        raise NotImplementedError


class ReLU(Activation):

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        grad =(x > 0).astype(float)
        return grad


class Sigmoid(Activation):

    def forward(self, x):
        x = _clip(x)
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        s = self.forward(x)
        return s * (1.0 - s)


class Tanh(Activation):

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        t = np.tanh(x)
        return 1.0 - t * t


class Softmax(Activation):

    def forward(self, z):
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


    def backward(self, z):
        s = self.forward(z)
        return s * (1.0 - s)


class Identity(Activation):

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


def get_activation(name):
    activations = {
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'relu': ReLU(),
        'softmax': Softmax(),
        'identiy':Identity()
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    
    return activations[name.lower()]