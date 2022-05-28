"""
Optimizers for the network
"""
from typing import Protocol
import numpy as np

class Optimizer(Protocol):
    def __call__(self, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        """Returns optimized gradients"""
        
class GradientDescent():
    def __call__(self, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        return gradients * learning_rate

class Momentum():
    beta = 0.9

    def __call__(self, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        vgrad = self.beta * gradients + (1 - self.beta) * gradients
        return vgrad * learning_rate

class RMSprop():
    beta = 0.9
    epsilon = 1e-7

    def __call__(self, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        vgrad = self.beta * gradients + (1 - self.beta) * np.power(gradients, 2) 
        return learning_rate * (vgrad / np.sqrt(vgrad + self.epsilon))

class Adam():
    beta1 = 0.9
    beta2 = 0.99
    epsilon = 1e-7
    
    def __call__(self, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        vgrad = self.beta1 * v["dW" + str(l)] + (1 - self.beta1) * grads['dW' + str(l)]
        pass

