"""
Neural network optimizers
"""
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from redmind.layers import Layer

class Optimizer(ABC):
    """
    Only one optimizer can be assigned to the entire NN

    The optimizer is in charge of keeping track of optimization
    variable states for all layers. Check Adam optimizer for reference.

    Optimizer workflow
    1. Loop through self.layers
    2. Fetch all trainable parameters
    3. Optimize those parameters according to strategy
    4. (optional) track optimization
    5. Update layer parameters on learning rate scale
    """
    def set_layers(self, layers: List[Layer]) -> None:
        self.layers = layers
    
    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def __call__(self) -> None:
        """Runs optimizer for all trainable parameters returned by the layer"""
        
class GradientDescent(Optimizer):
    def __call__(self) -> None:
        for layer in self.layers:
            trainable_params = layer.get_trainable_params()
            for k, v in trainable_params.items():
                trainable_params[k] = v * self.learning_rate
            layer.update_trainable_params(trainable_params)

class Momentum(Optimizer):
    beta = 0.9
    velocity = None

    def __init__(self) -> None: 
        self.gradients_velocity = []

    def __call__(self) -> None:
        vgrad = self.beta * self.velocity + (1 - self.beta) * gradients
        return vgrad * learning_rate

class RMSprop(Optimizer):
    beta = 0.9
    epsilon = 1e-7

    def __call__(self) -> None:
        vgrad = self.beta * gradients + (1 - self.beta) * np.power(gradients, 2) 
        return learning_rate * (vgrad / np.sqrt(vgrad + self.epsilon))

class Adam(Optimizer):
    beta1 = 0.9
    beta2 = 0.99
    epsilon = 1e-7

    def __call__(self) -> None:
        vgrad = self.beta1 * v["dW" + str(l)] + (1 - self.beta1) * grads['dW' + str(l)]
        pass

