"""
Neural network optimizers
"""
import numpy as np
from abc import ABC, abstractmethod
import torch
from typing import Dict, List

def init_velocity_vector(layers):
    velocity = {}
    # build layers velocity dict with np zeros array for each trainable pram
    for idx, layer in enumerate(layers):
        trainable_params = layer.get_trainable_params_gradients()
        velocity[idx] = trainable_params
        for param, grads in trainable_params.items():
            velocity[idx][param] = np.zeros(grads.shape)
    return velocity

class Optimizer(ABC):
    """
    Only one optimizer can be assigned to the entire NN

    The optimizer is in charge of keeping track of optimization
    variable states for all layers. Check Adam optimizer for reference.

    Optimizer workflow
    1. Loop through self.params (coming from Neural Network layers)
    3. Optimize those parameters according to strategy
    4. (optional) track optimization
    5. Update layer parameters on learning rate scale
    """
    def __init__(self, layers_params: List[Dict[str, torch.Tensor]], learning_rate: float = 1e-2):
        #assert isinstance(network, NeuralNetwork), "network should be a NeuralNetwork object"
        self.params = layers_params
        self.learning_rate = learning_rate
    
    def zero_grad(self) -> None:
        """Sets all parameters gradients to zero"""
        for layer in self.params:
            for _, param_value in layer.items():
                param_value.grad = torch.zeros(param_value.shape)
        
    @abstractmethod
    def step(self) -> None:
        """Runs optimizer for all trainable parameters returned by the layer"""
        
class GradientDescent(Optimizer):
    def step(self) -> None:
        for layer in self.params:
            for param_name, param_value in layer.items():
                direction = param_value.grad * self.learning_rate
                # Running in place operation with context manager
                # because we dont need to track this operation
                with torch.no_grad():
                    layer[param_name] -= direction

class Momentum(Optimizer):
    beta = 0.9

    def __init__(self, layers_params: List[Dict[str, torch.Tensor]], learning_rate: float = 1e-2):
        super().__init__(layers_params, learning_rate)
        self.gradients_velocity = init_velocity_vector(self.params)


    def step(self) -> None:
        for idx, layer in enumerate(self.params):
            trainable_params = layer.get_trainable_params_gradients()
            for param, grads in trainable_params.items():
                self.gradients_velocity[idx][param] = self.beta * self.gradients_velocity[idx][param] + (1 - self.beta) * grads
                trainable_params[param] = self.gradients_velocity[idx][param] * self.learning_rate
            layer.update_trainable_params(trainable_params)

class RMSprop(Optimizer):
    beta = 0.9
    epsilon = 1e-7

    def __init__(self, layers_params: List[Dict[str, torch.Tensor]], learning_rate: float = 1e-2):
        super().__init__(layers_params, learning_rate)
        self.gradients_velocity = init_velocity_vector(self.params)

    def step(self) -> None:
        for idx, layer in enumerate(self.params):
            trainable_params = layer.get_trainable_params_gradients()
            for param, grads in trainable_params.items():
                self.gradients_velocity[idx][param] = self.beta * self.gradients_velocity[idx][param] + (1 - self.beta) * np.power(grads, 2)
                trainable_params[param] = (grads / np.sqrt(self.gradients_velocity[idx][param] + self.epsilon)) * self.learning_rate
            layer.update_trainable_params(trainable_params)

class Adam(Optimizer):
    """Adam is a combination of momentum and RMSprop, thats why the velocity names"""
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-7

    def __init__(self, layers_params: List[Dict[str, torch.Tensor]], learning_rate = 1e-2):
        super().__init__(layers_params, learning_rate)
        self.momentum_velocity = init_velocity_vector(self.params)
        self.rmsprop_velocity = init_velocity_vector(self.params)

    def step(self) -> None:
        for idx, layer in enumerate(self.params):
            trainable_params = layer.get_trainable_params_gradients()
            for param, grads in trainable_params.items():
                self.momentum_velocity[idx][param] = ((self.beta1 * self.momentum_velocity[idx][param]) + ((1 - self.beta1) * grads)) 
                self.rmsprop_velocity[idx][param] = ((self.beta2 * self.rmsprop_velocity[idx][param]) + ((1 - self.beta2) * np.power(grads, 2)))
                trainable_params[param] = (self.momentum_velocity[idx][param] / np.sqrt(self.rmsprop_velocity[idx][param] + self.epsilon)) * self.learning_rate
            layer.update_trainable_params(trainable_params)

