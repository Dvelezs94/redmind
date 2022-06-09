"""
Neural network optimizers
"""
from abc import ABC, abstractmethod
import torch

def init_velocity_vector(layers: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    velocity_layers_list = []
    # build layers velocity dict with np zeros array for each trainable param
    for layer in layers:
        velocity_map = {}
        for param_name, param_value in layer.items():
            velocity_map[param_name] = torch.zeros(param_value.shape)
        velocity_layers_list.append(velocity_map)
    return velocity_layers_list

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
    def __init__(self, layers_params: list[dict[str, torch.Tensor]], learning_rate: float = 1e-2):
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
    beta: float

    def __init__(self, layers_params: list[dict[str, torch.Tensor]], learning_rate: float = 1e-2, beta = 0.9):
        super().__init__(layers_params, learning_rate)
        self.beta = beta
        self.gradients_velocity = init_velocity_vector(self.params)


    def step(self) -> None:
        for idx, layer in enumerate(self.params):
            for param_name, param_value in layer.items():
                self.gradients_velocity[idx][param_name] = self.beta * self.gradients_velocity[idx][param_name] + (1 - self.beta) * param_value.grad
                direction = self.gradients_velocity[idx][param_name] * self.learning_rate
                # Running in place operation with context manager
                # because we dont need to track this operation
                with torch.no_grad():
                    layer[param_name] -= direction

class RMSprop(Optimizer):
    beta: float
    epsilon: float = 1e-7

    def __init__(self, layers_params: list[dict[str, torch.Tensor]], learning_rate: float = 1e-2, beta = 0.9):
        super().__init__(layers_params, learning_rate)
        self.beta = beta
        self.gradients_velocity = init_velocity_vector(self.params)

    
    def step(self) -> None:
        for idx, layer in enumerate(self.params):
            for param_name, param_value in layer.items():
                self.gradients_velocity[idx][param_name] = self.beta * self.gradients_velocity[idx][param_name] + (1 - self.beta) * torch.pow(param_value.grad, 2)
                direction = param_value.grad / torch.sqrt(self.gradients_velocity[idx][param_name] + self.epsilon) * self.learning_rate
                # Running in place operation with context manager
                # because we dont need to track this operation
                with torch.no_grad():
                    layer[param_name] -= direction


### Pending to fix
class Adam(Optimizer):
    """Adam is a combination of momentum and RMSprop, thats why the velocity names"""
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-7

    def __init__(self, layers_params: list[dict[str, torch.Tensor]], learning_rate = 1e-2):
        super().__init__(layers_params, learning_rate)
        self.momentum_velocity = init_velocity_vector(self.params)
        self.rmsprop_velocity = init_velocity_vector(self.params)

    def step(self) -> None:
        for idx, layer in enumerate(self.params):
            for param_name, param_value in layer.items():
                self.momentum_velocity[idx][param_name] = (self.beta1 * self.momentum_velocity[idx][param_name]) + ((1 - self.beta1) * param_value.grad)
                self.rmsprop_velocity[idx][param_name] = (self.beta2 * self.rmsprop_velocity[idx][param_name]) + ((1 - self.beta2) * torch.pow(param_value.grad, 2))
                direction = (self.momentum_velocity[idx][param_name] / torch.sqrt(self.rmsprop_velocity[idx][param_name] + self.epsilon)) * self.learning_rate
                # Running in place operation with context manager
                # because we dont need to track this operation
                with torch.no_grad():
                    layer[param_name] -= direction

