from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """
    Base layer class.

    All layers must inherit from this class
    because the NN works with Layer objects
    """
    forward_inputs = None
    forward_outputs = None
    backward_inputs = None
    backward_outputs = None

    def __repr__(self, extra_fields = {}):
        return str({'Type:': type(self).__name__, 
                'forward_inputs': self.forward_inputs,
                'forward_outputs': self.forward_outputs,
                'backward_inputs': self.backward_inputs,
                'backward_outputs': self.backward_outputs, 
                **extra_fields})
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    weights = None
    bias = None
    bias_prime = None
    weights_prime = None

    def __init__(self, n_rows, n_columns):
        self.weights = np.random.randn(n_rows, n_columns) * 0.1
        self.bias = np.zeros((n_rows, 1))

    def __repr__(self):
        fields = {'weights': self.weights, 'bias': self.bias}
        return super().__repr__(extra_fields = fields)

    def forward(self, x):
        self.forward_inputs = x
        self.forward_outputs = np.dot(self.weights, self.forward_inputs) + self.bias
        return self.forward_outputs

    def backward(self, output_gradient, learning_rate):
        self.backward_inputs = output_gradient
        self.backward_outputs = np.dot(self.weights.T, self.backward_inputs)
        self.update_params(learning_rate)
        return self.backward_outputs

    def update_params(self, learning_rate) -> None:
        # compute w gradient
        self.weights_prime = np.dot(self.backward_inputs, self.forward_inputs.T) / self.forward_inputs.shape[1]
        # compute b gradient
        self.bias_prime = np.sum(self.backward_inputs, axis=1, keepdims=True) / self.forward_inputs.shape[1]
        # update w and b with gradient descent
        self.weights -= self.weights_prime * learning_rate
        self.bias -= self.bias_prime * learning_rate
        return None

###################
# Activation Layers
###################

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x) -> np.ndarray:
        self.forward_inputs = x
        self.forward_outputs = self.activation(self.forward_inputs)
        return self.forward_outputs

    def backward(self, output_gradient, *kwargs) -> np.ndarray:
        self.backward_inputs = output_gradient
        self.backward_outputs = self.activation_prime(self.backward_inputs)
        return self.backward_outputs

class Sigmoid(ActivationLayer):
    def __init__(self) -> None:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)

class ReLU(ActivationLayer):
    def __init__(self) -> None:
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x>0).astype(x.dtype)
        super().__init__(relu, relu_prime)