from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
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
    def backward(self, output_gradient):
        pass

    @abstractmethod
    def update_params(self, learning_rate):
        pass


class Dense(Layer):
    weights = None
    bias = None
    bias_prime = None
    weights_prime = None

    def __init__(self, n_rows, n_columns):
        self.weights = np.random.randn(n_rows, n_columns) * 0.01
        self.bias = np.zeros((n_rows, 1))

    def __repr__(self):
        fields = {'bias': self.bias}
        return super().__repr__(extra_fields = fields)

    def forward(self, x):
        self.forward_inputs = x
        self.forward_outputs = np.dot(self.weights, self.forward_inputs) + self.bias
        return self.forward_outputs

    def backward(self, output_gradient):
        pass

    def update_params(self, learning_rate):
        pass

###################
# Activation Layers
###################

class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x):
        self.forward_inputs = x
        self.forward_outputs = self.activation(self.forward_inputs)
        #print(self.forward_outputs)
        return self.forward_outputs

    def backward(self, output_gradient):
        pass

    def update_params(self, learning_rate):
        pass

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