from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """
    Base layer class.

    All layers must inherit from this class
    because the NN works with Layer objects
    """

    def __init__(self) -> None:
        self.forward_inputs = None
        self.forward_outputs = None
        self.backward_inputs = None
        self.backward_outputs = None

    def __repr__(self, extra_fields = {}):
        return str({'Type:': type(self).__name__, 
                'forward_inputs': self.forward_inputs,
                'forward_outputs': self.forward_outputs,
                'backward_inputs': self.backward_inputs,
                'backward_outputs': self.backward_outputs, 
                **extra_fields})
    
    @abstractmethod
    def forward(self, x: np.ndarray = None):
        pass

    @abstractmethod
    def backward(self, output_gradient: float = None, learning_rate: float = None, **kwargs):
        pass


class Dense(Layer):
    def __init__(self, n_rows: int = None, n_columns: int = None) -> None:
        self.weights = np.random.randn(n_rows, n_columns) * 0.1
        self.bias = np.random.randn(n_rows, 1)
        self.bias_prime = None
        self.weights_prime = None
        super().__init__()

    def __repr__(self):
        fields = {'weights': self.weights, 'bias': self.bias}
        return super().__repr__(extra_fields = fields)

    def forward(self, x: np.ndarray = None) -> np.ndarray:
        self.forward_inputs = x
        self.forward_outputs = np.dot(self.weights, self.forward_inputs) + self.bias
        return self.forward_outputs

    def backward(self, output_gradient: float = None, learning_rate: float = None) -> np.ndarray:
        self.backward_inputs = output_gradient
        self.backward_outputs = np.dot(self.weights.T, self.backward_inputs)
        self.update_params(learning_rate=learning_rate)
        return self.backward_outputs

    def update_params(self, learning_rate: float = 0.1) -> None:
        # compute gradients
        self.weights_prime = np.dot(self.backward_inputs, self.forward_inputs.T)
        self.bias_prime = np.sum(self.backward_inputs, axis=1, keepdims=True)
        # update w and b
        self.weights = self.weights - (self.weights_prime * learning_rate)
        self.bias = self.bias - (self.bias_prime * learning_rate)
        return None

class Dropout(Layer):
    def __init__(self, drop_rate: float = 0) -> None:
        assert 0 <= drop_rate <= 1, "Drop rate should be between 0 and 1"
        self.drop_rate = drop_rate
        super().__init__()

    def __repr__(self):
        fields = {'weights': self.weights}
        return super().__repr__(extra_fields = fields)

    def forward(self, x: np.ndarray = None, training=True) -> np.ndarray:
        self.forward_inputs = x
        prob = [self.drop_rate, 1 - self.drop_rate]
        if not training:
            prob = [0, 1]
        drop_matrix = np.random.choice([0, 1], size=x.shape, p=prob)
        self.forward_outputs = np.multiply(drop_matrix, self.forward_inputs)
        return self.forward_outputs

    def backward(self, output_gradient: float = None, **kwargs) -> np.ndarray:
        self.backward_inputs = output_gradient
        self.backward_outputs = output_gradient
        return self.backward_outputs


###################
# Activation Layers
###################

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime
        super().__init__()

    def forward(self, x) -> np.ndarray:
        self.forward_inputs = x
        self.forward_outputs = self.activation(self.forward_inputs)
        return self.forward_outputs

    def backward(self, output_gradient: float = None, **kwargs) -> np.ndarray:
        self.backward_inputs = output_gradient
        self.backward_outputs = self.backward_inputs * self.activation_prime(self.forward_inputs)
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