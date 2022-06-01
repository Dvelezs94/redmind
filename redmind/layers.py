import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

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
        self._freeze = False
        self._train = False

    def __repr__(self, extra_fields = {}):
        return str({'Type:': type(self).__name__, 
                'forward_inputs': self.forward_inputs,
                'forward_outputs': self.forward_outputs,
                'backward_inputs': self.backward_inputs,
                'backward_outputs': self.backward_outputs, 
                **extra_fields})
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradient: float) -> np.ndarray:
        pass

    def get_trainable_params(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary of all trainable parameters by the layer"""
        return {}

    def get_trainable_params_gradients(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary of all trainable parameters GRADIENTS by the layer
        This is mainly used by the optimizer
        """
        return {}

    def update_trainable_params(self, optimized_params: Dict[str, np.ndarray]) -> None:
        """Update trainable params. Order of optimized_params is the same as get_trainable_params_gradients"""
        pass

    def get_train(self):
        return self._train

    def set_train(self, state):
        assert type(state) == bool
        self._train = state
    
    def freeze(self):
        """
        Freeze layer. Useful when doing transfer learning (pending implementation)
        """
        self._freeze = True
    
    def unfreeze(self):
        self._freeze = False
    
class Dense(Layer):
    def __init__(self, n_rows: int = None, n_columns: int = None, weight_init_scale = 0.1, seed: int = None) -> None:
        if seed:
            np.random.seed(seed)
        self.weights = np.random.randn(n_rows, n_columns) * weight_init_scale
        self.bias = np.random.randn(n_rows, 1)
        self.bias_prime = np.zeros(self.bias.shape)
        self.weights_prime = np.zeros(self.weights.shape)
        super().__init__()

    def __repr__(self):
        fields = {'weights': self.weights, 'bias': self.bias, 
                'weights_prime': self.weights_prime, 'bias_prime': self.bias_prime}
        return super().__repr__(extra_fields = fields)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.forward_inputs = x
        self.forward_outputs = np.dot(self.weights, self.forward_inputs) + self.bias
        return self.forward_outputs

    def backward(self, output_gradient: float) -> np.ndarray:
        self.backward_inputs = output_gradient
        self.backward_outputs = np.dot(self.weights.T, self.backward_inputs)
        # compute gradients
        self.weights_prime = np.dot(self.backward_inputs, self.forward_inputs.T)
        self.bias_prime = np.sum(self.backward_inputs, axis=1, keepdims=True)
        return self.backward_outputs

    def get_trainable_params(self) -> Dict[str, np.ndarray]:
        return {'weights': self.weights, 'bias': self.bias}

    def get_trainable_params_gradients(self) -> Dict[str, np.ndarray]:
        return {'weights_prime': self.weights_prime, 'bias_prime': self.bias_prime}

    def update_trainable_params(self, optimized_params: Dict[str, np.ndarray]) -> None:
        if not self._freeze:
            # update weights and biases
            self.weights -= optimized_params['weights_prime']
            self.bias -= optimized_params['bias_prime']
        return None

class Dropout(Layer):
    def __init__(self, drop_prob: float = 0) -> None:
        assert 0 <= drop_prob < 1, "Drop probability rate should be between 0 and 0.9"
        self.drop_prob = drop_prob
        self.keep_prob = 1 - self.drop_prob
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.forward_inputs = x
        matrix_probs = [self.drop_prob, 1 - self.drop_prob]
        if self._train:
            matrix_probs = [self.drop_prob, 1 - self.drop_prob]
            self.drop_matrix = np.random.choice([0, 1], size=x.shape, p=matrix_probs)
            self.forward_outputs = np.multiply(self.drop_matrix, self.forward_inputs) / self.keep_prob
        else:
            self.forward_outputs = self.forward_inputs
        return self.forward_outputs

    def backward(self, output_gradient: float) -> np.ndarray:
        self.backward_inputs = output_gradient
        self.backward_outputs = np.multiply(self.drop_matrix, output_gradient) / self.keep_prob
        return self.backward_outputs

####################
# Activation Layer #
####################

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime
        super().__init__()

    def forward(self, x) -> np.ndarray:
        self.forward_inputs = x
        self.forward_outputs = self.activation(self.forward_inputs)
        return self.forward_outputs

    def backward(self, output_gradient: float) -> np.ndarray:
        self.backward_inputs = output_gradient
        self.backward_outputs = np.multiply(self.backward_inputs, self.activation_prime(self.forward_inputs))
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

# softmax is a bit different from other activations,
# its more like a fully connected since it depends on all
# the input variables to compute the sum to create the
# probability distribution which adds up to 1.
# thats why it receives a different treatment and inherits
# directoly from the Layer superclass
class Softmax(Layer):
    def forward(self, x) -> np.ndarray:
        self.forward_inputs = x
        input_stable = np.exp(x - np.max(x, axis=0))
        self.forward_outputs =  input_stable / input_stable.sum(axis=0)
        return self.forward_outputs
    
    def backward(self, output_gradient: float) -> np.ndarray:
        self.backward_inputs = output_gradient
        tmp = np.sum(self.backward_inputs * self.forward_outputs, axis = 0)
        self.backward_outputs = self.forward_outputs * (self.backward_inputs - tmp)
        return self.backward_outputs
        