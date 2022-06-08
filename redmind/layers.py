import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class Layer(ABC):
    """
    Base layer class.

    All layers must inherit from this class
    because the NN works with Layer objects
    """
    parameters: torch.Tensor
    inputs: torch.Tensor
    outputs: torch.Tensor

    def __init__(self, parameters: dict[str, torch.Tensor] = {}) -> None:
        self.inputs = None
        self.outputs = None
        self.parameters = parameters
        self._freeze = False
        self._train = False
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_params(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of all trainable parameters by the layer"""
        return self.parameters

    def get_train(self):
        return self._train

    def set_train(self, state):
        assert type(state) == bool
        self._train = state
    
    def set_freeze(self, state):
        """
        Freeze layer. Useful when doing transfer learning (pending implementation)
        """
        assert type(state) == bool
        self._freeze = state

class Dense(Layer):
    def __init__(self, in_size: int, out_size: int, weight_init_scale = 0.1, seed: int = 0) -> None:
        if seed:
            torch.random.manual_seed(seed)
        self.weights = torch.rand(in_size, out_size, dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros((in_size, 1), dtype=torch.float32, requires_grad=True)
        super().__init__({'weights': self.weights, 'bias': self.bias})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs = x
        self.outputs = torch.mm(self.weights, self.inputs) + self.bias
        return self.outputs

class Dropout(Layer):
    def __init__(self, drop_prob: float = 0) -> None:
        assert 0 <= drop_prob < 1, "Drop probability rate should be between 0 and 0.9"
        self.drop_prob = drop_prob
        self.keep_prob = 1 - self.drop_prob
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs = x
        matrix_probs = [self.drop_prob, 1 - self.drop_prob]
        if self._train:
            matrix_probs = [self.drop_prob, 1 - self.drop_prob]
            self.drop_matrix = torch.from_numpy(np.random.choice([0, 1], size=x.shape, p=matrix_probs))
            self.outputs = (self.drop_matrix * self.inputs) / self.keep_prob
        else:
            self.outputs = self.inputs
        return self.outputs

    
# class BatchNormalization(Layer):
#     def __init__(self, n_rows: int = None, n_columns: int = None, weight_init_scale = 0.1, seed: int = None) -> None:
#         if seed:
#             np.random.seed(seed)
#         self.gamma = np.random.randn(n_rows, n_columns) * weight_init_scale
#         self.beta = np.random.randn(n_rows, 1)
#         self.gamma_prime = np.zeros(self.bias.shape)
#         self.beta_prime = np.zeros(self.weights.shape)
#         super().__init__()

####################
# Activation Layer #
####################

class ActivationLayer(Layer):
    def __init__(self, activation) -> None:
        self.activation = activation
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        self.inputs = x
        self.outputs = self.activation(self.inputs)
        return self.outputs

class Sigmoid(ActivationLayer):
    def __init__(self) -> None:
        sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        super().__init__(sigmoid)

class ReLU(ActivationLayer):
    def __init__(self) -> None:
        relu = lambda x: torch.maximum(x, torch.zeros(x.shape))
        super().__init__(relu)


class Softmax(Layer):
    def forward(self, x) -> torch.Tensor:
        self.inputs = x
        input_stable = torch.exp(x - torch.max(x, axis=0))
        self.outputs = input_stable / input_stable.sum(axis=0)
        return self.outputs
        