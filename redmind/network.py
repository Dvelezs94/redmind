import numpy as np
from typing import List
from redmind.layers import Layer

class NeuralNetwork:
    def __init__(self, layers: List[Layer], verbose=False) -> None:
        self.layers = layers
        self._verbose = verbose
        if self._verbose:
            print(f"Neural Network initialized with {len(self.layers)} layers")

    def details(self) -> None:
        for index, layer in enumerate(self.layers):
            print(f"Layer {index + 1} {layer}")

    def forward(self, x: np.ndarray = None) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, gradient: float = None) -> None:
        grad = gradient
        for layer in reversed(self.layers):
            grad = layer.backward(output_gradient=grad)
        return None

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Same as forward, its just used for convention
        """
        return self.forward(x)

    def set_train(self, state=False):
        """
        Set network train state
        This is useful if you are using dropout layers. If so make sure properly
        changing the training states for better accuracy
        """
        if self._verbose:
            print(f"updating NN layers training to: {state}")
        for layer in self.layers:
            layer.set_train(state=state)

    def set_verbose(self, state):
        assert type(state) == bool
        self._verbose = state


