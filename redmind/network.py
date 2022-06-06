from redmind.layers import Layer
from dataclasses import dataclass
import torch

@dataclass
class NeuralNetwork:
    layers: list[Layer]

    def __init__(self, layers: list[Layer] = [], verbose=False) -> None:
        self.layers = layers
        self._verbose = verbose
        if self._verbose:
            print(f"Neural Network initialized with {len(self.layers)} layers")

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
    
    def layers_parameters(self):
        return [p.get_params() for p in self.layers]

    def set_verbose(self, state):
        assert type(state) == bool
        self._verbose = state

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Same as forward, its just used for convention
        """
        return self.forward(x)


