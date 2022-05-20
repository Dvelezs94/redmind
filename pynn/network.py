from pynn.layers import Layer
import numpy as np
import pynn.functions as fn
import matplotlib.pyplot as plt
from typing import List, Dict

class NeuralNetwork:
    costs: Dict[int, float] = {}

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        print(f"Neural Network initialized with {len(self.layers)} layers")
        return None

    def details(self) -> None:
        for index, layer in enumerate(self.layers):
            print(f"Layer {index + 1} {layer}")

    def forward(self, x) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, gradient: float, learning_rate: float) -> None:
        grad = gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return None

    def predict(self, x):
        return self.forward(x).reshape(-1)

    def train(self, epochs, x, Y, learning_rate=0.01, verbose=True):
        for epoch in range(epochs):
            # forward
            y_pred = self.forward(x)
            # calculate error and cost
            cost = fn.binary_cross_entropy(Y, y_pred)
            self.costs[epoch] = cost
            error_gradient = fn.binary_cross_entropy_prime(Y, y_pred)
            # backward
            self.backward(error_gradient, learning_rate)
            # print cost to console
            if verbose:
                print(f"epoch: {epoch + 1}/{epochs}, cost: {round(self.costs[epoch], 4)}, accuracy: {round(100 - (self.costs[epoch] * 100), 2)}%")

    def graph_costs(self) -> None:
        plt.plot(list(self.costs.keys()), list(self.costs.values()))
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()