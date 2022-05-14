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

    def details(self):
        for index, layer in enumerate(self.layers):
            print(f"Layer {index +1} {layer}")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, error_gradient: float, learning_rate: float) -> None:
        for layer in reversed(self.layers):
            error_gradient = layer.backward(error_gradient, learning_rate)
        return None

    
    def train(self, epochs, x, Y, learning_rate=0.01):
        for epoch in range(epochs):
            # forward
            y_pred = self.forward(x)
            # calculate error and cost
            error = fn.mse(y_pred, Y)
            self.costs[epoch] = error
            error_gradient = fn.mse_prime(y_pred, Y)
            # backward
            self.backward(error_gradient, learning_rate)
            # print cost to console
            print(f"cost: {self.costs[epoch]}")

    def graph_costs(self):
        plt.plot(list(self.costs.keys()), list(self.costs.values()))
        plt.xlabel("Generation")
        plt.ylabel("Cost")
        plt.show()

    def predict(self, x):
        return self.forward(x)