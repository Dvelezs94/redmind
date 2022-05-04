from pynn.layers import Layer
import numpy as np
import pynn.functions as fn

class NeuralNetwork:
    layers: list[Layer]
    generations: dict[int, np.ndarray]

    def __init__(self, layers) -> None:
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

    def backward(self, Y):
        pass

    
    def train(self, epochs, x, Y, learning_rate=0.01):
        for epoch in range(epochs):
            # forward
            x = self.forward(x)
            # calculate error
            loss = fn.cross_entropy_loss(x, Y)
            print(loss)
            print(-(np.mean(loss)/np.size(loss)))
            # backward
            # update params

    def predict(self, x):
        return self.forward(x)