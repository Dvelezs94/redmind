import numpy as np
import matplotlib.pyplot as plt
import redmind.functions as fn
from redmind.network import NeuralNetwork
from redmind.layers import Layer
from redmind.dataloader import Dataloader
from redmind.optimizers import Optimizer
from redmind.optimizers import GradientDescent


class Trainer():
    """
    Trainer class makes training easier. It has predefined trainig functions so you
    only need to send the arguments. You can also train manually if you want.
    """
    def __init__(self, network: NeuralNetwork, learning_rate: float = 1e-2, lr_decay_function = None, decay_rate: float = None, 
                cost_function = fn.mse, grad_function = fn.mse_prime, optimizer: Optimizer = None):
        assert isinstance(network, NeuralNetwork), "network should be a NeuralNetwork object"
        self.network = network
        if not optimizer:
            optimizer = GradientDescent(self.network)
        assert isinstance(optimizer, Optimizer), "optimizer should be an Optimizer object"
        self.learning_rate = learning_rate
        self.lr_decay_function = lr_decay_function
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.costs = {}
        self.cost_function = cost_function
        self.grad_function = grad_function
    
    def train(self, X: np.ndarray = None, Y: np.ndarray = None, epochs: int = 100, batch_size: int = 1, early_stoping: float = 0.0):
        data = Dataloader(X=X, Y=Y, batch_size=batch_size)
        self.network.set_train(state=True)
        learning_rate = self.learning_rate
        for epoch in range(epochs):
            # learning rate decay
            if self.lr_decay_function:
                learning_rate = self.lr_decay_function(self.learning_rate, epoch, self.decay_rate)

            # Train network
            self.optimizer.set_learning_rate(learning_rate)
            for x, y in data:
                # forward
                y_pred = self.network.forward(x)
                # calculate error and cost
                cost = self.cost_function(y, y_pred)
                self.costs[epoch] = cost
                error_gradient = self.grad_function(y, y_pred)
                # backward
                self.network.backward(gradient=error_gradient)
                # Optimize layers params
                self.optimizer()

            # print cost to console
            accuracy = round(100 - (self.costs[epoch] * 100), 3)
            print(f"epoch: {epoch + 1}/{epochs}, cost: {round(self.costs[epoch], 4)}, accuracy: {accuracy}%")

            # Early stoping
            if early_stoping > 0.0 and accuracy >= early_stoping:
                print(f"Early stoping threshold reached (over {early_stoping}%)")
                break
        self.network.set_train(state=False)

    def graph_costs(self) -> None:
        plt.plot(list(self.costs.keys()), list(self.costs.values()))
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()