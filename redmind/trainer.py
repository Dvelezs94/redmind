import numpy as np
import matplotlib.pyplot as plt
import redmind.functions as fn
from redmind.network import NeuralNetwork
from redmind.layers import Layer
from redmind.dataloader import Dataloader
from redmind.optimizers import Optimizer
from redmind.loss import Loss
import torch


class Trainer():
    """
    Trainer class simply helps by automating common training process. 
    It has predefined trainig functions so you only need to send the arguments. 
    
    Using this class is not required, however its very convenient.
    """
    def __init__(self, network: NeuralNetwork, loss_function: Loss, optimizer: Optimizer):
        assert isinstance(network, NeuralNetwork), "network should be a NeuralNetwork object"
        assert isinstance(optimizer, Optimizer), "optimizer should be an Optimizer object"
        assert isinstance(loss_function, Loss), "loss_function should be a Loss object"
        self.network = network
        self.optimizer = optimizer
        self.costs = {}
        self.loss_function = loss_function
    
    def train(self, X: torch.Tensor, Y: torch.Tensor, epochs: int = 100, batch_size: int = 1, early_stoping: float = 0.0):
        assert type(X) == torch.Tensor, "expecting X to be a torch tensor"
        assert type(Y) == torch.Tensor, "expecting Y to be a torch tensor"

        data = Dataloader(X=X, Y=Y, batch_size=batch_size)
        self.network.set_train(state=True)

        for epoch in range(epochs):
            epoch_losses = []
            for x, y in data:
                # forward
                y_pred = self.network.forward(x)
                # clear gradients
                self.optimizer.zero_grad()

                # calculate error and cost
                loss = self.loss_function(y, y_pred)
                epoch_losses.append(loss.detach())
                loss.backward()
                
                # Gradient descent step
                self.optimizer.step()

            # print cost to console
            self.costs[epoch] = torch.stack(epoch_losses).mean().item()
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