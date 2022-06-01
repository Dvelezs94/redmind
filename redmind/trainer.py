import numpy as np
import matplotlib.pyplot as plt
import redmind.functions as fn
from typing import List, Dict
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

    def grad_check(self, X: np.ndarray = None, Y: np.ndarray = None, epsilon=1e-7):
        """Performs gradient checking on the given NN"""
        # Helper functions for grad check
        def params_to_vector(params: Dict[int, np.ndarray]) -> np.ndarray:
            """Converts layers parameters to a row vector"""
            flattened_params = []
            for layer in params.values():
                for param in layer.values():
                    flattened_params.append(param.ravel())
            return np.concatenate(flattened_params)

        def vector_to_params(vector: np.ndarray, layer_params: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
            """Returns dict of params with values from row vector"""
            assert params_to_vector(layer_params).size == vector.size, "vector and layer_params sizes do not match"
            updated_params = layer_params.copy()
            for k, v in layer_params.items():
                for param_name, val in v.items():
                    # get number of elements in the parameter values
                    param_size = val.size
                    # get items from idx 0 to param_size from vector
                    new_values = vector[:param_size]
                    assert val.size == new_values.size, "values sizes do not match"
                    # set values for that param in that layer in updated_params
                    updated_params[k][param_name] = new_values.reshape(updated_params[k][param_name].shape)
                    # remove assigned values from vector
                    vector = np.delete(vector, range(param_size))
            assert vector.size == 0, "Something is wrong, vector still has values!"
            return updated_params

        def update_layer_params(layer_params: Dict[int, np.ndarray]) -> None:
            """
            Update network layer parameters with given layer_params values.
            Order of the dict should be the same as self.network.layers
            """
            for idx, layer in enumerate(self.network.layers):
                for k, v in layer_params[idx].items():
                    layer.__dict__[k] = v
        
        print("Starting gradient checking")
        # Prepare data and get random element

        # compute NN gradients through forward and backward pass
        y_pred = self.network.forward(X)
        error_gradient = self.grad_function(Y, y_pred)
        self.network.backward(gradient=error_gradient)
        ###
        # get gradients for each layer and flatten in a row vector
        ###
        gradients = {}
        for idx, layer in enumerate(self.network.layers):
            gradients[idx] = layer.get_trainable_params_gradients()
        
        flattened_gradients = params_to_vector(gradients) / Y.shape[1]
        ###
        # get original params for each layer and flatten in single row vector
        ###
        original_params = {}
        for idx, layer in enumerate(self.network.layers):
            original_params[idx] = layer.get_trainable_params()

        flattened_params = params_to_vector(original_params)
        ###
        # Compute numgrad with cost function for each param
        ###
        numgrad = np.zeros(flattened_params.shape)
        preturb = np.zeros(numgrad.shape)

        # print("====ORIGINAL====")
        # print(self.network.layers[0].get_trainable_params())
        for i in range(numgrad.size):
            preturb[i] = epsilon
            # convert params vector plus epsilon
            plus_params = vector_to_params(vector = flattened_params + preturb, layer_params = original_params)
            update_layer_params(plus_params)
            # print("====PLUS EPSILON====")
            # print(self.network.layers[0].get_trainable_params())
            y_plus = self.network.forward(X)
            loss_plus = self.cost_function(Y, y_plus)

            # convert params vector minus epsilon
            minus_params = vector_to_params(vector = flattened_params - preturb, layer_params = original_params)
            update_layer_params(minus_params)
            # print("====MINUS EPSILON====")
            # print(self.network.layers[0].get_trainable_params())
            y_minus = self.network.forward(X)
            loss_minus = self.cost_function(Y, y_minus)
            # compute the gradient
            numgrad[i] = (loss_plus - loss_minus) / (2*epsilon)
            # reset preturb
            preturb[i] = 0

        numerator = np.linalg.norm(numgrad - flattened_gradients)
        denominator = np.linalg.norm(flattened_gradients) + np.linalg.norm(numgrad) 
        difference = numerator / denominator
        if difference > 2e-7 or np.isnan(difference):
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
        print(flattened_gradients)
        print(numgrad)

        plt.plot(flattened_gradients)
        plt.plot(numgrad)
        plt.legend(["gradients", "numgrad"])
        plt.xlabel("i")
        plt.ylabel("val")
        plt.show()