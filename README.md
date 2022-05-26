# RedMind

This is a python library made to help you build machine learning models.

Developed by Diego Velez 2022

## Quickstart

```python
import numpy as np
from redmind.layers import Dense, Sigmoid
from redmind.network import NeuralNetwork
import redmind.functions as fn

# Prepare data
xor = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

y = np.array([0, 1, 1, 0]).reshape(1,4)
x_test = xor.T

n_weights_1 = 10 # 3 neurons in the first layer
n_weights_2 = 1 # 1 neuron in the second layer (output)
nn = NeuralNetwork(layers=[
        Dense(n_weights_1, x_test.shape[0]),
        Sigmoid(),
        Dense(n_weights_2, n_weights_1),
        Sigmoid()
    ], cost_function=fn.mse, 
    grad_function=fn.mse_prime)
nn.set_verbose(True)

# Train
nn.train(X = x_test, Y = y, epochs = 1000, n_batches = 4, learning_rate=0.5)

# Predict
prediction_vector = nn.predict(np.array([[0],[0]]))
```

Go to `samples` folder for more samples

## Objectives

- [X] Classes definition and construction
- [X] Forward propagation fully working
- [X] Backward propagation working
- [X] Train and predict fully working
- [X] Add Optimization layers
- [X] Add mini batch Gradient descent (through Dataloader)
- [ ] Add Gradient checking
- [ ] Support for multiple optimizers
- [ ] Learning rate decay
- [ ] Add early stoping support
- [ ] Add convolutional layers
- [ ] Add native pyplot support
