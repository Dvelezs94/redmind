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

## Save and Load Models

Redmind support saving and loading models as well, here is how to do it

### Save model
```python
from redmind.utils import save_model

# build a descent size model
n_weights_1 = 300 
n_weights_2 = 750
nn = NeuralNetwork(layers=[
        Dense(n_weights_1, x_train.shape[0]),
        Sigmoid(),
        Dense(n_weights_2, n_weights_1),
        Sigmoid()
    ], cost_function=fn.binary_cross_entropy, 
    grad_function=fn.binary_cross_entropy_prime)

# train
nn.train(X = x_test, Y = y, epochs = 100000 ,n_batches = 500, learning_rate=0.5, early_stoping=99.0)

# predict
nn.predict(x_test)

# Save NN model
save_model(nn, filename='bigNN.dill')
```
### Load model

```python
from redmind.utils import load_model

# Load pretrained model
nn = load_model(filename='bigNN.dill')

# predict
nn.predict(x_test)
```


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
- [X] Add early stoping support
- [X] Save and Load models
- [ ] Add convolutional layers
- [ ] Add native pyplot support
