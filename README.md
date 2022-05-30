# RedMind

This is a python library made to help you build machine learning models.

Developed by Diego Velez 2022

## Installation

```shell
pip3 install redmind
```

## Quickstart (XOR sample)

```python
import numpy as np
import redmind.functions as fn
from redmind.layers import Dense, Sigmoid
from redmind.network import NeuralNetwork
from redmind.trainer import Trainer

# Prepare data
xor = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

y = np.array([0, 1, 1, 0]).reshape(1,4)
x_test = xor.T

n_weights_1 = 3 # 3 neurons in the first layer
n_weights_2 = 1 # 1 neuron in the second layer (output)
# use seeds for consistency in results
nn = NeuralNetwork(layers=[
    Dense(n_weights_1, x_test.shape[0], seed=1),
    Sigmoid(),
    Dense(n_weights_2, n_weights_1, seed=1),
    Sigmoid()
])

# Create trainer object
trainer = Trainer(network=nn, learning_rate=0.01)
# Train
trainer.train(X = x_test, Y = y, epochs = 600, batch_size = 1)

# Predict
prediction_vector = nn.predict(np.array([[1],[0]]))
if prediction_vector > 0.5:
    print(1)
else:
    print(0)
```

Go to `samples` folder for more samples

You can also opt to not use the `Trainer` class and manually train the network, here is how to do it

### Manual Train (XOR sample)

```python
import numpy as np
import matplotlib.pyplot as plt
import redmind.optimizers as optimizer
import redmind.functions as fn
from redmind.layers import Dense, Sigmoid
from redmind.network import NeuralNetwork
from redmind.dataloader import Dataloader

def main() -> None:
    # Prepare data
    xor = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    
    y = np.array([0, 1, 1, 0]).reshape(1,4)
    x_test = xor.T
    
    # Build NN
    n_weights_1 = 10 # 3 neurons in the first layer
    n_weights_2 = 1 # 1 neuron in the second layer (output)
    nn = NeuralNetwork(layers=[
        Dense(n_weights_1, x_test.shape[0], seed=1),
        Sigmoid(),
        Dense(n_weights_2, n_weights_1, seed=1),
        Sigmoid()
    ])

    # Load data in dataloader so we can loop it
    data = Dataloader(x_test, y)
    
    # training variables
    learning_rate = 1e-2
    epochs = 1000
    costs = {}

    # prepare optimizer
    adam = optimizer.Adam(nn)
    adam.set_learning_rate(learning_rate)

    # Manual train
    for epoch in range(epochs):
        for x, y in data:
            # forward
            y_pred = nn.forward(x)
            # calculate error and cost
            cost = fn.mse(y, y_pred)
            costs[epoch] = cost
            error_gradient = fn.mse_prime(y, y_pred)
            # backward
            nn.backward(gradient=error_gradient)
            # Optimize layers params
            adam()
        accuracy = round(100 - (costs[epoch] * 100), 3)
        print(f"epoch: {epoch + 1}/{epochs}, cost: {round(costs[epoch], 4)}, accuracy: {accuracy}%")

    # Predict
    prediction_vector = nn.predict(np.array([[1],[0]]))
    if prediction_vector > 0.5:
        print(1)
    else:
        print(0)

if __name__ == "__main__":
    main()
```


## Cost/Grad functions

You can use different cost functions and even create your own, you just need to send the function as an argument to the `Trainer` as `cost_function` and `grad_function`. 

cost_function: this function is used to print the cost, and early stoping in case you enable.

grad_function: This function computes the gradients from the forward pass output

### Defining custom cost and grad functions

Cost and grad functions have the same signature however the cost should output a scalar while the gradient should output a matrix

```python
def custom_cost(y, y_pred) -> np.float64:
    ...

def custom_grad(y, y_pred) -> np.ndarray:
    ...
```

## Optimizers

Redmind has support for different optimizers.

Native supported optimizers

- GradientDescent

- Momentum

- RMSprop

- Adam

### Using a different Optimizer

The default optimizer is Gradient Descent, however you can change it.

The optimizer object expects the NeuralNetwork as argument, so it can read the network layers

```python
import redmind.optimizers as optimizer
from redmind.network import NeuralNetwork

nn = NuralNetwork(...)

adam = optimizer.Adam(nn)
trainer = Trainer(network=nn, optimizer=adam, learning_rate=1e-2)
trainer.train(X = X_train, Y = Y_train, epochs = 20, batch_size = 128)
```

### Creating your own optimizer

You can create your own optimizer and use that in the `Trainer` class, you just need to inherit from the Optimizer class

```python
from redmind.optimizers import Optimizer, init_velocity_vector

class CustomOptimizer(Optimizer):
    # Optional __init__ method if you want to save states in the object 
    def __init__(self, network: NeuralNetwork):
        super().__init__(network)
        self.gradients_velocity = init_velocity_vector(self.layers)

    def __call__(self) -> None:
        for idx, layer in enumerate(self.layers):
            trainable_params = layer.get_trainable_params()
            for param, grads in trainable_params.items():
                # Run your computations for each layer trainable params
                ...
            # update trainable params for that layer
            layer.update_trainable_params(trainable_params)

nn = NeuralNetwork(...)

myCustomOpt = CustomOptimizer(nn)
trainer = Trainer(network=nn, optimizer=myCustomOpt, learning_rate=1e-2)
```

## Save and Load Models

You can also save and load your trained models, this makes easy for you to
package, shit and use your models everywhere you want.

### Save model
```python
from redmind.utils import save_model

...
nn = NeuralNetwork(...)

# Create trainer object
trainer = Trainer(network=nn, learning_rate=0.01)
# Train
trainer.train(X = x_test, Y = y, epochs = 600, batch_size = 1)

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

## Learning Rate Decay

The `Trainer` class also supports learning_rate decay.

```python
from redmind.functions import lr_decay
...
nn = NeuralNetwork(...)

# Create trainer object
trainer = Trainer(network=nn, learning_rate=0.01, lr_decay_function = lr_decay, decay_rate: 0.1)
# Train
trainer.train(X = x_test, Y = y, epochs = 600, batch_size = 1)
```

## Features

- [X] Classes definition and construction
- [X] Forward propagation fully working
- [X] Backward propagation working
- [X] Train and predict fully working
- [X] Add Optimization layers
- [X] Add mini batch Gradient descent (through Dataloader)
- [ ] Add Gradient checking
- [X] Support for multiple optimizers
- [X] Learning rate decay
- [X] Add early stoping support
- [X] Save and Load models
- [ ] Add convolutional layers
- [ ] Add native pyplot support
