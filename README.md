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
import redmind.optimizers as optimizer
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

# Prepare optimizer
adam = optimizer.Adam(nn)
# Create trainer object
trainer = Trainer(network=nn, optimizer=adam, learning_rate=0.01)
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


## Cost functions

You can use different cost functions and even create your own, you just need to send the function as an argument to cost_function and grad_function. 

cost_function: this function is used to print the cost, it is not used to calculate any number for the layers or the neural network

grad_function: This function computes the gradients from the forward pass output

### Defining custom cost and grad functions

Cost and grad functions have the same signature however cost functions expect an output as np.float64, while gradients expect a numpy array

```python
def custom_cost(y, y_pred) -> np.float64:
    ...

def custom_grad(y, y_pred) -> np.ndarray:
    ...
```

## Optimizers

Redmind has support for using different optimizers. We include the most widely used ones, but you can also create your own very easily.

### Using a different Optimizer

The default optimizer is Gradient Descent, however you can change it.

Note: learning rate for cost function is set at training time, you just need to initialize it and pass it as argument to the NN

```python
from redmind.optimizers import Adam

nn = NeuralNetwork(layers=[
    Dense(n_weights_1, x_train.shape[0]),
    ReLU(),
    Dense(n_weights_2, n_weights_1),
    Sigmoid()
], cost_function=fn.mse, 
grad_function=fn.mse_prime, optimizer=Adam())
```

Native supported optimizers

- GradientDescent

- Momentum

- RMSprop

- Adam


### Creating your own optimizer

You can create your own optimizer and use that in the NN, you just need to inherit from the Optimizer class

```python
from redmind.optimizers import Optimizer

class CustomOptimizer(Optimizer):
    # define optional class attributes if you want to save states
    # check adam optimizer for reference
    def __call__(self) -> None:
        for layer in self.layers:
            trainable_params = layer.get_trainable_params()
            for k, v in trainable_params.items():
                # Run your computations for each layer trainable params
                ...
            # update trainable params for that layer
            layer.update_trainable_params(trainable_params)

nn = NeuralNetwork(layers=[
    Dense(n_weights_1, x_train.shape[0]),
    ReLU(),
    Dense(n_weights_2, n_weights_1),
    Sigmoid()
], cost_function=fn.mse, 
grad_function=fn.mse_prime, optimizer=CustomOptimizer())
```

## Save and Load Models

You can also save and load your trained models, this makes easy for you to
package, shit and use your models everywhere you want.

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
nn.train(X = x_test, Y = y, epochs = 100000 ,batch_size = 512, learning_rate=0.5, early_stoping=99.0)

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

## Features

- [X] Classes definition and construction
- [X] Forward propagation fully working
- [X] Backward propagation working
- [X] Train and predict fully working
- [X] Add Optimization layers
- [X] Add mini batch Gradient descent (through Dataloader)
- [ ] Add Gradient checking
- [X] Support for multiple optimizers
- [ ] Learning rate decay
- [X] Add early stoping support
- [X] Save and Load models
- [ ] Add convolutional layers
- [ ] Add native pyplot support
