# RedMind

This is a python library made to help you build machine learning models.

Developed by Diego Velez 2022

## Quickstart (XOR example)

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

n_weights_1 = 3 # 3 neurons in the first layer
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
nn.train(X = x_test, Y = y, epochs = 1000, batch_size = 4, learning_rate=0.5)

# Predict
prediction_vector = nn.predict(np.array([[0],[0]]))
if prediction_vector > 0.5:
    print(1)
else:
    print(0)
```

Go to `samples` folder for more samples

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
