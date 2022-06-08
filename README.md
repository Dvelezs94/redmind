# RedMind

This is a python library made to help you build machine learning models.

Developed by Diego Velez 2022

```text
There are some known issues with softmax and adam optimizer
```

## Installation

```shell
pip3 install redmind
```

## Quickstart (XOR sample)

```python
import matplotlib.pyplot as plt
import redmind.optimizers as optim
from redmind.layers import Dense, Sigmoid, ReLU
from redmind.network import NeuralNetwork
from redmind.loss import BinaryCrossEntropyLoss
from redmind.trainer import Trainer
import torch

# Prepare data
xor = torch.tensor([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]], dtype=torch.float32)
    
y = torch.tensor([0, 1, 1, 0], dtype=torch.float32).reshape(1,4)
# xor = torch.tensor([[0,1]], dtype=torch.float32)
# y = torch.tensor([1], dtype=torch.float32).reshape(1,1)
x_test = xor.T

# Build NN
n_weights_1 = 3 # 3 neurons in the first layer
n_weights_2 = 1 # 1 neuron in the second layer (output)
nn = NeuralNetwork(layers=[
    Dense(n_weights_1, x_test.shape[0], seed=1),
    ReLU(),
    Dense(n_weights_2, n_weights_1, seed=1),
    Sigmoid()
])

learning_rate = 1e-1
epochs = 100
loss_fn = BinaryCrossEntropyLoss()
optimizer = optim.RMSprop(nn.layers_parameters(), learning_rate=learning_rate)
# Initialize trainer
trainer = Trainer(network=nn, loss_function=loss_fn,optimizer=optimizer)

# Train
trainer.train(X = x_test, Y = y, epochs = epochs, batch_size = 1)

# Predict
prediction_vector = nn.predict(torch.tensor([[1.],[0.]]))
if prediction_vector > 0.5:
    print(1)
else:
    print(0)
```

Go to `samples` folder for more samples

You can also opt to not use the `Trainer` class and manually train the network, here is how to do it

### Manual Train (XOR sample)

```python
import matplotlib.pyplot as plt
import redmind.optimizers as optim
from redmind.layers import Dense, Sigmoid, ReLU
from redmind.network import NeuralNetwork
from redmind.dataloader import Dataloader
from redmind.loss import BinaryCrossEntropyLoss
import torch

# Prepare data
xor = torch.tensor([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]], dtype=torch.float32)

y = torch.tensor([0, 1, 1, 0], dtype=torch.float32).reshape(1,4)
# xor = torch.tensor([[0,1]], dtype=torch.float32)
# y = torch.tensor([1], dtype=torch.float32).reshape(1,1)
x_test = xor.T

# Build NN
n_weights_1 = 3 # 3 neurons in the first layer
n_weights_2 = 1 # 1 neuron in the second layer (output)
nn = NeuralNetwork(layers=[
    Dense(n_weights_1, x_test.shape[0], seed=1),
    ReLU(),
    Dense(n_weights_2, n_weights_1, seed=1),
    Sigmoid()
])

# Load data in dataloader so we can loop it
data = Dataloader(x_test, y, batch_size=2)

# training variables
learning_rate = 1e-1
epochs = 600
costs = {}
loss_fn = BinaryCrossEntropyLoss()
optimizer = optim.GradientDescent(nn.layers_parameters(), learning_rate=learning_rate)

# Manual train
for epoch in range(epochs):
    epoch_losses = []
    for x, y in data:
        # forward
        y_pred = nn.forward(x)

        # clear gradients
        optimizer.zero_grad()

        # calculate loss
        loss = loss_fn(y, y_pred)
        epoch_losses.append(loss.detach())
        loss.backward()

        # Gradient descent step
        optimizer.step()

    # Calculate total run cost
    costs[epoch] = torch.stack(epoch_losses).mean().item()
    accuracy = round(100 - (costs[epoch] * 100), 3)
    print(f"epoch: {epoch + 1}/{epochs}, cost: {round(costs[epoch], 4)}, accuracy: {accuracy}%")
```


## Loss functions

You can use different loss functions and even create your own, you just need to send the function as an argument to the `Trainer` as `loss_function`

```python
learning_rate = 1e-1
loss_fn = CategoricalCrossEntropyLoss()
optimizer = optim.RMSprop(nn.layers_parameters(), learning_rate=learning_rate)
# Initialize trainer
trainer = Trainer(network=nn, loss_function=loss_fn,optimizer=optimizer)
```



### Defining custom loss function

If you want to create your own loss function, you will need to inherit from the base `Loss` superclass and implement the `__call__` method

```python
from redmind.loss import Loss

class CustomLoss(Loss):
    def __call__(self, y_pred, y):
        ...
custom_loss = CustomLoss()
...

trainer = Trainer(network=nn, loss_function=custom_loss,optimizer=optimizer)

```

## Optimizers

Redmind has support for different optimizers.

Native supported optimizers

- GradientDescent

- Momentum

- RMSprop

- Adam (pending fix)

### Using a different Optimizer

The default optimizer is Gradient Descent, however you can change it.

The optimizer object expects the NeuralNetwork as argument, so it can read the network layers

```python
import redmind.optimizers as optim
...
nn = NuralNetwork(...)

adam = optimizer.Adam(nn.layers_parameters(), learning_rate=1e-2)
trainer = Trainer(network=nn, loss_function=loss_fn,optimizer=optimizer)
```

### Creating your own optimizer

You can create your own optimizer, you just need to inherit from the Optimizer class

```python
from redmind.optimizers import Optimizer
...

class CustomOptimizer(Optimizer):
    def __call__(self) -> None:
        for layer in self.params:
            for param_name, param_value in layer.items():
                direction = ... # your learning algorithm

                # make sure the in place operation runs with no_grad
                with torch.no_grad():
                    layer[param_name] -= direction

nn = NeuralNetwork(...)
myCustomOpt = CustomOptimizer(nn.layers_parameters(),  learning_rate=1e-2)
trainer = Trainer(network=nn, loss_function=loss_fn,optimizer=myCustomOpt)

```

## Save and Load Models

You can also save and load your trained models, this makes easy for you to
package, ship and use your models everywhere you want.

### Save model
```python
from redmind.utils import save_model

...
nn = NeuralNetwork(...)

# Create trainer object
trainer = Trainer(network=nn, loss_function=loss_fn,optimizer=optimizer)

# Train
trainer.train(X = x_test, Y = y, epochs = epochs, batch_size = 64)

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

## Preprocessing

Redmind also has a few handy preprocessing tools. These are tools to make your life a bit easier to handle data

### Normalizer (pending fix)

In case the features of your data have very high variance or are scaled in different way, normalizing makes it
fit between 0 and 1 (mostly). This is very useful to make your model train faster and avoid exploding gradients

Usage:

```python
import numpy as np
from redmind.normalizer import Normalizer

# column 1 Age
# Column 2 Weight
xtrain = np.array([[10, 40],
              [11, 35],
              [12, 40],
              [13, 41],
              [13, 70],
              [15, 60],
              [19, 64],
              [15, 60],
              [20, 80],
              [40, 100],
              [56, 85]])

# Initialize normalizer and fit the data
norm = Normalizer()
norm.fit(xtrain)

# scale xtrain
xtrain = norm.scale(xtrain)

xtest = np.array([[20, 60],
                [21, 75],
                [22, 80],
                [23, 59],
                [23, 85],
                [25, 77]])

# no need to refit the normalizer for new data
# You need to use the same scale 
xnorm = norm.scale(xtest)
```

### Dataloader

The dataloader is a useful tool to loop through the trainig examples and its labels.
It can also split your data in mini-batches very easily.

`Note: Make sure your data is entered as column vectors`

```python
from redmind.dataloader import Dataloader

xor = torch.tensor([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

y_train = torch.tensor([0, 1, 1, 0]).reshape(1,4)
# we need to input data as column vectors to dataloader
x_train = xor.T

data = Dataloader(X=X, Y=Y, batch_size=2)

# then we can loop over the mini-batches
# you can do forward and backpropagation like this
for x, y in data:
    print(x)
    print(y)
    #forward..
    ...
```


## Features

- [X] Classes definition and construction
- [X] Forward propagation fully working
- [X] Backward propagation working
- [X] Train and predict fully working
- [X] Add Optimization layers
- [X] Add mini batch Gradient descent (through Dataloader)
- [X] Support for multiple optimizers
- [X] Learning rate decay
- [X] Add early stoping support
- [X] Save and Load models
- [ ] Batch normalization
- [ ] Add convolutional layers
