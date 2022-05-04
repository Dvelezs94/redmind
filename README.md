# PyNN

This is a python library made to help you build machine learning models.

## Quicksttart

```python
from pynn.layers import Dense, Sigmoid
from pynn.ann import NeuralNetwork
    
n_weights_1 = 5 # 5 neurons in the first layer
n_weights_2 = 1 # 1 neuron in the second layer (output)
nn = NeuralNetwork([
        Dense(n_weights_1, x_test.shape[0]),
        Sigmoid(),
        Dense(n_weights_2, n_weights_1),
        Sigmoid()
    ])
```

## Objectives

- [X] Classes definition and construction
- [X] Forward propagation fully working
- [] Backward propagation working
- [] Train and predict fully working
- [] Add Optimization layers
- [] Add convolutional layers
- [] Add batch support for SGD
- [] Add native pyplot support
- [] Add CUDA support??