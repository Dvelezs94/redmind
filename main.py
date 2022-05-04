import numpy as np

from pynn.layers import Dense, Sigmoid, ReLU
from pynn.ann import NeuralNetwork

def main() -> None:

    xor = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    
    y = np.array([0, 1, 1, 0])
    
    x_test = xor.T
    
    n_weights_1 = 5 # 5 neurons in the first layer
    n_weights_2 = 1 # 1 neuron in the second layer (output)
    nn = NeuralNetwork([
            Dense(n_weights_1, x_test.shape[0]),
            Sigmoid(),
            Dense(n_weights_2, n_weights_1),
            Sigmoid()
        ])
        
    nn.train(epochs = 1, x = x_test, Y = y)
    # nn.details()

if __name__ == "__main__":
    main()