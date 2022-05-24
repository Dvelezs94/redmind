import numpy as np

from pynn.layers import Dense, Sigmoid
from pynn.network import NeuralNetwork
import pynn.functions as fn

def main() -> None:
    # Prepare data
    xor = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    
    y = np.array([0, 1, 1, 0]).reshape(1,4)
    x_test = xor.T
    
    # Build NN object
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
    nn.train(X = x_test, Y = y, epochs = 1, learning_rate=0.5)

    # Predict
    prediction_vector = nn.predict(np.array([[0],[0]]))
    # if prediction_vector > 0.5:
    #     print(1)
    # else:
    #     print(0)

    #nn.details()
    #nn.graph_costs()
    nn.grad_check(X = x_test, Y = y)

if __name__ == "__main__":
    main()