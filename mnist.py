"""
MNIST implementation with pynn
"""
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from pynn.layers import Dense, Dropout, Sigmoid, ReLU
from pynn.network import NeuralNetwork
from pynn.utils import one_hot_encode
import pynn.functions as fn
from pynn.dataloader import Dataloader

def fetch_mnist_data():
    # training set
    traininig_set_batch=datasets.MNIST('datasets/',train=True, download=True)
    X_train_unflattened = traininig_set_batch.data.numpy()
    X_train = np.array([image.ravel()/255 for image in X_train_unflattened])
    Y_train = np.array([one_hot_encode(i, 10) for i in traininig_set_batch.targets.numpy()])

    # test set
    test_set_batch=datasets.MNIST('datasets/',train=False, download=True)
    X_test_unflattened = test_set_batch.data.numpy()
    X_test = np.array([image.ravel()/255 for image in X_test_unflattened])
    Y_test = np.array([one_hot_encode(i, 10) for i in test_set_batch.targets.numpy()])

    return X_train, Y_train, X_test, Y_test

def plot_image(x, title) -> None:
    image = x
    #fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def main() -> None:
    X_train, Y_train, X_test, Y_test = fetch_mnist_data()
    # transpose all to column vectors
    X_train, Y_train, X_test, Y_test = X_train.T, Y_train.T, X_test.T, Y_test.T

    n_neurons_l1 = 100
    n_neurons_l2 = 100
    n_neurons_l3 = 10 # 10 output classes

    nn = NeuralNetwork(layers=[
            Dense(n_neurons_l1, 784, weight_init_scale=np.sqrt(2/784)),
            ReLU(),
            #Dropout(0.1),
            Dense(n_neurons_l2, n_neurons_l1, weight_init_scale=np.sqrt(2/n_neurons_l1)),
            ReLU(),
            Dense(n_neurons_l3, n_neurons_l2, weight_init_scale=np.sqrt(2/n_neurons_l2)),
            Sigmoid()
        ], cost_function=fn.binary_cross_entropy, grad_function=fn.binary_cross_entropy_prime)
    nn.set_verbose(state=True)

    nn.train(X = X_train, Y = Y_train, epochs = 20, n_batches = 100, learning_rate=0.3)
    #nn.graph_costs()

    # Run test set predictions
    predictions = nn.predict(x=X_train)
    train_cost = fn.binary_cross_entropy(Y_train, predictions)
    print(f"Test set cost: {train_cost}, accuracy: {round(100 - (train_cost * 100), 4)}%")
    
    # # predict a random image
    test_data = Dataloader(X_test, Y_test, 1)
    rand_x, rand_y = test_data.get_random_element()
    prediction = nn.predict(rand_x.reshape(784,1))
    plot_image(rand_x.reshape(28,28), f"real: {rand_y.argmax()} / predicted: {prediction.argmax()}")
    

if __name__ == "__main__":
    main()