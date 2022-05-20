"""
MNIST implementation with pynn
"""
from torchvision import datasets
import matplotlib.pyplot as plt
import random
import numpy as np
from pynn.layers import Dense, Sigmoid, ReLU
from pynn.network import NeuralNetwork
from pynn.dataloader import Dataloader
from pynn.utils import one_hot_encode

def prepare_mnist_data():
    # training set
    traininig_set_batch=datasets.MNIST('datasets/',train=True, download=True)
    X_train_unflattened = traininig_set_batch.data.numpy()
    X_train = np.array([image.ravel()/255 for image in X_train_unflattened])
    Y_train = np.array([one_hot_encode(i, 10) for i in traininig_set_batch.targets.numpy()])

    # test set
    test_set_batch=datasets.MNIST('datasets/',train=False, download=True)
    X_test_unflattened = test_set_batch.data.numpy()
    X_test = [image.ravel()/255 for image in X_test_unflattened]
    Y_test = np.array([one_hot_encode(i, 10) for i in test_set_batch.targets.numpy()])

    return X_train_unflattened, X_train, Y_train, X_test_unflattened, X_test, Y_test

def plot_image(x, y) -> None:
    image = x
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.title(f"Number: {y}")
    plt.show()


def main() -> None:

    X_train_unflattened, X_train, Y_train, X_test_unflattened, X_test, Y_test = prepare_mnist_data()
    # Generate Dataloader object with X and Y as column vectors
    data = Dataloader(X_train.T, Y_train.T, 1)
    
    n_neurons_l1 = 100
    n_neurons_l2 = 100
    n_neurons_l3 = 10 # 10 output classes

    nn = NeuralNetwork([
            Dense(n_neurons_l1, 784),
            ReLU(),
            Dense(n_neurons_l2, n_neurons_l1),
            ReLU(),
            Dense(n_neurons_l3, n_neurons_l2),
            Sigmoid()
        ])

    # batch training
    for x, y in data:
       nn.train(epochs = 100, x = x, Y = y, learning_rate=0.5)
    # nn.graph_costs()

    # predict random number
    rand_x, rand_y = data.get_random_element()
    print(nn.predict(rand_x.reshape(784,1)))
    plot_image(rand_x.reshape(28,28), rand_y)

if __name__ == "__main__":
    main()