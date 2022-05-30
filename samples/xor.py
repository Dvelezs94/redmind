import numpy as np
import matplotlib.pyplot as plt
import redmind.optimizers as optimizer
from redmind.layers import Dense, Sigmoid
from redmind.network import NeuralNetwork
from redmind.trainer import Trainer

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
    
    adam = optimizer.Adam(nn)
    trainer = Trainer(network=nn, optimizer=adam, learning_rate=0.01)

    # Train
    trainer.train(X = x_test, Y = y, epochs = 600, batch_size = 1)

    # Predict
    prediction_vector = nn.predict(np.array([[1],[0]]))
    if prediction_vector > 0.5:
        print(1)
    else:
        print(0)

    #trainer.graph_costs()

    # decision boundary plot
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = nn.predict(x=[[x], [y]])
            points.append([x, y, z[0,0]])

    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis")
    plt.show()

if __name__ == "__main__":
    main()