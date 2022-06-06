import matplotlib.pyplot as plt
import redmind.optimizers as optim
from redmind.layers import Dense, Sigmoid, ReLU
from redmind.network import NeuralNetwork
from redmind.loss import BinaryCrossEntropyLoss
from redmind.trainer import Trainer
import torch

def main() -> None:
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
    
    # training variables
    learning_rate = 1e-1
    epochs = 100
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = optim.RMSprop(nn.layers_parameters(), learning_rate=learning_rate)
    # Initialize trainer
    trainer = Trainer(network=nn, loss_function=loss_fn,optimizer=optimizer)

    # Train
    trainer.train(X = x_test, Y = y, epochs = epochs, batch_size = 1)

    # Predict, this should print "1"
    prediction_vector = nn.predict(torch.tensor([[1.],[0.]]))
    if prediction_vector > 0.5:
        print(1)
    else:
        print(0)

    # Construct decision boundary plot
    points = []
    for x in torch.linspace(0, 1, 20):
        for y in torch.linspace(0, 1, 20):
            z = nn.predict(x=torch.tensor([[x], [y]], dtype=torch.float32))
            points.append([x, y, z[0,0]])

    points = torch.tensor(points, dtype=torch.float32)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis")
    plt.show()

if __name__ == "__main__":
    main()
