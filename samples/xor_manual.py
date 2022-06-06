import matplotlib.pyplot as plt
import redmind.optimizers as optim
from redmind.layers import Dense, Sigmoid, ReLU
from redmind.network import NeuralNetwork
from redmind.dataloader import Dataloader
from redmind.loss import BinaryCrossEntropyLoss
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

    def graph_costs() -> None:
        plt.plot(list(costs.keys()), list(costs.values()))
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()
    
    # graph_costs()

    # Predict
    prediction_vector = nn.predict(torch.tensor([[1.],[0.]]))
    if prediction_vector > 0.5:
        print(1)
    else:
        print(0)

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
