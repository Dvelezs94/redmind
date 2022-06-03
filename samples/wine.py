import pandas as pd
import numpy as np
import redmind.functions as fn
from redmind.utils import split_dataframe
from redmind.network import NeuralNetwork
from redmind.layers import Dense, Softmax, ReLU
import redmind.optimizers as optim
from redmind.trainer import Trainer
from redmind.dataloader import Dataloader
from redmind.normalizer import Normalizer

# You will need to manually download the dataset
# Dataset URL https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

# set correct path for the dataset
df = pd.read_csv("./datasets/wine/WineQT.csv")
# Remove last column which is not important at all (id)
df = df.iloc[:,:-1]

num_classes = 9
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_dataframe(df, y_col_idx = -1, train_percent=70, y_one_hot_encode=True, num_classes=9)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = X_train.T, Y_train.T, X_dev.T, Y_dev.T, X_test.T, Y_test.T

# Normalize X's
norm = Normalizer()
norm.fit(X_train, axis=1)
X_train = norm.scale(X_train)
X_dev = norm.scale(X_dev)
X_test = norm.scale(X_test)

l1_neurons = 10
l2_neurons = 10
l3_neurons = num_classes
# Create NN
nn = NeuralNetwork(layers = [
    Dense(l1_neurons, X_train.shape[0], weight_init_scale=np.sqrt(2/l1_neurons)),
    ReLU(),
    Dense(l2_neurons, l1_neurons, weight_init_scale=np.sqrt(2/l2_neurons)),
    ReLU(),
    Dense(l3_neurons, l2_neurons, weight_init_scale=np.sqrt(2/l3_neurons)),
    Softmax()
])

optimizer=optim.RMSprop(nn)
trainer = Trainer(network=nn, learning_rate=1e-1, cost_function=fn.mse, grad_function=fn.mse_prime, optimizer=optimizer)
trainer.train(X=X_train, Y=Y_train, epochs=1000, batch_size=64)

#trainer.graph_costs()

y_pred = nn.forward(X_dev)
cost = fn.mse(Y_dev, y_pred)
print("Dev Set accuracy: ", round(100 - (cost * 100), 3) )

y_pred = nn.forward(X_test)
cost = fn.mse(Y_test, y_pred)
print("Test Set accuracy: ", round(100 - (cost * 100), 3) )

test_data = Dataloader(X_test, Y_test)
rand_x, rand_y = test_data.get_random_element()
prediction = nn.predict(rand_x)
print("predicted: " , np.argmax(prediction), ". Real: ", np.argmax(rand_y))