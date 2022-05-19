"""
Sample neural network with no classes. Useful for testing and debuging
"""
import numpy as np

np.random.seed(1)

X = np.array([[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
#X = np.array([[0, 0]])
Y = np.array([[0, 1, 1, 0]])
w1 = np.random.randn(3, 2) * 0.1
b1 = np.zeros((3,1))
w2 = np.random.randn(1, 3) * 0.1
b2 = np.zeros((1,1))
cache = {}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward(X, w1, w2, b1, b2):
    Z1 = np.dot(w1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)
    return {'x': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

def backward(X, Z1, A1, Z2, A2, error_gradient, w1, w2, b1, b2, learning_rate=0.1):
    # LAYER 2
    dA2 = np.multiply(error_gradient, sigmoid_prime(Z2))
    dZ2 = np.dot(w2.T, dA2)
    # update w and b for layer 2
    dw2 = np.dot(dA2, A1.T)
    db2 = dA2
    w2 -= dw2 * learning_rate
    b2 -= db2 * learning_rate

    # LAYER 1     
    dA1 = np.multiply(dZ2, sigmoid_prime(Z1))
    dZ1 = np.dot(w1.T, dA1)
    # update w and b for layer 1
    dw1 = np.dot(dA1, X.T)
    db1 = dA1
    w1 -= dw1 * learning_rate
    #print(db1 * learning_rate)
    b1 -= db1 * learning_rate
    
    return {'x': X, 'dZ1': dZ1, 'dA1': dA1, 'dZ2': dZ2, 'dA2': dA2,
            'w2': w2, 'b2': b2, 'w1': w1, 'b1': b1, 'dw2': dw2, 'db2': db2, 'dw1': dw1, 'db1': db1}

def calculate_cost(y, y_guess):
    cost = np.power(y - y_guess, 2)
    return np.squeeze(cost)

def mse_prime(y, y_pred):
    return 2 * (y_pred - y)

def predict(X, w1, w2, b1, b2):
    return forward(X, w1, w2, b1, b2)

def train(X, Y, w1, w2, b1, b2, epochs=100, learning_rate=0.01):
    for epoch in range(epochs):
        cost = 0
        for i, val in enumerate(X):
            x = val.reshape(2,1)
            out = predict(x, w1, w2, b1, b2)
            y_guess = out["A2"]
            cost += calculate_cost(Y[0][i], y_guess)
            error_gradient = mse_prime(Y[0][i], y_guess)
            back = backward(x, out["Z1"], out["A1"], out["Z2"], out["A2"], error_gradient, w1, w2, b1, b2)
            # update params
            w1 = back["w1"]
            b1 = back["b1"]
            w2 = back["w2"]
            b2 = back["b2"]
        print(f"epoch: {epoch + 1}/{epochs}, cost: {cost/X.shape[0]}")
        
train(X, Y, w1, w2, b1, b2, epochs=10000, learning_rate=0.1)