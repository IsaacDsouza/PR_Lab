# Train a neuron to learn the AND pattern classification problem using Perceptron learning.

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
weights, bias, lr = np.zeros(2), 0, 0.1

for _ in range(10):
    for i in range(len(X)):
        pred = int(np.dot(X[i], weights) + bias > 0)
        error = y[i] - pred
        weights += lr * error * X[i]
        bias += lr * error

for i in X:
    print(f"Input: {i}, Prediction: {int(np.dot(i, weights) + bias > 0)}")