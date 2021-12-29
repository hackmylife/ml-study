import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_dunction(x):
    return x


W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])


X = np.array([1.0, 0.5])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)


A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print(Z2)


A3 = np.dot(Z2, W3) + B3
Z3 = identity_dunction(A3)

print(Z3)