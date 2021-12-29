import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1, label="sigmoid")
plt.plot(x, y2, label="step-function", linestyle='--')
plt.ylim(-0.1, 1.1)
plt.xlim(-6, 6)
plt.show()