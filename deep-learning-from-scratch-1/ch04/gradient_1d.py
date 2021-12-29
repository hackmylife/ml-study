# coding: utf-8
import os, sys
sys.path.append(os.path.pardir)

import numpy as np
import matplotlib.pylab as plt

from common.my_func import tangent_line


def function_1(x):
    return 0.01*x**2 + 0.1*x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("X")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 10)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
