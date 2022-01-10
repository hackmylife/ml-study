if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


model = MLP((10,3))

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
p = F.softmax_cross_entropy(y, t)
print(y)
print(p)