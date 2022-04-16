if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.gridworld import GridWorld

env = GridWorld()
v = {}
for state in env.states():
    v[state] = np.random.randn()
env.render_v(v)