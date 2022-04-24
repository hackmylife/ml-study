if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def greedy_probs(Q, state, epsilon=0, actions_size=4):
    qs = [Q[(state, action)] for action in range(actions_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / actions_size
    action_probs = {action: base_prob for action in range(actions_size)}
    action_probs[max_action] += (1 - epsilon)
    return action_probs
