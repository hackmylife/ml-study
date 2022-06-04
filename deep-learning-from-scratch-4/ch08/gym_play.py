import numpy as np
import gym
import time

env = gym.make('CartPole-v1')
state = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, info = env.step(action)
    print('next_state: {}, reward: {}, done: {}, info: {}', next_state, reward, done, info)
    time.sleep(0.2)
env.close()
