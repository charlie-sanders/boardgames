import sys
from os.path import dirname
sys.path.append(dirname(__file__) + '/..')

from gym_checkers.agents import RandomAgent
import gym_checkers.envs as envs
import sys
import stable_baselines3
from stable_baselines3 import common
from stable_baselines3.common.env_checker import check_env

from pprint import pprint
import gym

import importlib

env = gym.make('checkers-v0')
board = env.generate_starting_board()
env.render()
env.set_opponent(RandomAgent())

done = False
ctr = 0
while not done:
    ctr += 1
    action = env.get_random_valid_action(1)
    print(f'Action={action}')
    obs, reward, done, info = env.step(action)
    env.render()
    input('Hit enter to continue')
    print('Is done = ' + str(done))
print(f'Reward: {reward}, {done}')
env.render()
# player = input('Push player to continue')

