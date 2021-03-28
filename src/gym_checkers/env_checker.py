import sys
from os.path import dirname
sys.path.append(dirname(__file__) + '/..')

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


done = False
ctr = 0
while not done:
    ctr += 1
    action = env.get_random_valid_action(0)
    print(f'Action={action}')
    obs, reward, done, info = env.step(action)
    print(f'Reward: {reward}, {done}')
    if ctr % 10 == 0:
        env.render()
print(f'Reward: {reward}, {done}')
env.render()
# player = input('Push player to continue')
res = common.env_checker.check_env(env)
pprint(res)
