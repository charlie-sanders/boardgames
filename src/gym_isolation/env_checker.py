import sys
import stable_baselines3
from stable_baselines3 import common
from stable_baselines3.common.env_checker import check_env

from pprint import pprint
import gym

import importlib
import gym_isolation.envs as envs
COLS = 5

env = gym.make('isolation-v0')

env.reset()
# a legal move
# action=env._get_valid_player_action()
# print(f'ACTION={action}')
done = False
ctr = 0
while not done:
    ctr += 1
    obs, reward, done, info = env.step(env.get_random_valid_action(0))
    print(f'Reward: {reward}, {done}')
    if ctr % 10 == 0:
        env.render()
print(f'Reward: {reward}, {done}')
env.render()
# player = input('Push player to continue')
res = common.env_checker.check_env(env)
pprint(res)
