import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from pprint import pprint
import gym
import importlib
import gym_tictictoe.envs


env = gym.make('tictictoe-v0')
res = check_env(env)
pprint(res)
