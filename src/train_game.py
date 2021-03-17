from collections import defaultdict
import gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from gym_isolation.agents import RandomAgent as IsoRandomAgent, HumanAgent as IsoHumanAgent, \
    TrainedAgent as IsoTrainedAgent
from gym_tictictoe.agents import RandomAgent as TTTRandomAgent, HumanAgent as TTTHumanAgent, \
    TrainedAgent as TTTTrainedAgent

from game_engine import GameEngine, default_mask_function, DefaultScores
from gym_tictictoe.envs import *

RL_ALGO = PPO
TIMESTEPS = 100_000
LEARNING_RATE = 0.003
LEARNING_STARTS = 5_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 50
ENV_NAME = 'tictictoe-v0'
LOGDIR = "train_game_log_dir"

SAVE_PATH = f'./{ENV_NAME}_save_{TIMESTEPS}.zip'
print(f'Loading and Saving to/from {SAVE_PATH}')
LOAD_PATH = SAVE_PATH
# Force training
# LOAD_PATH = None

eval_env = gym.make(ENV_NAME)

engine = GameEngine()
model = engine.train_or_load(TTTRandomAgent(),
                             RL_ALGO,
                             env_name=ENV_NAME,
                             log_path=LOGDIR,
                             eval_freq=EVAL_FREQ,
                             n_eval_episodes=EVAL_EPISODES,
                             learning_rate=LEARNING_RATE,
                             timesteps=TIMESTEPS,
                             save_path=SAVE_PATH,
                             load_path=LOAD_PATH)
engine.evaluate(model, TTTRandomAgent(), ENV_NAME=ENV_NAME, scores=DefaultScores)

# Now train it again with a new save_path to indicate training it against itself
engine.train_or_load(IsoTrainedAgent(model),
                     PPO,
                     env_name=ENV_NAME,
                     log_path=LOGDIR + '_trained_twice',
                     eval_freq=EVAL_FREQ,
                     n_eval_episodes=EVAL_EPISODES,
                     learning_rate=LEARNING_RATE * 0.01,
                     timesteps=TIMESTEPS * 2,
                     save_path=SAVE_PATH + '_trained_twice',
                     load_path=None)
engine.evaluate(model, TTTHumanAgent(), ENV_NAME=ENV_NAME, scores=DefaultScores)
