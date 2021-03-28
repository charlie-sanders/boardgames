from collections import defaultdict
import gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from gym_isolation.agents import RandomAgent as IsoRandomAgent, HumanAgent as IsoHumanAgent, \
    TrainedAgent as IsoTrainedAgent
from gym_tictictoe.agents import RandomAgent as TTTRandomAgent, HumanAgent as TTTHumanAgent, \
    TrainedAgent as TTTTrainedAgent

from game_engine import GameEngine, default_mask_function, DefaultScores
from gym_tictictoe.envs import TicTicToe

ISO_ENV = 'isolation-v0'
TTT_ENV = 'tictictoe-v0'

# # # # # # #
#
# SET THE ENV NAME HERE
# and any other variables
#
# # # # # # #

ENV_NAME = TTT_ENV 

RL_ALGO = PPO
TIMESTEPS = 100_000
LEARNING_RATE = 0.003
LEARNING_STARTS = 5_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 50
LOGDIR = "train_game_log_dir"

SAVE_PATH = f'./{ENV_NAME}_save_{TIMESTEPS}'
print(f'Loading and Saving to/from {SAVE_PATH}')
# LOAD_PATH = SAVE_PATH
# Force training
LOAD_PATH = None

games = {
    ISO_ENV: {
        'human': IsoHumanAgent,
        'random': IsoRandomAgent,
        'trained': IsoTrainedAgent
    },
    TTT_ENV: {
        'human': TTTHumanAgent,
        'random': TTTRandomAgent,
        'trained': TTTTrainedAgent
    }
}

eval_env = gym.make(ENV_NAME)
engine = GameEngine()

randomAgent = games[ENV_NAME]['random']
humanAgent = games[ENV_NAME]['human']
trainedAgent = games[ENV_NAME]['trained']

model = engine.train_or_load(randomAgent(),
                             RL_ALGO,
                             env_name=ENV_NAME,
                             log_path=LOGDIR,
                             eval_freq=EVAL_FREQ,
                             n_eval_episodes=EVAL_EPISODES,
                             learning_rate=LEARNING_RATE,
                             timesteps=TIMESTEPS,
                             save_path=SAVE_PATH,
                             load_path=LOAD_PATH)
engine.evaluate(model, randomAgent(), ENV_NAME=ENV_NAME, scores=DefaultScores)

# After training the first time , start training it against itself
# new_model = model
# for i in range(0, 5):
#     print(f'Training {i}th iteration')
#     new_model = engine.train_or_load(trainedAgent(new_model),
#                                      PPO,
#                                      env_name=ENV_NAME,
#                                      log_path=LOGDIR,
#                                      eval_freq=EVAL_FREQ,
#                                      n_eval_episodes=EVAL_EPISODES,
#                                      learning_rate=LEARNING_RATE,
#                                      timesteps=TIMESTEPS,
#                                      save_path=SAVE_PATH,
#                                      load_path=None)

engine.evaluate(model, randomAgent(), ENV_NAME=ENV_NAME, scores=DefaultScores)
engine.evaluate(model, humanAgent(), ENV_NAME=ENV_NAME, scores=DefaultScores)
