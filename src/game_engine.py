from collections import defaultdict
from numbers import Number
from typing import Any, Optional

import gym
from enforce_typing import enforce_types
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


def default_mask_function(env):
    return env.valid_action_mask()


class GameEngine(object):

    @classmethod
    @enforce_types
    def train_or_load(self, agent: Any,
                      rl_algo: Any,
                      env_name: str,
                      eval_freq: Number,
                      n_eval_episodes: Number,
                      learning_rate: Number,
                      timesteps: Number,
                      save_path: str,
                      log_path: str,
                      load_path: Optional[str] = None):
        env = gym.make(env_name)
        if load_path is not None:
            print(f'Loading from {load_path}')
            return rl_algo.load(load_path)
        else:
            eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=eval_freq,
                                         n_eval_episodes=n_eval_episodes)
            env.set_opponent(agent)
            model = rl_algo('MlpPolicy', env=env, verbose=1, action_mask_fn=default_mask_function,
                            learning_rate=learning_rate)
            model.learn(total_timesteps=timesteps, eval_freq=eval_freq, callback=eval_callback)
            model.save(path=save_path)

            return model

    @classmethod
    @enforce_types
    def evaluate(self, model: Any, agent: Any, ENV_NAME: str, scores):
        done = False
        evaluate_env = gym.make(ENV_NAME)
        evaluate_env.set_opponent(agent)
        obs = evaluate_env.reset()
        ttl_rewards = 0
        count = 0
        all_rewards = defaultdict(int)
        for i in range(100):
            count += 1
            action, _states = model.predict(obs, deterministic=True, action_masks=default_mask_function(evaluate_env))
            # print(f'ACTION={action}')
            obs, reward, done, info = evaluate_env.step(action)

            ttl_rewards += reward
            if done:
                evaluate_env.render()
                if reward == scores.win:
                    print('WON!')
                elif reward == scores.loss:
                    print('LOSS!')
                elif reward == scores.illegal_move:
                    print('LOSS ILLEGAL MOVE')
                elif reward == scores.draw:
                    print('DRAW!')
                else:
                    print(f'UNKNOWN_DONE ({reward}')
                all_rewards[reward] += 1
                obs = evaluate_env.reset()
                print(f'Average reward = {ttl_rewards / count}')
                ttl_rewards = 0
                count = 0
        print(str(all_rewards[scores.win]) + ' Wins')
        print(str(all_rewards[scores.draw]) + ' Draws')
        print(str(all_rewards[scores.loss]) + ' Losses')
        print(str(all_rewards[scores.illegal_move]) + ' Illegal Moves')
        print(str(all_rewards[scores.legal_move]) + ' Legal Moves')
        mean_reward, std_reward = evaluate_policy(model, evaluate_env, n_eval_episodes=1)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
