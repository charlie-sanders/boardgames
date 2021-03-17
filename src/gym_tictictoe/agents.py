from numbers import Number
from typing import List
import secrets
import gym
from enforce_typing import enforce_types

from game_engine import default_mask_function, Agent


class TrainedAgent(Agent):
    def __init__(self, model):
        self.model = model

    def predict(self, obs, env) -> int:
        ret = self.model.predict(env._encode_observations(obs), deterministic=True,
                                 action_masks=default_mask_function(env))
        return ret[0]


class RandomAgent(Agent):
    @enforce_types
    def predict(self, observations: List[Number], env: gym.Env) -> int:
        return env._get_valid_cpu_action(observations)


class HumanAgent(Agent):
    @enforce_types
    def predict(self, observations: List[Number], env: gym.Env) -> int:
        env.render()
        is_done = False
        idx = 0
        valid = env.get_valid_action_indices()
        if len(valid) == 0:
            print('GAME OVER YOOU PROBABLY LOST')
            return 0

        msg = f"\n\nEnter the 0-9 index of where to place the piece: ({valid})"
        while not is_done:
            idx = int(input(msg))
            valid: List = env.get_valid_action_indices()

            try:
                valid.index(idx)
                is_done = True
            except:
                is_done = False
                msg = f"THAT({idx}) WAS INVALID({valid}), PLEASE TRY AGAIN: Enter the 0-9 index of where to place the piece: "
        return idx
