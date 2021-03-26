import random
from dataclasses import dataclass
from datetime import datetime

from numbers import Number

from typing import Dict, Tuple, List

import gym
import numpy as np
from enforce_typing import enforce_types
from gym import spaces

from game_engine import Agent, DefaultScores
import secrets
from .source import MairlySource

N_ACTIONS = 3
VERBOSE = False


@dataclass
class Action():
    BUY: int = 1
    SELL: int = 2
    HOLD: int = 0


@dataclass
class Scores():
    LEGAL_MOVE: int = 1
    ILLEGAL_MOVE: int = -10
    PROFIT_TO_SCORE_MODIFIER: int = 3


class SingleStockEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def render(self, mode='human') -> None:
        print('Coming soon')

    metadata = {'render.modes': ['human']}

    @enforce_types
    def __init__(self, ticker: str, start: datetime, end: datetime):
        super(SingleStockEnv, self).__init__()
        # Our actions are 0: Hold, 1: Buy, 2: Sell
        self.ticker = ticker
        self.start = start
        self.end = end
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.number_of_shares = 0
        self.buy_price = 0
        # Our observations are a single row in the trade api namely
        # 'number_of_shares','exchange', 'price', 'size', 'conditions'
        self.observation_space = spaces.Box(0, 2, shape=(9,), dtype=np.int32)
        self.source = MairlySource(self.ticker, self.start, self.end)  # source here will give us our 'board'

    def reset(self) -> None:
        pass

    # Action here is 0: Hold, 1: Buy, 2: Sell
    @enforce_types
    def step(self, action) -> Tuple[np.array, float, bool, Dict]:
        obs = self.source.step()
        if self.is_valid_move(action):
            if action == Action.BUY:
                self.number_of_shares = 100
                self.buy_price = obs[2]
            elif action == Action.SELL:
                if self.number_of_shares <= 0:
                    raise Exception('Trying to sell with no shares!')
                sell_price = obs[2]
                reward = (sell_price - self.buy_price) * Scores.PROFIT_TO_SCORE_MODIFIER
                self.number_of_shares = 0
                self.buy_price = 0
                return obs, reward, False, {}
            elif action == Action.HOLD:
                print(f'Holding {self.number_of_shares}')
        else:
            raise Exception(f'Invalid move {action}')

    @enforce_types
    def is_valid_move(self, action) -> bool:
        if action == Action.HOLD:
            return True
        if action == Action.BUY:
            if self.number_of_shares <= 0:
                return True
            else:
                return False
        if action == Action.SELL:
            if self.number_of_shares > 0:
                return True
            else:
                return False
        else:
            print(f'Unknown move {action}')

    @enforce_types
    def valid_actions(self) -> List[bool]:
        return [self.is_valid_move(move) for move in range(0, 2)]

    @enforce_types
    def get_random_valid_action(self) -> Number:
        ret = secrets.choice([0, 1, 2])
        while not self.is_valid_move(ret):
            ret = secrets.choice([0, 1, 2])
        return ret
