import random

import gym
import numpy as np
from enforce_typing import enforce_types
from gym import spaces

from game_engine import Agent, DefaultScores

ROWS = 3
COLS = 3
N_ACTIONS = ROWS * COLS
VERBOSE = False


class TicTicToe(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TicTicToe, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # Tuple(player_number,board_index_to_move_at)
        # action here is the index to where to put the symbol
        # Example for using image as input:
        self.opponent_agent = None
        self.board = [None] * N_ACTIONS  # MAX LENGTH DIMS
        self._cpu_player = self._get_valid_cpu_action
        self.reset()

    @enforce_types
    def set_opponent(self, opponent: Agent):
        self.opponent_agent = opponent

    def reset(self):
        # Reset the state of the environment to an initial state
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.board = [None] * N_ACTIONS  # MAX LENGTH DIMS
        self.observation_space = spaces.Box(0, 2, shape=(9,), dtype=np.int32)
        return self._encode_observations(self.board)

    def _encode_observations(self, board):

        ret = [x + 1 if x is not None else x for x in board]
        ret = [0 if x is None else x for x in ret]
        # print(f'IN _encode_observations: {ret}')
        res = np.asarray(ret)
        return res

    # Every step is followed immediately by a counter step  and victory is checked
    #
    def step(self, action):
        # Execute one time step within the environment
        # Assert that actions meet the specs
        # print(f'ACTION!!={action}')
        # RETURN VALUES
        global VERBOSE
        ret = {
            'observation': self.board,  # Have to recompute this after moves below to return it
            'reward': 0,
            'done': False,
            'info': {}
        }

        # END RETURN VALUES

        # if self.expected_player != player:
        # Then assume a self training session and just choose the next valid legal_move for the opponent
        # return ret['observation'], ret['reward'], ret['done'], ret['info']
        # raise ValueError(f'Invalid player moving, expected {self.expected_player} , was {player}')
        # assert (player == 0 or player == 1)
        # print('Action={},player={},y={}'.format(action, player, offset))
        # Now set it for the next play
        # self.expected_player = (player + 1) % 2
        player = 0
        offset = action

        if not self._check_move(player, offset):
            ret['info'] = {'error': f'INVALID MOVE {player}->{offset}, was {self.board[offset]}'}
            ret['reward'] = -1
            ret['done'] = True
            # raise ValueError('INVALID MOVE!!')
            # return -1  # FAILED
        else:
            self.board[offset] = player
            self.set_victory_conditions(offset, player, ret)
        # print(f'HERE NOW MAKE THE OPPONENT MOVE ')
        if not ret['done']:
            self._opponent_play(ret)
            self.set_victory_conditions(offset, 1, ret)

        # Now check for a draw game
        none_count = self.board.count(None)
        if none_count == 1:
            idx = self.board.index(None)
            # print(f'NOW MAKE THE DEFAULT PLAY of IDX({idx})')
            self.board[idx] = 0
            self.set_victory_conditions(idx, 0, ret)
            if not ret['done']:
                ret['done'] = True
                ret['reward'] = DefaultScores.draw

        if VERBOSE:
            if ret['reward'] == 1:
                # print('.', end='')
                pass
            elif ret['reward'] == DefaultScores.draw:
                print('DRAW!')
                if VERBOSE >= 2:
                    self.render()
            elif ret['reward'] == DefaultScores.win:
                print('WIN!')
                if VERBOSE >= 2:
                    self.render()
            elif ret['reward'] == DefaultScores.loss:
                print('LOSS!')
                if VERBOSE >= 2:
                    self.render()
            elif ret['reward'] == DefaultScores.illegal_move:
                print('LOSS_ILLEGAL_MOVE')
                if VERBOSE >= 2:
                    self.render()

        ret['observation'] = self._encode_observations(self.board)
        return ret['observation'], ret['reward'], ret['done'], ret['info']

    def set_victory_conditions(self, offset, player, ret):
        if self._is_victorious(player, offset):
            msg = f'{"WIN" if player == 0 else "LOSS"}'.upper()
            # Our player
            if player == 0:
                ret['reward'] = DefaultScores.win
            else:
                ret['reward'] = DefaultScores.loss
            ret['info'] = {'error': msg}
            ret['done'] = True
        else:
            ret['reward'] = DefaultScores.move
            ret['done'] = False

            # O, 0 is our O

    def _opponent_play(self, ret):
        player = 1
        # action = self._cpu_player(self._encode_observations(self.board))
        # offset = action
        offset = self.opponent_agent.predict(self.board, self)
        self.board[offset] = player
        return ret

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        me = self
        brd = self.board
        for i in range(0, COLS):
            ctr = i * COLS
            print(
                f'{ctr}-{(ctr + COLS) - 1}|{self.print_cell(brd[ctr])}|{self.print_cell(brd[ctr + 1])}|{self.print_cell(brd[ctr + 2])}|')

    def print_cell(self, x):
        if x is None:
            return ' '
        if x == 0:
            return 'O'
        return 'X'

    def _is_victorious(self, player, offset):
        ret = False
        brd = self.board
        # Check verticals
        for column in range(0, 3):
            idx1 = column
            idx2 = column + (COLS * 1)
            idx3 = column + (COLS * 2)

            if brd[idx1] == player and brd[idx2] == player and brd[idx3] == player:
                ret = True
            # if self.board[column] == player or self.board[column
            # print(f'Checking: {idx1},{idx2},{idx3} == {brd[idx1]},{brd[idx2]},{brd[idx3]}')

        # Check Horizontal
        for column in range(0, 3):
            idx = column * COLS
            if brd[idx] == player and brd[idx + 1] == player and brd[idx + 2] == player:
                ret = True
            # if self.board[column] == player or self.board[column
            # print(f'Checking: {idx1},{idx2},{idx3} == {brd[idx1]},{brd[idx2]},{brd[idx3]}')
        # Check diagonals

        if brd[0] == player and brd[4] == player and brd[8] == player:
            ret = True
        if brd[2] == player and brd[4] == player and brd[6] == player:
            ret = True

        # print(f'Checking victory ... {ret}')
        return ret

    def _check_move(self, player, _offset):
        # opposite_player = (player + 1) % 2
        if self.board[_offset] is not None:
            return False
        return True

    def _get_valid_cpu_action(self, obs):
        return self._get_valid_action(1)

    def valid_action_mask(self):
        return [True if self.board[x] is None else False for x in range(0, len(self.board))]

    def get_valid_action_indices(self):
        ret = self.valid_action_mask()
        indices = [i for i, x in enumerate(ret) if x is True]
        return indices

    def _get_valid_action(self, player):
        idx = random.randint(0, len(self.board) - 1)
        # print(f'IDX={idx}')
        while self.board[idx] is not None:
            idx = random.randint(0, len(self.board) - 1)

        return idx
