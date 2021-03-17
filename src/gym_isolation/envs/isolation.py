import secrets
import sys
from pprint import pprint
from typing import List, Dict, Optional

import gym
import numpy as np
from gym import spaces

from gym_isolation.agents import Agent
from gym_isolation.engine import *

VERBOSE = False


def log(msg):
    global VERBOSE
    if VERBOSE:
        print(msg)


@enforce_types
class Isolation(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Isolation, self).__init__()
        # Flatten the board to a 1d vector indexable by
        # (row,col) -> index=rol*col -> board[index]
        self.board = [None] * N_ACTIONS  # MAX LENGTH DIMS
        self.opponent_agent = None
        self.previous_action = -1
        self.opponent_prev_action = -1
        self.current_player = 0  # 0 for us, 1 for theoretical CPU
        # These are static legal moves given a persons top down view of the board
        # We test to see if they are in bounds by converting them to offsets
        self.legal_moves = [(2, 1), (-2, 1), (-2, -1), (2, -1),
                            (1, 2), (-1, 2), (-1, -2), (1, -2)]
        self.reset()

    @enforce_types
    def set_opponent(self, opponent: Agent):
        self.opponent_agent = opponent

    @enforce_types
    def reset(self) -> np.array:
        self.observation_space = spaces.Box(0, 2, shape=(N_ACTIONS,), dtype=np.int32)
        self.board = [None] * N_ACTIONS
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.previous_action = -1
        self.opponent_prev_action = -1

        return self._encode_observations(self.board)

    @enforce_types
    def _encode_observations(self, board: List) -> np.array:
        ret: List = [x + 1 if x is not None else x for x in board]
        ret = [0 if x is None else x for x in ret]
        return np.asarray(ret)

    # Every step is followed immediately by a counter step  and victory is checked
    #
    @enforce_types
    def step(self, _action: np.int64) -> Tuple[np.array, float, bool, Dict]:
        action = _action
        # print(f'step(action={action})')
        # self.render()
        # RETURN VALUES
        global VERBOSE
        ret = {
            'observation': self.board,  # Have to recompute this after moves below to return it
            'reward': 0,
            'done': False,
            'info': {}
        }

        player = 0
        offset = action
        self.current_player = player
        if not self._check_move(player, offset, self.previous_action):
            ret['info'] = {'error': f'INVALID MOVE {self.previous_action} to {offset}, board is {self.board[offset]}'}
            ret['reward'] = Scores.illegal_move
            ret['done'] = False
            pprint(ret)
            # sys.exit()

            # return -1  # FAILED
        else:
            self.board[offset] = player
            self.previous_action = offset

        if not ret['done']:
            self.current_player = 1
            self._opponent_play(ret)
            self.set_victory_conditions(self.opponent_prev_action, 1, ret)
            self.current_player = 0

        log(f'Reward={ret["reward"]}')
        # Now encode to 0,1,2
        ret['observation'] = self._encode_observations(self.board)
        return ret['observation'], ret['reward'], ret['done'], ret['info']

    @enforce_types
    def set_victory_conditions(self, offset: Number, player: Number, ret: Dict) -> None:
        is_over, winner, o_count, x_count = self._is_game_over(player, offset)

        if is_over:
            msg = f'player({winner}) wins (O={o_count},offset={self.previous_action},X={x_count},offset={self.opponent_prev_action}) , '
            log(msg)
            # self.render()
            # Our player
            if winner == -1:
                ret['reward'] = Scores.draw
            elif winner == 0:
                ret['reward'] = Scores.win
            else:
                ret['reward'] = Scores.loss
            ret['info'] = {'error': msg}
            ret['done'] = True
        else:
            if ret['reward'] is None or ret['reward'] == 0:
                ret['reward'] = Scores.legal_move
                ret['done'] = False
            else:
                pass

    @enforce_types
    def _opponent_play(self, ret: Dict) -> Dict:
        player = 1

        action = self.opponent_agent.predict(self.board, self)
        if action is not None:
            log('Opponent moves from {} to {}'.format(self.opponent_prev_action, action))
            offset = action  # action.index(max(action))
            self.opponent_prev_action = offset
            self.board[offset] = player
        return ret

    @enforce_types
    def render(self, mode: str = 'human', close: bool = False) -> None:
        # Render the environment to the screen
        me = self
        brd = self.board
        for i in range(0, COLS):
            ctr = i * COLS
            high_bound = ctr + COLS
            str = ''
            for x in range(0, ROWS):
                str += f'|{self.print_cell(brd[ctr + x], ctr + x)}'

            print(f'{ctr:02}-{high_bound:02}\t{str}')

    @enforce_types
    def print_cell(self, player: Optional[int], idx: int = None):
        if player is None:
            if idx is None:
                return '  '
            else:
                return f'{idx:02}'
        if player == 0:
            return '__'
        elif player == 1:
            return '\\\\'
        return 'ERROR'

    # Only using best of 4 in a row
    @enforce_types
    def _is_game_over(self, player: Number, offset: Number) -> Tuple[bool, int, int, int]:
        ret = False
        brd = self.board
        o_count = sum([1 for i in self.board if i == 0])
        x_count = sum([1 for i in self.board if i == 1])
        n_count = sum([1 for x in self.board if x is None])
        winner = 0 if o_count > x_count else 1
        # print(f'O_COUNT={o_count},X_COUNT={x_count},winner={winner}')
        # Now check if there are no more valid movies
        p1_out_of_moves = self.get_random_valid_action(0) is None
        p2_out_of_moves = self.get_random_valid_action(1) is None

        # Draw
        if p1_out_of_moves or p2_out_of_moves:
            if p1_out_of_moves and not p2_out_of_moves:
                winner = 1
                ret = True
            elif p2_out_of_moves and not p1_out_of_moves:
                winner = 0
                ret = True
            elif o_count == x_count:
                winner = -1
                ret = True
            else:
                ret = True

        return ret, winner, o_count, x_count

    @enforce_types
    def _check_move(self, player: Number, _offset: Number, prev_offset: Number) -> bool:
        # Shortcut to assign the offset on initial move
        if prev_offset == -1:
            return True

        valid_indicies = self.get_valid_action_indices(player, prev_offset)
        if len(valid_indicies) == 0:
            return False
        ret = True
        if _offset in valid_indicies:
            ret = True
        else:
            ret = False
        return ret

    @enforce_types
    def _get_valid_cpu_action(self, obs: List) -> np.int64:
        return self.get_random_valid_action(1)

    @enforce_types
    def get_random_valid_action(self, player: Number = 0) -> int:
        ret = self.get_valid_action_indices(player, self.get_previous_action(player))
        idx = None
        if ret:
            idx = secrets.choice(ret)
        return idx

    @enforce_types
    def get_valid_action_indices(self, player: Number, prev_offset: Number) -> Number:
        moves = self.get_valid_actions(player)
        indices = [get_offset_from_tuple(m, prev_offset) for m in moves]
        idx1 = [i for i in indices if i >= 0]
        idx2 = [i for i in idx1 if self.board[i] is None]
        return idx2

    @enforce_types
    def valid_action_mask(self) -> List[bool]:
        ret = [False] * N_ACTIONS

        # print('LAST_PLAYER={}, LAST_OPPONENT={}'.format(self.previous_action, self.opponent_prev_action))
        idx = self.get_valid_action_indices(self.current_player, self.get_previous_action(self.current_player))
        for i in idx:
            ret[i] = True
        return ret

    @enforce_types
    def get_previous_action(self, player: Number = None) -> Number:
        idx = self.current_player
        if player is not None:
            idx = player
        if idx == 0:
            return self.previous_action
        else:
            return self.opponent_prev_action

    @enforce_types
    def get_valid_actions(self, player: Number) -> List[Tuple[int, int]]:
        prev_offset = self.get_previous_action(player)
        legal_from_here = []
        # print(self.legal_moves)
        pos_tuple = get_tuple_from_offset(prev_offset)
        for m in self.legal_moves:
            res = tuple_add(pos_tuple, m)
            # print(f'TESTING {pos_tuple}+{m}={res}!!')
            # print(res)
            if res[0] >= COLS or res[0] < 0:
                continue
            if res[1] >= COLS or res[1] < 0:
                continue
            if self.board[get_offset_from_tuple(res)] is not None:
                continue
            legal_from_here.append(tuple(m))
        return legal_from_here
