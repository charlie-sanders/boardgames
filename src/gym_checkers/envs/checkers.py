import secrets
from pprint import pprint
from typing import List, Dict, Optional, Tuple

import gym
import numpy as np
from gym import spaces

from game_engine import DefaultScores, Agent
from gym_checkers.engine import ROWS, COLS, N_ACTIONS, get_offset_from_tuple, get_tuple_from_offset, tuple_add, tuple_subtract
from enforce_typing import enforce_types
from numbers import Number

VERBOSE = False
NO_LAST_MOVE = (-1, -1)


def log(msg):
    global VERBOSE
    if VERBOSE:
        print(msg)


@enforce_types
class CheckersEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CheckersEnv, self).__init__()
        # Flatten the board to a 1d vector indexable by
        # (row,col) -> index=rol*col -> board[index]
        self.board = [None] * N_ACTIONS  # MAX LENGTH DIMS
        self.opponent_agent = None
        self.previous_action = NO_LAST_MOVE
        self.opponent_prev_action = NO_LAST_MOVE
        self.current_player = 0  # 0 for us, 1 for theoretical CPU
        # These are static legal moves given a persons top down view of the board
        # We test to see if they are in bounds by converting them to offsets
        self.legal_moves = [(1, -1), (1, 1)]
        self.reset()

    @enforce_types
    def set_opponent(self, opponent: Agent):
        self.opponent_agent = opponent

    @enforce_types
    def reset(self) -> np.array:
        self.observation_space = spaces.Box(
            0, 3, shape=(N_ACTIONS,), dtype=np.int32)
        self.board = self.generate_starting_board()
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])
        self.previous_action = NO_LAST_MOVE
        self.opponent_prev_action = NO_LAST_MOVE

        return self._encode_observations(self.board)

    @enforce_types
    def _encode_observations(self, board: List) -> np.array:
        return np.asarray(board)

    # Every step is followed immediately by a counter step  and victory is checked
    #
    # @enforce_types
    def step(self, _action: Tuple[int, int]) -> Tuple[np.array, float, bool, Dict]:
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
        # if not self._check_move(player, offset, self.previous_action):
        #     ret['info'] = {
        #         'error': f'INVALID MOVE {self.previous_action} to {offset}, board is {self.board[offset]}'}
        #     ret['reward'] = DefaultScores.illegal_move
        #     ret['done'] = False
        #     pprint(ret)
        #     # sys.exit()

        #     # return -1  # FAILED
        # else:
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
    def generate_starting_board(self):
        # Our board is a 1d array of 0 ( empty ), 1 (red) , and 2 (black)
        l = [0] * N_ACTIONS

        # First the white side, the start of the array , every other item
        whites = [1, 3, 5, 7, 8, 10, 12, 14, 17, 19, 21, 23]
        blacks = [40, 42, 44, 46,  49, 51, 53, 55, 56, 58, 60, 62]
        for w in whites:
            l[w] = 1
        for b in blacks:
            l[b] = 2

        return l

    def get_random_valid_action(self,player : Number) -> Tuple[int,int]:
        actions = self.get_all_valid_actions(player)
        return secrets.choice(actions)

    def get_all_valid_actions(self, player: Number):
        ret = []
        if player == 0:
            # First get all the indicies of player 1
            idxs = [idx for idx, e in enumerate(self.board) if e == 1]
            # For each idx generate possible new moves , only add them if possible
            for idx in idxs:
                moves = self.get_valid_offset_for_piece(idx)
                if len(moves):
                    for m in moves:
                        ret.append((idx, m))

        return ret

    def get_valid_offset_for_piece(self, offset: Number):
        prev_offset = offset
        legal_from_here = []
        # print(self.legal_moves)
        pos_tuple = get_tuple_from_offset(prev_offset)
        for m in self.legal_moves:
            res = tuple_add(pos_tuple, m)
            # print(f'TESTING was({prev_offset} -- {pos_tuple}+{m}={res}!!')
            # print(res)
            if res[0] >= COLS or res[0] < 0:
                continue
            if res[1] >= COLS or res[1] < 0:
                continue
            # If the square is empty then continue, else add it
            new_offset = get_offset_from_tuple(res)
            if self.board[new_offset] != 0:
                continue
            legal_from_here.append(new_offset)
        return legal_from_here

    @enforce_types
    def set_victory_conditions(self, offset: Number, player: Number, ret: Dict) -> None:
        pass
        # is_over, winner, o_count, x_count = self._is_game_over(player, offset)

        # if is_over:
        #     msg = f'player({winner}) wins (O={o_count},offset={self.previous_action},X={x_count},offset={self.opponent_prev_action}) , '
        #     log(msg)
        #     # self.render()
        #     # Our player
        #     if winner == -1:
        #         ret['reward'] = DefaultScores.draw
        #     elif winner == 0:
        #         ret['reward'] = DefaultScores.win
        #     else:
        #         ret['reward'] = DefaultScores.loss
        #     ret['info'] = {'error': msg}
        #     ret['done'] = True
        # else:
        #     if ret['reward'] is None or ret['reward'] == 0:
        #         ret['reward'] = DefaultScores.legal_move
        #         ret['done'] = False
        #     else:
        #         pass

    @enforce_types
    def _opponent_play(self, ret: Dict) -> Dict:
        player = 1

        action = self.opponent_agent.predict(self.board, self)
        if action is not None:
            log('Opponent moves from {} to {}'.format(
                self.opponent_prev_action, action))
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
            high_bound = (ctr + COLS)-1
            str = ''
            for x in range(0, ROWS):
                str += f'|{self.print_cell(brd[ctr + x])}'

            print(f'{ctr:02}-{high_bound:02}\t{str}')

    @enforce_types
    def print_cell(self, player: Optional[int]):
        if player == 0:
            return '__'
        elif player == 1:
            return '//'
        elif player == 2:
            return '00'
        return 'ERROR'
