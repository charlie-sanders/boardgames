import secrets
from pprint import pprint
from typing import List, Dict, Optional, Tuple

import gym
import numpy as np
from gym import spaces

from game_engine import Agent
from gym_checkers.engine import ROWS, COLS, N_ACTIONS, get_offset_from_tuple, get_tuple_from_offset, tuple_add, tuple_subtract, CheckersScores
from enforce_typing import enforce_types
from numbers import Number

VERBOSE = False
NO_LAST_MOVE = (-1, -1)
PLAYER_SELF = 1
PLAYER_OPPONENT = 2
PLAYER_EMPTY = 0


def log(msg):
    global VERBOSE
    if VERBOSE:
        print(msg)


@enforce_types
class CheckersEnv(gym.Env):
    """CheckersEnv  that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CheckersEnv, self).__init__()
        # Flatten the board to a 1d vector indexable by
        # (row,col) -> index=rol*col -> board[index]
        self.board = [None] * N_ACTIONS  # MAX LENGTH DIMS
        self.opponent_agent = None
        self.previous_action = NO_LAST_MOVE
        self.opponent_prev_action = NO_LAST_MOVE
        # These are static legal moves given a persons top down view of the board
        # We test to see if they are in bounds by converting them to offsets

        # THESE POSITIONS ARE IMPORTANT, we use the index in possible_moves to see if that spot is occupied
        self.legal_moves_white = [(1, -1), (1, 1)]
        self.legal_moves_black = [(-1, -1), (-1, 1)]

        self.possible_moves_white = [(2, -2), (2, 2)]
        self.possible_moves_black = [(-2, -2), (-2, 2)]

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

    @enforce_types
    def handle_capture_move(self, action: Tuple[int, int], move: Tuple[int, int], ret: Dict) -> bool:
        # Its a capature
        if abs(move[0]) >= 2:
            needed_move = self.get_needed_move_for_capture(move)
            captured_offset = get_offset_from_tuple(
                tuple_subtract(
                    get_tuple_from_offset(action[0]),
                    needed_move
                )
            )
            self.board[captured_offset] = PLAYER_EMPTY
            ret['reward'] = CheckersScores.capture_move
            return True
        return False
    # Every step is followed immediately by a counter step  and victory is checked
    #

    @enforce_types
    def step(self, _action: Tuple[int, int]) -> Tuple[np.array, float, bool, Dict]:

        action = _action

        print(f'step(action={action})')
        # self.render()
        # RETURN VALUES
        global VERBOSE
        ret = {
            'observation': self.board,  # Have to recompute this after moves below to return it
            'reward': 0,
            'done': False,
            'info': {}
        }

        old_offset = action[0]
        new_offset = action[1]

        move = tuple_subtract(
            get_tuple_from_offset(old_offset),
            get_tuple_from_offset(new_offset)
        )

        self.board[old_offset] = PLAYER_EMPTY
        self.board[new_offset] = PLAYER_SELF

        self.previous_action = action

        self.handle_capture_move(action, move, ret)

        # Now opponentplay

        opponent_action = self.opponent_agent.predict(self.board, self)
        if opponent_action is not None:
            self.render()
            opponent_old_offset = opponent_action[0]
            opponent_new_offset = opponent_action[1]
            opponent_move = tuple_subtract(
                get_tuple_from_offset(opponent_old_offset),
                get_tuple_from_offset(opponent_new_offset)
            )
            print(f'OponnentAction({opponent_action})')
            self.opponent_prev_action = opponent_action
            self.board[opponent_action[0]] = PLAYER_EMPTY
            self.board[opponent_action[1]] = PLAYER_OPPONENT
            opponent_ret = {}
            if self.handle_capture_move(opponent_action, opponent_move, opponent_ret):
                print(f'OPPONENT_CAPTURE opponent_ret={opponent_ret}')


        self.set_victory_conditions(self.opponent_prev_action, 1, ret)

        # if not ret['done']:
        #     self.current_player = 1
        #     self._opponent_play(ret)
        #     self.set_victory_conditions(self.opponent_prev_action, 1, ret)
        #     self.current_player = 0

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

    @enforce_types
    def get_all_captures(self, player: Number):
        opposite_player = PLAYER_OPPONENT
        if player == PLAYER_SELF:
            legal_moves = self.possible_moves_white
            opposite_player = PLAYER_OPPONENT

        elif player == PLAYER_OPPONENT:
            legal_moves = self.possible_moves_black
            opposite_player = PLAYER_SELF

        idxs = [idx for idx, e in enumerate(self.board) if e == player]
        legal_from_here = []

        for idx in idxs:
            # print(self.legal_moves)
            pos_tuple = get_tuple_from_offset(idx)
            for lm_idx, m in enumerate(legal_moves):
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

                needs_capture_move = self.get_needed_move_for_capture(m)
                new_res = tuple_add(pos_tuple, needs_capture_move)
                if new_res[0] >= COLS or new_res[0] < 0:
                    continue
                if new_res[1] >= COLS or new_res[1] < 0:
                    continue

                capture_offset = get_offset_from_tuple(new_res)
                if self.board[capture_offset] == opposite_player:
                    legal_from_here.append((idx, new_offset))
        print(f'All captures {legal_from_here}')
        return legal_from_here

    def get_needed_move_for_capture(self, move):

        if move == (2, -2):
            return (1, -1)
        elif move == (2, 2):
            return (1, 1)
        elif move == (-2, -2):
            return (-1, -1)
        elif move == (-2, 2):
            return (-1, 1)

    @enforce_types
    def get_random_valid_action(self, player: Number) -> Tuple[int, int]:

        captures = self.get_all_captures(player)
        if len(captures):
            return secrets.choice(captures)
        else:
            actions = self.get_all_valid_actions(player)
            return secrets.choice(actions)

    @enforce_types
    def get_all_valid_actions(self, player: Number):
        ret = []
        # First get all the indicies of player 1
        idxs = [idx for idx, e in enumerate(self.board) if e == player]
        # For each idx generate possible new moves , only add them if possible
        for idx in idxs:
            moves = self.get_valid_offset_for_piece(
                idx, self.get_legal_moves_for_player(player))
            if len(moves):
                for m in moves:
                    ret.append((idx, m))

        return ret

    @enforce_types
    def get_legal_moves_for_player(self, player: Number) -> List:
        if player == PLAYER_SELF:
            return self.legal_moves_white
        elif player == PLAYER_OPPONENT:
            return self.legal_moves_black

    @enforce_types
    def get_valid_offset_for_piece(self, offset: Number, legal_moves: List):
        prev_offset = offset
        legal_from_here = []
        # print(self.legal_moves)
        pos_tuple = get_tuple_from_offset(prev_offset)
        for m in legal_moves:
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
    def set_victory_conditions(self, offset: Tuple[int, int], player: Number, ret: Dict) -> None:
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
