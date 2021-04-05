import secrets
from pprint import pprint
from typing import List, Dict, Optional, Tuple

import gym
import numpy as np
from gym import spaces
from collections import defaultdict

from game_engine import Agent
from gym_checkers.engine import N_ROWS, N_COLS, N_ACTIONS, get_offset_from_tuple, get_tuple_from_offset, tuple_add, tuple_subtract, CheckersScores
from enforce_typing import enforce_types
from numbers import Number
from icecream import ic

VERBOSE = False
NO_LAST_MOVE = (-1, -1)
PLAYER_SELF = 1
PLAYER_OPPONENT = 2
PLAYER_EMPTY = 0


def log(msg):
    global VERBOSE
    if VERBOSE:
        print(msg)


# @enforce_types
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

        self.is_king_map = {}
        self.current_player = PLAYER_SELF
        self.reset()

    # @enforce_types
    def set_opponent(self, opponent: Agent):
        self.opponent_agent = opponent

    # @enforce_types
    def reset(self) -> np.array:
        self.observation_space = spaces.Box(
            0, 3, shape=(N_ACTIONS,), dtype=np.int32)
        self.board = self._generate_starting_board()
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])
        self.previous_action = NO_LAST_MOVE
        self.opponent_prev_action = NO_LAST_MOVE

        return self._encode_observations(self.board)

    # @enforce_types
    def _encode_observations(self, board: List) -> np.array:
        return np.asarray(board)

    # Every step is followed immediately by a counter step  and victory is checked
    #

    # @enforce_types
    def _make_move(self, player: int,  action: Tuple[int, int]) -> Number:
        self.current_player = player
        reward = 0

        old_offset = action[0]
        new_offset = action[1]

        if player == PLAYER_SELF:
            self.previous_action = action
        else:
            self.opponent_prev_action = action

        move = tuple_subtract(
            get_tuple_from_offset(old_offset),
            get_tuple_from_offset(new_offset)
        )

        self.board[old_offset] = PLAYER_EMPTY
        self.board[new_offset] = player
        if abs(move[0]) >= 2:
            needed_move = self._get_needed_move_for_capture(move)
            captured_offset = get_offset_from_tuple(
                tuple_subtract(
                    get_tuple_from_offset(action[0]),
                    needed_move
                )
            )
            self.board[captured_offset] = PLAYER_EMPTY
            reward = CheckersScores.capture_move
        else:
            reward = CheckersScores.legal_move

        # Now ensure that the king status moves with the piece movement
        if self._is_king(old_offset):
            self._is_king(old_offset, False)
            self._is_king(new_offset, True)
            ic(f'moving is_king from {old_offset} to {new_offset}')
        else:
            # Now check if its NEWLY minted king king
            if player == PLAYER_SELF:
                if new_offset > (N_ACTIONS - N_ROWS):
                    self._is_king(new_offset, True)
                    ic(f'is_king to true for {new_offset}')
                    reward += CheckersScores.king_move
            if player == PLAYER_OPPONENT:
                if new_offset < (0+N_ROWS):
                    self._is_king(new_offset, True)
                    ic(f'is_king to true for {new_offset}')

        return reward

    # @enforce_types
    def step(self, _action: Tuple[int, int]) -> Tuple[np.array, float, bool, Dict]:
        ic(f'step({_action})')
        action = tuple(_action)
        ic(action)
        # self.render()
        # RETURN VALUES
        global VERBOSE
        ret = {
            'observation': self.board,  # Have to recompute this after moves below to return it
            'reward': 0,
            'done': False,
            'info': {}
        }
        reward = self._make_move(PLAYER_SELF, action)
        ic(reward)
        new_offset = self.previous_action[1]

        if reward >= CheckersScores.capture_move:
            results = self._get_captures_for_square(PLAYER_SELF, new_offset)
            while len(results):
                a = secrets.choice(results)
                reward += self._make_move(PLAYER_SELF, a)
                ic(reward)
                new_offset = self.previous_action[1]
                results = self._get_captures_for_square(
                    PLAYER_SELF, new_offset)

        self.render()
        input('Hit enter to continue')

        # Now opponentplay
        opponent_action = self.opponent_agent.predict(self.board, self)
        if opponent_action is not None:
            self.render()
            opponent_reward = self._make_move(PLAYER_OPPONENT, opponent_action)
            ic(opponent_action)
            ic(opponent_reward)
            if opponent_reward >= CheckersScores.capture_move:
                new_offset = self.opponent_prev_action[1]
                results = self._get_captures_for_square(
                    PLAYER_OPPONENT, new_offset)
                while len(results):
                    a = secrets.choice(results)
                    opponent_reward += self._make_move(PLAYER_OPPONENT, a)
                    ic(reward)
                    new_offset = self.opponent_prev_action[1]
                    results = self._get_captures_for_square(
                        PLAYER_OPPONENT, new_offset)

            if opponent_reward >= CheckersScores.capture_move:
                reward = reward - 1

        ret['observation'] = self._encode_observations(self.board)
        ret['reward'] = reward

        # Now determine if its done

        player1 = sum([1 for p in self.board if p == PLAYER_SELF])
        player2 = sum([1 for p in self.board if p == PLAYER_OPPONENT])
        if player1 <= 0 or player2 <= 0:
            ret['done'] = True
            if player1 <= 0:
                ret['info']['winning_player'] = PLAYER_OPPONENT
                reward = CheckersScores.loss
            if player2 <= 0:
                ret['info']['winning_player'] = PLAYER_SELF
                reward = CheckersScores.win

        return ret['observation'], ret['reward'], ret['done'], ret['info']

    # @enforce_types
    def _generate_starting_board(self):
        # Our board is a 1d array of 0 ( empty ), 1 (red) , and 2 (black)
        l = [0] * N_ACTIONS

        # First the white side, the start of the array , every other item
        whites = [1, 3, 5, 7, 8, 10, 12, 14, 17, 19, 21, 23]
        blacks = [40, 42, 44, 46, 49, 51, 53, 55, 56, 58, 60, 62]
        for w in whites:
            l[w] = 1
        for b in blacks:
            l[b] = 2

        return l

    # @enforce_types
    def _is_king(self, index: int, should_set_value: Optional[bool] = None) -> bool:
        if should_set_value is not None:
            self.is_king_map[index] = should_set_value
            return should_set_value

        if index in self.is_king_map:
            return True

        return False

    def _get_captures_for_square(self, player: Number, index: Number):

        opposite_player = 0

        if player == PLAYER_SELF:
            legal_moves = self.possible_moves_white
            opposite_player = PLAYER_OPPONENT

        elif player == PLAYER_OPPONENT:
            legal_moves = self.possible_moves_black
            opposite_player = PLAYER_SELF

        idx = index
        legal_from_here = []
        pos_tuple = get_tuple_from_offset(idx)
        all_moves = legal_moves
        if self._is_king(idx):
            all_moves = self.possible_moves_black + self.possible_moves_white
        for lm_idx, m in enumerate(all_moves):
            res = tuple_add(pos_tuple, m)
            # print(f'TESTING was({prev_offset} -- {pos_tuple}+{m}={res}!!')
            # print(res)
            if res[0] >= N_COLS or res[0] < 0:
                continue
            if res[1] >= N_COLS or res[1] < 0:
                continue
            # If the square is empty then continue, else add it
            new_offset = get_offset_from_tuple(res)
            if self.board[new_offset] != 0:
                continue

            needs_capture_move = self._get_needed_move_for_capture(m)
            new_res = tuple_add(pos_tuple, needs_capture_move)
            if new_res[0] >= N_COLS or new_res[0] < 0:
                continue
            if new_res[1] >= N_COLS or new_res[1] < 0:
                continue

            capture_offset = get_offset_from_tuple(new_res)
            if self.board[capture_offset] == opposite_player:
                legal_from_here.append((idx, new_offset))
        # print(f'All captures {legal_from_here}')
        return legal_from_here

    # @enforce_types
    def _get_all_captures(self, player: Number):
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
            all_moves = legal_moves
            if self._is_king(idx):
                all_moves = self.possible_moves_black + self.possible_moves_white
            for lm_idx, m in enumerate(all_moves):
                res = tuple_add(pos_tuple, m)
                # print(f'TESTING was({prev_offset} -- {pos_tuple}+{m}={res}!!')
                # print(res)
                if res[0] >= N_COLS or res[0] < 0:
                    continue
                if res[1] >= N_COLS or res[1] < 0:
                    continue
                # If the square is empty then continue, else add it
                new_offset = get_offset_from_tuple(res)
                if self.board[new_offset] != 0:
                    continue

                needs_capture_move = self._get_needed_move_for_capture(m)
                new_res = tuple_add(pos_tuple, needs_capture_move)
                if new_res[0] >= N_COLS or new_res[0] < 0:
                    continue
                if new_res[1] >= N_COLS or new_res[1] < 0:
                    continue

                capture_offset = get_offset_from_tuple(new_res)
                if self.board[capture_offset] == opposite_player:
                    legal_from_here.append((idx, new_offset))
        # print(f'All captures {legal_from_here}')
        return legal_from_here

    @ enforce_types
    def _get_needed_move_for_capture(self, move: Tuple[int, int]) -> Tuple[int, int]:

        if move == (2, -2):
            return (1, -1)
        elif move == (2, 2):
            return (1, 1)
        elif move == (-2, -2):
            return (-1, -1)
        elif move == (-2, 2):
            return (-1, 1)

    @ enforce_types
    def get_random_valid_action(self, player: Number) -> Tuple[int, int]:

        captures = self._get_all_captures(player)
        if len(captures):
            return secrets.choice(captures)
        else:
            actions = self.get_all_valid_actions(player)
            if not len(actions):
                return None
            else:
                return secrets.choice(actions)

    @ enforce_types
    def get_all_valid_actions(self, player: Number):
        ret = []
        # First get all the indicies of player 1
        idxs = [idx for idx, e in enumerate(self.board) if e == player]
        # For each idx generate possible new moves , only add them if possible
        for idx in idxs:
            moves = self._get_valid_offset_for_piece(
                idx,
                self._get_legal_moves_for_player(player, idx))
            if len(moves):
                for m in moves:
                    ret.append((idx, m))

        return ret

    def valid_action_mask(self):
        actions = self.get_all_valid_actions(self.current_player)
        ic(actions)
        valid_actions = self.get_all_valid_actions(PLAYER_SELF)
        row1 = [False] * N_ACTIONS
        row2 = [False] * N_ACTIONS
        for action in valid_actions:
            row1[action[0]] = True
            row2[action[1]] = True
        ret = np.vstack((row2,row1))
        ic(ret)
        return ret

    @ enforce_types
    def _get_legal_moves_for_player(self, player: Number, index: Optional[Number]) -> List:
        if index and self._is_king(index):
            return self._get_legal_moves_for_king()
        if player == PLAYER_SELF:
            return self.legal_moves_white
        elif player == PLAYER_OPPONENT:
            return self.legal_moves_black

    @ enforce_types
    def _get_legal_moves_for_king(self) -> List:
        return self.legal_moves_white + self.legal_moves_black

    @ enforce_types
    def _get_valid_offset_for_piece(self, offset: Number, legal_moves: List):
        prev_offset = offset
        legal_from_here = []
        # print(self.legal_moves)
        pos_tuple = get_tuple_from_offset(prev_offset)
        for m in legal_moves:
            res = tuple_add(pos_tuple, m)
            # print(f'TESTING was({prev_offset} -- {pos_tuple}+{m}={res}!!')
            # print(res)
            if res[0] >= N_COLS or res[0] < 0:
                continue
            if res[1] >= N_COLS or res[1] < 0:
                continue
            # If the square is empty then continue, else add it
            new_offset = get_offset_from_tuple(res)
            if self.board[new_offset] != 0:
                continue
            legal_from_here.append(new_offset)
        return legal_from_here

    @ enforce_types
    def render(self, mode: str = 'human', close: bool = False) -> None:
        # Render the environment to the screen
        brd = self.board
        for i in range(0, N_COLS):
            ctr = i * N_COLS
            high_bound = (ctr + N_COLS)-1
            str = ''
            for x in range(0, N_ROWS):
                str += f'|{self.print_cell(brd[ctr + x])}'

            print(f'{ctr:02}-{high_bound:02}\t{str}')

    @ enforce_types
    def print_cell(self, player: Optional[int]):
        if player == 0:
            return '__'
        elif player == 1:
            return '//'
        elif player == 2:
            return '00'
        return 'ERROR'
