from numbers import Number
from typing import Tuple

from enforce_typing import enforce_types
from dataclasses import dataclass

N_ROWS = 8
N_COLS = 8
N_ACTIONS = N_ROWS * N_COLS
VERBOSE = 0


@dataclass
class CheckersScores:
    win: int = 25
    draw: int = 0
    loss: int = -25
    illegal_move: int = -1
    legal_move: int = 1
    capture_move: int = 5
    king_move: int = 5


@enforce_types
def tuple_subtract(t1: Tuple[int, int], t2: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(map(lambda i, j: i - j, t1, t2))


@enforce_types
def tuple_add(t1: Tuple[int, int], t2: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(map(lambda i, j: i + j, t1, t2))


@enforce_types
def get_tuple_from_offset(pos: Number) -> Tuple[int, int]:
    rem = pos % N_COLS
    times = int(pos / N_COLS)
    return (times, rem)


@enforce_types
def get_offset_from_tuple(m: Tuple[int, int], prev_offset: Number = 0) -> Number:
    ret = prev_offset + ((m[0] * N_ROWS) + m[1])
    return ret
