from dataclasses import dataclass
from numbers import Number
from typing import Tuple

from enforce_typing import enforce_types

ROWS = 5
COLS = 5
N_ACTIONS = ROWS * COLS
VERBOSE = 0


@enforce_types
def tuple_subtract(t1: Tuple[int, int], t2: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(map(lambda i, j: i - j, t1, t2))


@enforce_types
def tuple_add(t1: Tuple[int, int], t2: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(map(lambda i, j: i + j, t1, t2))


@enforce_types
def get_tuple_from_offset(pos: Number) -> Tuple[int, int]:
    rem = pos % COLS
    times = int(pos / COLS)
    return (times, rem)


@enforce_types
def get_offset_from_tuple(m: Tuple[int, int], prev_offset: Number = 0) -> Number:
    ret = prev_offset + ((m[0] * ROWS) + m[1])
    return ret


@dataclass
class Scores:
    win: int = 10
    draw: int = 0
    loss: int = -10
    illegal_move: int = -1
    legal_move: int = 1
