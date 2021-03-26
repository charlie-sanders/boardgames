from datetime import datetime, timedelta

from abc import ABC
from typing import List
from src.data.DataFactory import DataFactory
import pandas as pd

from src.util import DateUtil


class Source(ABC):

    def step(self) -> List:
        raise NotImplementedError()


class MairlySource(Source):

    def __init__(self, ticker: str, start: datetime, end: datetime):
        self.ticker = ticker
        self.start = start
        self.current = start
        self.end = end

    def step(self) -> pd.DataFrame:
        day = self.increase_day()
        df = DataFactory.get_trades(self.ticker, day, day)
        return df

    def increase_day(self):
        self.current = self.current + timedelta(days=1)
        ret =DateUtil.to_string(self.current)
        print(f'Returning {ret} in increase_day()')
        return ret
