import ta
import numpy as np
import pandas as pd

from .expression import Series, SimpleData, T


class TA:

    @classmethod
    def sma(cls, source: Series[int] | Series[float], length: SimpleData[int]) -> Series[float]:
        source = source[:length[0]][::-1]  # 反转序列
        source = pd.Series(source)
        result = ta.trend.sma_indicator(source, length[0]).iloc[-1]
        return [result]

    @classmethod
    def stdev(cls, source: Series[int] | Series[float], length: SimpleData[int]) -> Series[float]:
        source = source[: length[0]][::-1]
        source = pd.Series(source)
        result = source.rolling(window=length[0]).std().iloc[-1]  # 计算标准差
        return [result]

    @classmethod
    def rsi(cls, source: Series[int] | Series[float], length: SimpleData[int]) -> Series[float]:
        source = source[: length[0]][::-1]
        source = pd.Series(source)
        result = ta.momentum.rsi(source, length[0]).iloc[-1]
        return [result]

    @classmethod
    def crossover(cls, source1: Series[int] | Series[float], source2: Series[int] | Series[float]) -> Series[bool]:
        return [source1[0] > source2[0] and source1[1] <= source2[1]]

    @classmethod
    def crossunder(cls, source1: Series[int] | Series[float], source2: Series[int] | Series[float]) -> Series[bool]:
        return [source1[0] < source2[0] and source1[1] >= source2[1]]


class IsNa:
    def __call__(self, x: Series[T]) -> Series[bool]:
        return [x[0] is None or np.isnan(x[0])]

    def __eq__(self, other):
        if isinstance(other, Series):
            other = other[0]
        return other is None or isinstance(other, IsNa)
