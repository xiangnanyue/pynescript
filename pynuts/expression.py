from __future__ import annotations

import itertools

from dataclasses import dataclass
from typing import ClassVar, Union
from typing import Generic, TypeVar
from collections.abc import Sequence

from .config import PlotDisplay, ScaleType, StrategyDirection


T = TypeVar("T")


class Series(Generic[T]):
    data: list[T] | T

    def __init__(self, data: Sequence[T] | None = None):
        self.data = list(data) if data is not None else []

    def __getitem__(self, item):
        if isinstance(item, int):
            item = -1 - item
            return self.data[item]
        if isinstance(item, slice):
            start = -1 - item.start if item.start is not None else None
            stop = -1 - item.stop if item.stop is not None else None
            step = -item.step if item.step is not None else -1
            return self.data[start:stop:step]
        raise ValueError()

    def set(self, item):
        if isinstance(item, Series):
            item = item[0]
        self.data[-1] = item

    def add(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)


class SimpleData(Series[T]):
    def __init__(self, value: T | None = None):
        super().__init__()
        self.data = value

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data
        if isinstance(item, slice):
            return itertools.islice(itertools.repeat(self.data), item.start, item.stop, item.step)
        raise ValueError()

    def set(self, item):
        if isinstance(item, Series):
            item = item[0]
        self.data = item

    def add(self, item):
        self.data = item

    def extend(self, items):
        self.data = items[0]


class Const(SimpleData[T]):
    # def __int__(self) -> int:
    #     return int(self.data)
    #
    # def __float__(self) -> float:
    #     return float(self.data)

    def __str__(self) -> str:
        return str(self.data)

    def __eq__(self, other: Union[T, 'Const']) -> bool:
        return self.data == (other.data if isinstance(other, Const) else other)

    def __lt__(self, other: Union[T, 'Const']) -> bool:
        return self.data < (other.data if isinstance(other, Const) else other)

    def __le__(self, other: Union[T, 'Const']) -> bool:
        return self.data <= (other.data if isinstance(other, Const) else other)

    def __gt__(self, other: Union[T, 'Const']) -> bool:
        return self.data > (other.data if isinstance(other, Const) else other)

    def __ge__(self, other: Union[T, 'Const']) -> bool:
        return self.data >= (other.data if isinstance(other, Const) else other)

    def __pos__(self):
        return +self.data  # 一元正号

    def __neg__(self):
        return -self.data  # 一元负号

    def __add__(self, other: Union[T, 'Const']) -> Union[T, 'Const']:
        if isinstance(other, Const):
            return Const(self.data + other.data)
        # 处理与 int 或 float isinstance(other, (int, float))
        return self.data + other

    def __radd__(self, other: Union[T, 'Const']) -> Union[T, 'Const']:
        return self.__add__(other)

    def __sub__(self, other: Union[T, 'Const']) -> Union[T, 'Const']:
        if isinstance(other, Const):
            return Const(self.data - other.data)
        return self.data - other

    def __mul__(self, other: Union[T, 'Const']) -> Union[T, 'Const']:
        if isinstance(other, Const):
            return Const(self.data * other.data)
        return self.data * other

    def __rmul__(self, other: Union[T, 'Const']) -> Union[T, 'Const']:
        return self.__mul__(other)  # 调用 __mul__ 方法

    def __truediv__(self, other: Union[T, 'Const']) -> Union[T, 'Const']:
        if isinstance(other, Const):
            return Const(self.data / other.data)
        return self.data / other

    def __floordiv__(self, other: Union[T, 'Const']) -> Union[T, 'Const']:
        if isinstance(other, Const):
            return Const(self.data // other.data)
        return self.data // other


class Source(Series[T]):
    pass


class Input(SimpleData[T]):

    def __init__(self, defval: Const[T] | Source[T], title: Const[str] | None = None, tooltip: Const[str] | None = None,
                 inline: Const[str] | None = None, group: Const[str] | None = None,
                 display: Const[PlotDisplay] | None = None):
        super().__init__()
        self.defval = defval
        self.title = title
        self.tooltip = tooltip
        self.inline = inline
        self.group = group
        self.display = display

        self.set(self.defval)


@dataclass
class Strategy:
    title: Const[str]
    shorttitle: Const[str] | None = None
    overlay: Const[bool] | None = None
    format: Const[str] | None = None
    precision: Const[int] | None = None
    scale: Const[ScaleType] | None = None
    pyramiding: Const[int] | None = None
    calc_on_order_fills: Const[bool] | None = None
    cacl_on_every_tick: Const[bool] | None = None
    max_bars_back: Const[int] | None = None
    backtest_fill_limits_assumption: Const[int] | None = None
    default_qty_type: Const[str] | None = None
    default_qty_value: Const[int] | Const[float] | None = None
    initial_capital: Const[int] | Const[float] | None = None
    currency: Const[str] | None = None
    slippage: Const[int] | None = None
    commission_type: Const[str] | None = None
    commition_value: Const[int] | Const[float] | None = None
    process_orders_on_close: Const[bool] | None = None
    close_entries_rule: Const[str] | None = None
    margin_long: Const[int] | Const[float] | None = None
    margin_short: Const[int] | Const[float] | None = None
    explicit_plot_zorder: Const[bool] | None = None
    max_lines_count: Const[int] | None = None
    max_labels_count: Const[int] | None = None
    max_boxes_count: Const[int] | None = None
    risk_free_rate: Const[int] | Const[float] | None = None
    use_bar_magnifier: Const[bool] | None = None
    fill_orders_on_standard_ohlc: Const[bool] | None = None
    max_polylines_count: Const[int] | None = None

    long: ClassVar = StrategyDirection.long
    short: ClassVar = StrategyDirection.short

    fixed: ClassVar = "fixed"
    cash: ClassVar = "cash"
    percent_of_equity: ClassVar = "percent_of_equity"

    @dataclass
    class entry:
        id: Series[str]
        direction: Series[StrategyDirection]
        qty: Series[int] | Series[float] | None = None
        limit: Series[int] | Series[float] | None = None
        stop: Series[int] | Series[float] | None = None
        oca_name: Series[str] | None = None
        oca_type: Input[str] | None = None
        comment: Series[str] | None = None
        alert_message: Series[str] | None = None
        disable_alert: Series[bool] | None = None


if __name__ == "__main__":
    strategy = Strategy("test", default_qty_type="cash")
    print(strategy)
    print(strategy.default_qty_type, strategy.max_lines_count)

    const = Const(2.0)
    print(3.0 * const)
    print(3.0 + const)
