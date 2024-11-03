from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar
from typing import Generic
from typing import TypeAlias
from typing import TypeVar

from operators import Operator
from historical_data import hist
from pandas import DataFrame

from pynescript import ast
from pynescript.ast import NodeVisitor
from pynescript.ast import parse


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


class simple(Series[T]):
    def __init__(self, value: T | None = None):
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


class const(simple[T]):
    pass


class source(Series[T]):
    pass


class plot_display(Enum):
    all = 1
    data_window = 2
    none = 3
    pane = 4
    price_scale = 5
    status_line = 6


class input(simple[T]):
    def __init__(
        self,
        defval: const[T] | source[T],
        title: const[str] | None = None,
        tooltip: const[str] | None = None,
        inline: const[str] | None = None,
        group: const[str] | None = None,
        display: const[plot_display] | None = None,
    ):
        self.defval = defval
        self.title = title
        self.tooltip = tooltip
        self.inline = inline
        self.group = group
        self.display = display

        self.set(self.defval)


display = plot_display


class ta:

    @classmethod
    def sma(cls, source: Series[int] | Series[float], length: simple[int]) -> Series[float]:
        import ta

        source = source[:length[0]][::-1]  # 反转序列
        source = pd.Series(source)
        result = ta.trend.sma_indicator(source, length[0]).iloc[-1]
        # result = source.rolling(window=length[0]).mean().iloc[-1]  # 计算均值
        return [result]

    @classmethod
    def stdev(cls, source: Series[int] | Series[float], length: simple[int]) -> Series[float]:
        source = source[: length[0]][::-1]
        source = pd.Series(source)
        result = source.rolling(window=length[0]).std().iloc[-1]  # 计算标准差
        return [result]

    @classmethod
    def rsi(cls, source: Series[int] | Series[float], length: simple[int]) -> Series[float]:
        import ta

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


class strategy_direction(Enum):
    long = 1
    short = 2


void: TypeAlias = None


class scale_type(Enum):
    right = 1
    left = 2
    none = 3


scale = scale_type


@dataclass
class strategy:
    title: const[str]
    shorttitle: const[str] | None = None
    overlay: const[bool] | None = None
    format: const[str] | None = None
    precision: const[int] | None = None
    scale: const[scale_type] | None = None
    pyramiding: const[int] | None = None
    calc_on_order_fills: const[bool] | None = None
    cacl_on_every_tick: const[bool] | None = None
    max_bars_back: const[int] | None = None
    backtest_fill_limits_assumption: const[int] | None = None
    default_qty_type: const[str] | None = None
    default_qty_value: const[int] | const[float] | None = None
    initial_capital: const[int] | const[float] | None = None
    currency: const[str] | None = None
    slippage: const[int] | None = None
    commission_type: const[str] | None = None
    commition_value: const[int] | const[float] | None = None
    process_orders_on_close: const[bool] | None = None
    close_entries_rule: const[str] | None = None
    margin_long: const[int] | const[float] | None = None
    margin_short: const[int] | const[float] | None = None
    explicit_plot_zorder: const[bool] | None = None
    max_lines_count: const[int] | None = None
    max_labels_count: const[int] | None = None
    max_boxes_count: const[int] | None = None
    risk_free_rate: const[int] | const[float] | None = None
    use_bar_magnifier: const[bool] | None = None
    fill_orders_on_standard_ohlc: const[bool] | None = None
    max_polylines_count: const[int] | None = None

    long: ClassVar = strategy_direction.long
    short: ClassVar = strategy_direction.short

    fixed: ClassVar = "fixed"
    cash: ClassVar = "cash"
    percent_of_equity: ClassVar = "percent_of_equity"

    @dataclass
    class entry:
        id: Series[str]
        direction: Series[strategy_direction]
        qty: Series[int] | Series[float] | None = None
        limit: Series[int] | Series[float] | None = None
        stop: Series[int] | Series[float] | None = None
        oca_name: Series[str] | None = None
        oca_type: input[str] | None = None
        comment: Series[str] | None = None
        alert_message: Series[str] | None = None
        disable_alert: Series[bool] | None = None


class na_type:
    def __call__(self, x: Series[T]) -> Series[bool]:
        return [x[0] is None or np.isnan(x[0])]

    def __eq__(self, other):
        if isinstance(other, Series):
            other = other[0]
        return other is None or isinstance(other, na_type)


na = na_type()


class ExampleScriptExecutor:
    class Visitor(NodeVisitor):
        def __init__(self, executor: ExampleScriptExecutor):
            self.executor = executor

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                if self.executor.scopes:
                    for scope in self.executor.scopes:
                        if node.id in scope:
                            node_store = scope[node.id]
                            return self.executor.nodes[node_store]
                if node.id in self.executor.builtins:
                    return self.executor.builtins[node.id]
                if node.id in self.executor.sources:
                    return self.executor.sources[node.id]
            return node

        def visit_Attribute(self, node: ast.Attribute):
            if isinstance(node.ctx, ast.Load):
                value = self.visit(node.value)
                return getattr(value, node.attr)
            return node

        def visit_Constant(self, node: ast.Constant):
            return const(node.value)

        def visit_Call(self, node: ast.Call):
            func = self.visit(node.func)

            args = []
            kwargs = {}

            found_has_name = False
            for arg in node.args:
                if arg.name:
                    found_has_name = True
                    kwargs[arg.name] = self.visit(arg.value)
                elif found_has_name:
                    raise ValueError()
                else:
                    args.append(self.visit(arg.value))
            # print(node.func, args, kwargs)
            result = func(*args, **kwargs)

            if isinstance(result, strategy) and self.executor.declaration is None:
                if result.default_qty_type is None:
                    result.default_qty_type = strategy.fixed
                if result.default_qty_value is None:
                    result.default_qty_value = 1
                if result.initial_capital is None:
                    result.initial_capital = 1000000

                self.executor.declaration = result
                self.executor.cash = result.initial_capital
                print(f"initial cash: {self.executor.cash}")

            if isinstance(result, strategy.entry):
                price = (
                    self.executor.sources["close"][0]
                    if result.limit is None and result.stop is None
                    else result.limit[0] or result.stop[0]
                )

                if self.executor.position_size != 0:
                    if result.direction == strategy.long and self.executor.position_size < 0:
                        self.executor.cash += 2 * self.executor.position_amount + self.executor.position_size * price
                        print(
                            f"{self.executor.current_date}: action=exit  direction=short price={price} quantity={-self.executor.position_size} cash={self.executor.cash}"
                        )
                        self.executor.position_size = 0
                        self.executor.position_amount = 0
                    elif result.direction == strategy.short and self.executor.position_size > 0:
                        self.executor.cash += self.executor.position_size * price
                        print(
                            f"{self.executor.current_date}: action=exit  direction=long  price={price} quantity={self.executor.position_size} cash={self.executor.cash}"
                        )
                        self.executor.position_size = 0
                        self.executor.position_amount = 0

                if result.qty is not None:
                    quantity = result.qty[0]
                else:
                    if self.executor.declaration.default_qty_type == strategy.fixed:
                        quantity = self.executor.declaration.default_qty_value
                    elif self.executor.declaration.default_qty_type == strategy.cash:
                        cash = self.executor.declaration.default_qty_value
                        quantity = cash // price
                    elif self.executor.declaration.default_qty_type == strategy.percent_of_equity:
                        percent = self.executor.declaration.default_qty_value / 100
                        cash = self.executor.cash * percent
                        quantity = cash // price
                    else:
                        raise ValueError()

                cash_amount = price * quantity

                if self.executor.cash > cash_amount:
                    if result.direction == strategy.long and not self.executor.position_size > 0:
                        self.executor.cash -= cash_amount
                        print(
                            f"{self.executor.current_date}: action=enter direction=long  price={price} quantity={quantity} cash={self.executor.cash}"
                        )
                        self.executor.position_size = +quantity
                        self.executor.position_amount = cash_amount
                    elif result.direction == strategy.short and not self.executor.position_size < 0:
                        self.executor.cash -= cash_amount
                        print(
                            f"{self.executor.current_date}: action=enter direction=short price={price} quantity={quantity} cash={self.executor.cash}"
                        )
                        self.executor.position_size = -quantity
                        self.executor.position_amount = cash_amount

            return result

        def visit_Assign(self, node: ast.Assign):
            if node.target not in self.executor.nodes:
                self.executor.nodes[node.target] = Series([None])

            value = self.visit(node.value)
            if value is None:
                value = Series([None])

            if (
                isinstance(value, input)
                and isinstance(node.target, ast.Name)
                and node.target.id in self.executor.inputs
            ):
                value.set(self.executor.inputs[node.target.id])

            self.executor.nodes[node.target].set(value[0])
            # print('node.target:', node.target, "node value:", value[0])

            if isinstance(node.target, ast.Name):
                self.executor.scopes[-1][node.target.id] = node.target

        def visit_Expr(self, node: ast.Expr):
            return self.visit(node.value)

        def visit_BoolOp(self, node: ast.BoolOp):
            if isinstance(node.op, ast.And):
                return [all(self.visit(value)[0] for value in node.values)]
            if isinstance(node.op, ast.Or):
                return [any(self.visit(value)[0] for value in node.values)]
            msg = f"unexpected node operator: {node.op}"
            raise ValueError(msg)

        def visit_BinOp(self, node: ast.BinOp):
            if isinstance(node.op, ast.Add):
                return [Operator.add(self.visit(node.left), self.visit(node.right))]
            if isinstance(node.op, ast.Sub):
                return [Operator.sub(self.visit(node.left), self.visit(node.right))]
            if isinstance(node.op, ast.Mult):
                return [Operator.mul(self.visit(node.left), self.visit(node.right))]
            if isinstance(node.op, ast.Div):
                return [Operator.truediv(self.visit(node.left), self.visit(node.right))]
            if isinstance(node.op, ast.Mod):
                return [Operator.mod(self.visit(node.left), self.visit(node.right))]
            msg = f"unexpected node operator: {node.op}"
            raise ValueError(msg)

        def visit_UnaryOp(self, node: ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return [not self.visit(node.operand)[0]]
            if isinstance(node.op, ast.UAdd):
                return [+self.visit(node.operand)[0]]
            if isinstance(node.op, ast.USub):
                return [-self.visit(node.operand)[0]]
            raise ValueError()

        def visit_If(self, node: ast.If):
            logic_value = self.visit(node.test)
            logic_value = Series([None]) if logic_value is None else logic_value
            if logic_value[0]:
                self.executor.scopes.append({})
                for stmt in node.body:
                    self.visit(stmt)
                self.executor.scopes.pop()
            elif node.orelse:
                self.executor.scopes.append({})
                for stmt in node.orelse:
                    self.visit(stmt)
                self.executor.scopes.pop()

        def visit_Compare(self, node: ast.Compare):  # noqa: C901, PLR0911, PLR0912
            left = self.visit(node.left)
            comparators = map(lambda x: self.visit(x), itertools.chain([node.left], node.comparators))
            comparator_pairs = list(itertools.pairwise(comparators))
            compare_ops = node.ops  # map(self.visit, node.ops)

            for op, (left, right) in zip(compare_ops, comparator_pairs, strict=True):
                if isinstance(op, ast.Eq):
                    if not Operator.eq(left, right):
                        return [False]
                elif isinstance(op, ast.NotEq):
                    if not Operator.ne(left, right):
                        return [False]
                elif isinstance(op, ast.Lt):
                    if not Operator.lt(left, right):
                        return [False]
                elif isinstance(op, ast.LtE):
                    if not Operator.le(left, right):
                        return [False]
                elif isinstance(op, ast.Gt):
                    if not Operator.gt(left, right):
                        return [False]
                elif isinstance(op, ast.GtE):
                    if not Operator.ge(left, right):
                        return [False]
                else:
                    msg = f"unexpected node operator: {op}"
                    raise ValueError(msg)
            return [True]

        def visit_Script(self, node: ast.Script):
            self.executor.scopes.append({})
            for stmt in node.body:
                self.visit(stmt)
            self.executor.scopes.pop()

    def __init__(self, script_source: str):
        self.tree = parse(script_source)
        self.visitor = self.Visitor(self)
        self.sources = {
            "close": source(),
        }
        self.builtins = {
            "strategy": strategy,
            "input": input,
            "ta": ta,
            "na": na,
        }
        self.inputs = {}
        self.declaration = None
        self.nodes = {}
        self.scopes = []
        self.cash = 0
        self.position_size = 0
        self.position_amount = 0
        self.current_date = None

    def execute(self, data: DataFrame, inputs: dict | None = None):
        if inputs:
            self.input = dict(inputs)
        for row in data.itertuples():
            self.current_date = row.Index
            self.sources["close"].add(row.Close)
            for node, values in self.nodes.items():
                values.add(None)
            self.visitor.visit(self.tree)
        net_profit_percent = round((self.cash / self.declaration.initial_capital - 1) * 100, 2)
        print(f"final cash: {self.cash} ({'+' if net_profit_percent > 0 else ''}{net_profit_percent}%)")


script_source2 = """
//@version=5
strategy("Custom BOLL Strategy", overlay=true)

// 参数
length = input( 10 ) //均线长度
width = input( 1.0 ) //布林带上轨宽度
width2 = input( 1.0 ) //布林带下轨宽度

// 计算均线
price = close

// 计算买卖信号
ma5 = ta.sma(close, 5)
ma10 = ta.sma(close, 10)

// 计算布林带
middleLine = ta.sma(price, length) // 中线
stdValue = ta.stdev(price, length)  // 标准差
scaledStd = width * stdValue
scaledStd2 = width2 * stdValue
upperLine = middleLine + scaledStd // 上轨
lowerLine = middleLine - scaledStd2 // 下轨

if ((not na(ma5)) and (not na(ma10)) and (not na(upperLine)))
    // 根据条件生成信号
    if ((ma5 > ma10) and (price > upperLine))
        strategy.entry("RsiLE", strategy.long, comment="RsiLE")
    else if (price < lowerLine)
        strategy.entry("RsiLE", strategy.short, comment="RsiSE")
"""


script_source = """
//@version=5
strategy("RSI Strategy", overlay=true)
length = input( 14 )
overSold = input( 30 )
overBought = input( 70 )
price = close
vrsi = ta.rsi(price, length)
co = ta.crossover(vrsi, overSold)
cu = ta.crossunder(vrsi, overBought)
if (not na(vrsi))
	if (co)
		strategy.entry("RsiLE", strategy.long, comment="RsiLE")
	if (cu)
		strategy.entry("RsiSE", strategy.short, comment="RsiSE")
//plot(strategy.equity, title="equity", color=color.red, linewidth=2, style=plot.style_areabr)
"""

executor = ExampleScriptExecutor(script_source2)
executor.execute(hist)
