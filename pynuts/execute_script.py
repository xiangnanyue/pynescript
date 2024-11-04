from __future__ import annotations

from typing import TypeAlias
from pandas import DataFrame

from pynescript.ast import parse

from .tools import remove_indent_per_line
from .expression import *
from .pine_visitor import PineVisitor
from .function import TA, IsNa

display = PlotDisplay
void: TypeAlias = None
scale = ScaleType
na = IsNa()


class ScriptExecutor:

    def __init__(self, script: str):
        self.input = None
        self.tree = parse(remove_indent_per_line(script))
        self.visitor = PineVisitor(self)
        self.sources = {
            "close": Source(),
        }
        self.builtins = {
            "strategy": Strategy,
            "input": Input,
            "ta": TA,
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


if __name__ == "__main__":
    script_source2 = """
    //@version=5
    strategy("Custom BOLL Strategy", overlay=true, default_qty_type="fixed", default_qty_value=500)
    
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
            strategy.entry("SMA", strategy.long, comment="sma plus bollinger buy")
        else if (price < lowerLine)
            strategy.entry("SMA", strategy.short, comment="sma plus bollinger sell")
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
    from .historical_data import read_data

    ticker = "TSLA"
    filename = "./examples/tsla.csv"
    hist = read_data(ticker, filename)
    executor = ScriptExecutor(script_source2)
    executor.execute(hist)
