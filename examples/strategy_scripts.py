_Custom_Bollinger_strategy = """
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

_RSI_strategy = """
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
