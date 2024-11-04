
def remove_indent_per_line(multiline_string):
    # 拆分为行并计算每行开头的空白字符长度
    lines = multiline_string.splitlines()
    indent_lengths = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    # 找到最小的缩进长度
    min_indent = min(indent_lengths) if indent_lengths else 0

    # 去掉每行开头的最小缩进长度
    stripped_lines = [line[min_indent:] for line in lines if line.strip()]

    # 连接处理后的行
    result = "\n".join(stripped_lines)

    return result


if __name__ == '__main__':
    string = """
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
    print(remove_indent_per_line(string))
