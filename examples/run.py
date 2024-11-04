import os
import sys

# 添加当前目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# 如果没有安装 pynuts：添加路径到 sys.path
sys.path.append(os.path.join(current_dir, '..'))

from pynuts.execute_script import ScriptExecutor
from pynuts.historical_data import read_data

if __name__ == "__main__":
    from strategy_scripts import _RSI_strategy
    ticker = "TSLA"
    filename = "./tsla.csv"
    hist = read_data(ticker, filename)
    executor = ScriptExecutor(_RSI_strategy)
    executor.execute(hist)
