from __future__ import annotations

from pathlib import Path

import os
import pandas as pd
import yfinance as yf


def download_data(ticker, filename):
    tsla = yf.Ticker(ticker)
    hist = tsla.history(period="max", interval="1d")
    hist.to_csv(filename)


def read_data(ticker, filename):
    if not os.path.exists(filename):
        download_data(ticker=ticker, filename=filename)
    hist = None
    if Path(filename).exists():
        hist = pd.read_csv(filename, index_col=0, parse_dates=True)
    return hist


if __name__ == "__main__":
    ticker = "TSLA"
    filename = "../examples/tsla.csv"
    download_data(ticker, filename)
