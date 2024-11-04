from enum import Enum


class PlotDisplay(Enum):
    all = 1
    data_window = 2
    none = 3
    pane = 4
    price_scale = 5
    status_line = 6


class StrategyDirection(Enum):
    long = 1
    short = 2


class ScaleType(Enum):
    right = 1
    left = 2
    none = 3
