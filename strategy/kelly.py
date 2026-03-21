"""
Kelly Criterion — Position sizing based on edge and odds.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KELLY_FRACTION, MAX_POSITION_PCT


def kelly_fraction(win_prob: float, win_loss_ratio: float = 1.0,
                   fraction: float = KELLY_FRACTION) -> float:
    """Calculate fractional Kelly criterion bet size.

    Args:
        win_prob: Probability of winning (0-1)
        win_loss_ratio: Average win / average loss
        fraction: Kelly fraction (0.25 = quarter-Kelly)

    Returns:
        Fraction of bankroll to bet (0 to MAX_POSITION_PCT)
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0

    loss_prob = 1 - win_prob
    # Kelly formula: f* = (bp - q) / b
    # where b = win/loss ratio, p = win prob, q = loss prob
    b = win_loss_ratio
    full_kelly = (b * win_prob - loss_prob) / b

    if full_kelly <= 0:
        return 0.0

    sized = full_kelly * fraction
    return min(sized, MAX_POSITION_PCT)


def kelly_size(
    capital: float,
    win_prob: float,
    win_loss_ratio: float = 1.0,
    fraction: float = KELLY_FRACTION,
) -> float:
    """Calculate dollar position size using Kelly criterion."""
    frac = kelly_fraction(win_prob, win_loss_ratio, fraction)
    return round(capital * frac, 2)


def dynamic_kelly(
    win_prob: float,
    win_loss_ratio: float,
    current_drawdown: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """Kelly with drawdown-adaptive scaling.
    Reduces position size as drawdown increases.
    """
    base = kelly_fraction(win_prob, win_loss_ratio, fraction)

    # Scale down linearly as drawdown increases
    # At 0% drawdown: full size. At 20% drawdown: 0 size.
    drawdown_scalar = max(0, 1 - current_drawdown / 0.20)
    return base * drawdown_scalar
