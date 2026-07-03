"""
backtest.py
Trading-strategy backtest and risk metrics (Section 5.3, Eq. 10-14 of the paper).

Strategy:
    if predicted price(t+1) > observed price(t):  go LONG
    if predicted price(t+1) < observed price(t):  go SHORT
    else: hold no position
Transaction cost = 1 permille (0.001), as in the paper.
"""

import numpy as np


def trading_returns(y_pred, y_true_next, y_true_prev, cost=1e-3):
    """
    Eq. 10: R_{t+1} = ln(y_{t+1}/y_t) * sign(y_hat_{t+1} - y_t)  (minus transaction cost)

    y_pred:      predicted price at t+1               (array)
    y_true_next: actual price at t+1                  (array)
    y_true_prev: actual price at t (latest observed)   (array)
    """
    signal = np.sign(y_pred - y_true_prev)
    log_ret = np.log(y_true_next / y_true_prev)
    R = log_ret * signal
    # subtract transaction cost whenever a position is taken (long or short)
    R = R - cost * np.abs(signal)
    return R


def net_value(returns):
    """Eq. 11: NV_t = 1 + sum_{i=2}^t R_i"""
    nv = np.concatenate([[1.0], 1.0 + np.cumsum(returns)])
    return nv


def volatility(returns):
    """Eq. 12"""
    return float(np.std(returns))


def max_drawdown(nv):
    """Eq. 13"""
    nv = np.asarray(nv)
    running_max = np.maximum.accumulate(nv)
    drawdowns = (nv - running_max) / running_max
    return float(np.min(drawdowns)) * 100  # percent


def sharpe_ratio(returns, risk_free=0.0):
    """Eq. 14"""
    sigma = np.std(returns)
    if sigma == 0:
        return 0.0
    return float((np.mean(returns) - risk_free) / sigma)


def buy_and_hold_returns(y_true_next, y_true_prev):
    """Passive benchmark: always long."""
    return np.log(y_true_next / y_true_prev)


def backtest_summary(y_pred, y_true_next, y_true_prev, cost=1e-3):
    R = trading_returns(y_pred, y_true_next, y_true_prev, cost=cost)
    nv = net_value(R)
    total_return = (nv[-1] - 1) * 100
    return {
        "Return(%)": total_return,
        "Volatility(%)": volatility(R) * 100,
        "MaxDrawdown(%)": max_drawdown(nv),
        "Sharpe": sharpe_ratio(R),
        "net_value": nv,
        "returns": R,
    }
