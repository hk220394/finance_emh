from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(series: pd.Series, trading_days: int = 252) -> dict:
	rets = series.dropna()
	mu = rets.mean() * trading_days
	sd = rets.std(ddof=0) * np.sqrt(trading_days)
	sharpe = mu / sd if sd != 0 else 0.0
	cum = (1 + rets).cumprod()
	roll_max = cum.cummax()
	dd = (cum / roll_max) - 1.0
	max_dd = dd.min()
	# t-stat of mean daily returns
	t_stat = (rets.mean() / (rets.std(ddof=1) / np.sqrt(len(rets)))) if len(rets) > 1 and rets.std(ddof=1) != 0 else 0.0
	return {
		"ann_return": mu,
		"ann_vol": sd,
		"sharpe": sharpe,
		"max_drawdown": abs(max_dd),
		"t_stat_mean": t_stat,
		"num_days": int(len(rets)),
	}

