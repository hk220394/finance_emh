from __future__ import annotations

import pandas as pd


def simulate_portfolio(weights: pd.DataFrame, asset_returns: pd.DataFrame, bps_per_trade: float = 5.0) -> pd.DataFrame:
	# weights index: [date, ticker], column: weight
	# asset_returns wide: date x ticker
	w = weights["weight"].unstack().fillna(0)
	w_shift = w.shift().fillna(0)
	turnover = (w - w_shift).abs().sum(axis=1)
	trading_cost = turnover * (bps_per_trade / 10000.0)
	ret = (w_shift * asset_returns).sum(axis=1) - trading_cost
	out = pd.DataFrame({
		"strategy_ret": ret,
		"turnover": turnover,
		"trading_cost": trading_cost,
	})
	return out

