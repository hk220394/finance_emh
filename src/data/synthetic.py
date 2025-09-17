from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
	start_date: str = "2018-01-01",
	end_date: str = "2022-12-31",
	n_assets: int = 20,
	seed: int = 42,
) -> pd.DataFrame:
	"""Create a synthetic OHLCV panel with modest cross-sectional correlation and volatility clustering.

	Returns a DataFrame indexed by date with columns as a MultiIndex (field, ticker): close, volume.
	"""
	rng = np.random.default_rng(seed)
	dates = pd.bdate_range(start_date, end_date, freq="C")
	assets = [f"SYN{i:03d}" for i in range(n_assets)]

	# Simulate returns with a simple AR(1) market factor + idiosyncratic noise
	T = len(dates)
	market_shock = rng.normal(0, 0.01, size=T)
	market_ret = np.zeros(T)
	for t in range(1, T):
		market_ret[t] = 0.1 * market_ret[t - 1] + market_shock[t]

	panel_close = {}
	panel_vol = {}
	for a in assets:
		idio = rng.normal(0, 0.015, size=T)
		rets = market_ret + idio
		price = 100 * np.cumprod(1 + rets)
		volume = rng.lognormal(mean=12.0, sigma=0.5, size=T)
		panel_close[("close", a)] = price
		panel_vol[("volume", a)] = volume

	close_df = pd.DataFrame(panel_close, index=dates)
	vol_df = pd.DataFrame(panel_vol, index=dates)
	data = pd.concat([close_df, vol_df], axis=1)
	data.sort_index(axis=1, inplace=True)
	return data

