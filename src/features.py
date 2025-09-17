from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import zscore_by_date, winsorize_series


def compute_daily_returns(panel_prices: pd.DataFrame) -> pd.DataFrame:
	close = panel_prices["close"].copy()
	rets = close.groupby(level=1).pct_change().rename("ret")
	return rets.to_frame()


def engineer_factors(assets: pd.DataFrame, cfg: dict) -> pd.DataFrame:
	# assets index: [date, ticker], columns: close, volume
	assets = assets.sort_index()
	close = assets["close"].unstack()
	volume = assets["volume"].unstack()

	lookbacks = cfg.get("features", {}).get("momentum_lookbacks", [5, 10, 21, 63])
	vol_window = cfg.get("features", {}).get("volatility_window", 21)
	vol_z_win = cfg.get("features", {}).get("volume_zscore_window", 21)

	# Momentum: past return over lookback
	factor_frames = []
	for lb in lookbacks:
		mom = close.pct_change(lb)
		mom = mom.stack().to_frame(name=f"mom_{lb}")
		factor_frames.append(mom)

	# Volatility: rolling std of daily returns
	daily_rets = close.pct_change()
	vol = daily_rets.rolling(vol_window).std().stack().to_frame(name=f"vol_{vol_window}")
	factor_frames.append(vol)

	# Liquidity proxy: z-scored volume
	vol_z = ((volume / volume.rolling(vol_z_win).mean()) - 1).stack().to_frame(name="vol_z")
	factor_frames.append(vol_z)

	factors = pd.concat(factor_frames, axis=1).dropna()

	# Winsorize per factor then z-score cross-sectionally by date
	for c in factors.columns:
		factors[c] = factors.groupby(level=0)[c].transform(lambda s: winsorize_series(s, 0.01))
	factors = zscore_by_date(factors, list(factors.columns))
	return factors.sort_index()

