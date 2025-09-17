from __future__ import annotations

import numpy as np
import pandas as pd


def to_panel(df: pd.DataFrame) -> pd.DataFrame:
	if not isinstance(df.index, pd.MultiIndex):
		raise ValueError("Expected MultiIndex [date, asset]")
	return df.sort_index()


def winsorize_series(s: pd.Series, limits: float = 0.01) -> pd.Series:
	lo = s.quantile(limits)
	hi = s.quantile(1 - limits)
	return s.clip(lower=lo, upper=hi)


def zscore_by_date(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
	def _z(g: pd.DataFrame) -> pd.DataFrame:
		for c in cols:
			vals = g[c]
			mu = vals.mean()
			sd = vals.std(ddof=0)
			if sd == 0 or np.isnan(sd):
				g[c] = 0.0
			else:
				g[c] = (vals - mu) / sd
		return g
	return panel.groupby(level=0, group_keys=False).apply(_z)


def align_benchmark(assets: pd.DataFrame, bench: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
	# Align by date index intersection and forward-fill for missing
	idx = assets.index.get_level_values(0).unique().intersection(bench.index)
	assets = assets.loc[idx]
	bench = bench.loc[idx]
	return assets, bench


def annualize_vol(daily_vol: float, trading_days: int = 252) -> float:
	return daily_vol * np.sqrt(trading_days)


def daily_from_annual(annual_vol: float, trading_days: int = 252) -> float:
	return annual_vol / np.sqrt(trading_days)

