from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import daily_from_annual


def apply_target_volatility(weights: pd.DataFrame, asset_returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
	# weights index: [date, ticker], column: weight
	# asset_returns wide: date x ticker
	vol_win = int(cfg.get("risk", {}).get("vol_est_window_days", 63))
	target_ann = float(cfg.get("risk", {}).get("target_vol_annual", 0.15))
	target_daily = daily_from_annual(target_ann)

	w = weights["weight"].unstack()
	r = asset_returns
	port_ret = (w.shift().fillna(0) * r).sum(axis=1)
	daily_vol = port_ret.rolling(vol_win).std()
	scale = (target_daily / daily_vol).clip(upper=5.0)
	scale = scale.reindex(w.index).fillna(1.0)
	w_scaled = (w.T * scale).T
	return w_scaled.stack().to_frame(name="weight")


def enforce_drawdown_limit(weights: pd.DataFrame, asset_returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
	max_dd = float(cfg.get("risk", {}).get("max_drawdown", 0.2))
	w = weights["weight"].unstack().fillna(0)
	r = asset_returns
	port_ret = (w.shift().fillna(0) * r).sum(axis=1)
	cum = (1 + port_ret).cumprod()
	roll_max = cum.cummax()
	dd = 1 - cum / roll_max
	flat = dd > max_dd
	w.loc[flat, :] = 0.0
	return w.stack().to_frame(name="weight")

