from __future__ import annotations

import pandas as pd

from src.utils import align_benchmark


def make_excess_return_labels(assets: pd.DataFrame, benchmark_close: pd.Series, cfg: dict) -> pd.DataFrame:
	# assets index: [date, ticker], columns include 'close'
	assets, benchmark_close = align_benchmark(assets, benchmark_close)
	close_wide = assets["close"].unstack()
	asset_ret = close_wide.pct_change()
	bench_ret = benchmark_close.pct_change()

	if cfg.get("labeling", {}).get("excess_over", "benchmark") == "benchmark":
		excess = asset_ret.sub(bench_ret, axis=0)
	else:
		# risk_free not provided; default to zero excess over cash
		excess = asset_ret

	# Next-day horizon
	h = int(cfg.get("labeling", {}).get("horizon_days", 1))
	future_excess = excess.shift(-h)
	labels = future_excess.stack().rename("label_excess_ret").to_frame().dropna()
	return labels

