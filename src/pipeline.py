from __future__ import annotations

import os
import json
from typing import Tuple

import pandas as pd

from src.config import prepare_config
from src.data import load_market_data
from src.features import engineer_factors
from src.labeling import make_excess_return_labels
from src.model import walk_forward_train_predict
from src.strategy import map_predictions_to_weights
from src.risk import apply_target_volatility, enforce_drawdown_limit
from src.backtest import simulate_portfolio
from src.metrics import compute_metrics


def run_pipeline(config_path: str) -> Tuple[pd.DataFrame, dict]:
	cfg = prepare_config(config_path)
	art_dir = cfg["output"]["artifacts_dir"]
	rep_dir = cfg["output"]["reports_dir"]

	# 1) Load data
	assets, bench_close = load_market_data(cfg)
	if cfg.get("output", {}).get("save_intermediate", True):
		try:
			assets.to_parquet(os.path.join(art_dir, "assets.parquet"))
		except Exception:
			assets.to_csv(os.path.join(art_dir, "assets.csv"))
		try:
			bench_close.to_frame().to_parquet(os.path.join(art_dir, "benchmark.parquet"))
		except Exception:
			bench_close.to_frame().to_csv(os.path.join(art_dir, "benchmark.csv"))

	# 2) Features
	factors = engineer_factors(assets, cfg)
	# 3) Labels
	labels = make_excess_return_labels(assets, bench_close, cfg)
	# 4) Model: walk-forward predictions
	preds = walk_forward_train_predict(factors, labels, cfg)
	# 5) Map predictions to positions
	weights = map_predictions_to_weights(preds, cfg)

	# Asset returns (wide) for backtest
	close_wide = assets["close"].unstack()
	asset_returns = close_wide.pct_change()

	# 6) Risk: target vol and drawdown guardrail
	weights_tv = apply_target_volatility(weights, asset_returns, cfg)
	weights_rd = enforce_drawdown_limit(weights_tv, asset_returns, cfg)

	# 7) Backtest with costs
	bps = float(cfg.get("costs", {}).get("bps_per_trade", 5))
	bt = simulate_portfolio(weights_rd, asset_returns, bps_per_trade=bps)

	# 8) Metrics & report
	metrics = compute_metrics(bt["strategy_ret"])

	if cfg.get("output", {}).get("save_intermediate", True):
		for name, df in [
			("factors", factors),
			("labels", labels),
			("preds", preds),
			("weights", weights_rd),
			("backtest", bt),
		]:
			try:
				df.to_parquet(os.path.join(art_dir, f"{name}.parquet"))
			except Exception:
				df.to_csv(os.path.join(art_dir, f"{name}.csv"))

	with open(os.path.join(rep_dir, "metrics.json"), "w") as f:
		json.dump(metrics, f, indent=2)

	print("Metrics:", json.dumps(metrics, indent=2))
	return bt, metrics

