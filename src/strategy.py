from __future__ import annotations

import numpy as np
import pandas as pd


def map_predictions_to_weights(pred_panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
	# pred_panel index: [date, ticker], columns: pred, label_excess_ret
	panel = pred_panel.copy()
	panel = panel.dropna(subset=["pred"]).copy()
	long_short = bool(cfg.get("strategy", {}).get("long_short", True))
	positions_from = cfg.get("strategy", {}).get("positions_from", "rank")
	max_pos = float(cfg.get("strategy", {}).get("max_position_per_asset", 0.05))
	gross = float(cfg.get("strategy", {}).get("gross_leverage", 1.0))
	top_k = cfg.get("strategy", {}).get("top_k")

	def per_day_weights(g: pd.DataFrame) -> pd.Series:
		pred = g["pred"].copy()
		if positions_from == "linear":
			w = pred - pred.mean()
			if w.std(ddof=0) > 0:
				w = w / w.abs().sum()
			else:
				w[:] = 0.0
		else:  # rank
			r = pred.rank(method="first")
			n = len(r)
			w = (r - (n + 1) / 2.0)
			w = w / w.abs().sum()

		if top_k is not None and isinstance(top_k, int) and top_k > 0:
			k = min(top_k, len(w) // (2 if long_short else 1))
			if k > 0:
				long_idx = pred.nlargest(k).index
				short_idx = pred.nsmallest(k).index if long_short else []
				mask = g.index.isin(long_idx.union(short_idx))
				w = w.where(mask, 0.0)

		if not long_short:
			w = w.clip(lower=0)
			if w.sum() != 0:
				w = w / w.sum()

		w = w.clip(lower=-max_pos, upper=max_pos)
		if w.abs().sum() != 0:
			w = w * (gross / w.abs().sum())
		return w

	def _apply_day(group: pd.DataFrame) -> pd.DataFrame:
		w = per_day_weights(group.droplevel(0))
		return w.to_frame(name="weight")

	weights = panel.groupby(level=0).apply(_apply_day)
	# Ensure index is [date, ticker]
	weights.index.set_names(["date", "ticker"], inplace=True)
	return weights

