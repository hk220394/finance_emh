from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .synthetic import generate_synthetic_ohlcv


def _download_yf(universe: List[str], benchmark: str, start: str, end: str) -> Tuple[pd.DataFrame, pd.Series]:
	try:
		import yfinance as yf  # type: ignore
	except Exception as e:
		raise RuntimeError("yfinance not available") from e

	data = yf.download(universe, start=start, end=end, interval="1d", auto_adjust=True, progress=False, group_by="ticker")
	frames = []
	for tic in universe:
		if tic not in data or data[tic].empty:
			continue
		d = data[tic]
		d = d.rename(columns={"Close": "close", "Volume": "volume"})[["close", "volume"]]
		d["ticker"] = tic
		d.index.name = "date"
		frames.append(d)
	assets = pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()

	bench = yf.download(benchmark, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
	bench = bench.rename(columns={"Close": "close"})["close"].rename("benchmark_close")
	bench.index.name = "date"
	return assets, bench


def _from_synthetic(start: str, end: str, n_assets: int = 20):
	wide = generate_synthetic_ohlcv(start, end, n_assets=n_assets)
	# Convert wide (field, ticker) columns to long index [date, ticker]
	close_long = wide["close"].stack().rename("close")
	vol_long = wide["volume"].stack().rename("volume")
	assets = pd.concat([close_long, vol_long], axis=1).sort_index()
	assets.index.set_names(["date", "ticker"], inplace=True)
	bench = wide["close"].mean(axis=1).rename("benchmark_close")
	bench.index.name = "date"
	return assets, bench


def load_market_data(cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
	data_cfg = cfg.get("data", {})
	source = data_cfg.get("source", "yfinance")
	universe = data_cfg.get("universe")
	benchmark = data_cfg.get("benchmark", "^GSPC")
	start = data_cfg.get("start_date")
	end = data_cfg.get("end_date")

	if source == "yfinance":
		try:
			assets, bench = _download_yf(universe, benchmark, start, end)
		except Exception:
			assets, bench = _from_synthetic(start, end, n_assets=len(universe) if universe else 20)
		return assets, bench
	elif source == "synthetic":
		assets, bench = _from_synthetic(start, end, n_assets=len(universe) if universe else 20)
		return assets, bench
	else:
		raise ValueError(f"Unknown data source: {source}")

