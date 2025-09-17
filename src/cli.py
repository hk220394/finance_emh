import os
from typing import Optional

import typer

from src.pipeline import run_pipeline


app = typer.Typer(add_completion=False, help="ML trading pipeline CLI")


@app.command()
def run(
	config: str = typer.Option(..., "--config", help="Path to YAML config"),
):
	"""Run the end-to-end pipeline: ingest -> features -> labels -> train -> backtest -> report."""
	if not os.path.isabs(config):
		config = os.path.abspath(config)
	run_pipeline(config_path=config)


@app.command()
def make_synthetic(
	output: str = typer.Option("data/raw/synthetic.parquet", "--output", help="Path to save synthetic data"),
	start_date: str = typer.Option("2018-01-01", help="Start date (YYYY-MM-DD)"),
	end_date: str = typer.Option("2022-12-31", help="End date (YYYY-MM-DD)"),
):
	"""Generate and save a synthetic daily OHLCV dataset."""
	from src.data.synthetic import generate_synthetic_ohlcv
	import pandas as pd
	os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
	df = generate_synthetic_ohlcv(start_date=start_date, end_date=end_date)
	if output.endswith(".parquet"):
		try:
			import pyarrow  # type: ignore
			df.to_parquet(output)
		except Exception:
			# Fallback to CSV if parquet engine missing
			csv_path = output.rsplit(".", 1)[0] + ".csv"
			df.to_csv(csv_path)
			print(f"Saved synthetic data to {csv_path}")
	else:
		df.to_csv(output)
	print(f"Saved synthetic data to {output}")

