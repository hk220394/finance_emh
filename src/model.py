from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
	train_window_days: int
	test_window_days: int
	step_days: int


def _ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
	# Closed-form ridge: (X^T X + alpha I)^-1 X^T y
	# Add intercept by augmenting X with ones
	X_train_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_test_aug = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
	XtX = X_train_aug.T @ X_train_aug
	reg = alpha * np.eye(XtX.shape[0])
	beta = np.linalg.pinv(XtX + reg) @ X_train_aug.T @ y_train
	return X_test_aug @ beta


def walk_forward_train_predict(factors: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
	# Align available samples
	panel = factors.join(labels, how="inner")
	panel = panel.dropna()

	# Dates for walk-forward (unique, sorted)
	dates = panel.index.get_level_values(0).unique().sort_values()
	wcfg = cfg.get("model", {}).get("walk_forward", {})
	train_win = int(wcfg.get("train_window_days", 504))
	test_win = int(wcfg.get("test_window_days", 126))
	step = int(wcfg.get("step_days", 126))

	preds_frames = []
	start_idx = 0
	while True:
		train_start_idx = max(0, start_idx)
		train_end_idx = train_start_idx + train_win
		if train_end_idx >= len(dates):
			break
		test_start_idx = train_end_idx
		test_end_idx = min(len(dates), test_start_idx + test_win)  # exclusive

		train_start = dates[train_start_idx]
		train_end_exclusive = dates[train_end_idx]
		test_start = dates[test_start_idx]
		test_end_exclusive = dates[test_end_idx - 1] if test_end_idx - 1 < len(dates) else dates[-1]

		train_mask = (panel.index.get_level_values(0) >= train_start) & (panel.index.get_level_values(0) < train_end_exclusive)
		test_mask = (panel.index.get_level_values(0) >= test_start) & (panel.index.get_level_values(0) < (dates[test_end_idx] if test_end_idx < len(dates) else dates[-1] + pd.Timedelta(days=1)))

		train_df = panel.loc[train_mask]
		test_df = panel.loc[test_mask]
		if len(train_df) == 0 or len(test_df) == 0:
			break

		X_train = train_df.drop(columns=["label_excess_ret"]).values
		y_train = train_df["label_excess_ret"].values
		X_test = test_df.drop(columns=["label_excess_ret"]).values

		alpha = float(cfg.get("model", {}).get("ridge_alpha", 1.0))
		y_pred = _ridge_fit_predict(X_train, y_train, X_test, alpha=alpha)
		pred_frame = test_df[["label_excess_ret"]].copy()
		pred_frame["pred"] = y_pred
		preds_frames.append(pred_frame)

		start_idx += step

	preds = pd.concat(preds_frames).sort_index()
	return preds

