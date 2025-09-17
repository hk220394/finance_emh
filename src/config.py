from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
	with open(path, "r") as f:
		cfg = yaml.safe_load(f)
	return cfg


def ensure_dirs(*paths: str) -> None:
	for p in paths:
		if p is None:
			continue
		os.makedirs(p, exist_ok=True)


def make_absolute(path: str, base_dir: str) -> str:
	if path is None:
		return path
	if os.path.isabs(path):
		return path
	return os.path.abspath(os.path.join(base_dir, path))


def prepare_config(config_path: str) -> Dict[str, Any]:
	config_path = os.path.abspath(config_path)
	base_dir = os.path.dirname(config_path)
	cfg = load_yaml_config(config_path)

	# Normalize output directories
	artifacts_dir = cfg.get("output", {}).get("artifacts_dir", "artifacts")
	reports_dir = cfg.get("output", {}).get("reports_dir", "reports")
	cfg["output"]["artifacts_dir"] = make_absolute(artifacts_dir, base_dir)
	cfg["output"]["reports_dir"] = make_absolute(reports_dir, base_dir)
	ensure_dirs(cfg["output"]["artifacts_dir"], cfg["output"]["reports_dir"])

	return cfg

