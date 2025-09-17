## ML Trading Pipeline to Test Market Efficiency

This project provides an end-to-end, reproducible pipeline to build and backtest a machine learning strategy that predicts next-day excess returns and trades under explicit risk limits. It is designed to help evaluate claims around the Efficient Market Hypothesis (EMH) by measuring whether learned signals produce statistically significant risk-adjusted performance after realistic costs.

### Features
- Data ingestion from public sources (Yahoo Finance) or synthetic generation
- Daily factor engineering (momentum, volatility, liquidity proxies)
- Labels for next-day excess returns over a benchmark
- Walk-forward model training and out-of-sample predictions
- Long-short, market-neutral portfolio construction
- Risk management: target volatility and max drawdown guardrail
- Backtesting with transaction costs and turnover tracking
- Metrics report: Sharpe, Information Ratio, t-stats, drawdown, hit-rate
- Single CLI to run the end-to-end pipeline

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Quickstart (Public Data)

```bash
python main.py run --config configs/sample.yaml
```

If public data download fails (e.g., no internet), the pipeline will automatically fall back to a synthetic dataset with similar distributional properties for demonstration.

### Configuration
Edit `configs/sample.yaml` to set tickers, benchmark, date range, factor lookbacks, model, risk limits, and costs.

### EMH Framing
Under the EMH, all knowable information is already in prices; therefore, risk-adjusted alpha should not be systematically extractable. This project tests that notion by:
- Training exclusively on past data with strict walk-forward splits
- Enforcing no look-ahead by shifting features and labels appropriately
- Charging transaction costs and constraining risk
- Reporting out-of-sample performance with statistical significance

Important: Past performance in backtests is not indicative of future results. This code is for research and educational purposes only.

# Finance EMH

Finance experts often caution against trying to time the market, claiming it's irresponsible and unlikely to succeed. However, with the advent of machine learning, is it now worth attempting to time the market?

The **Efficient Market Hypothesis (EMH)** posits that all knowable information is already reflected in stock prices, making it nearly impossible to consistently outperform the market. But how true is that theory in today's data-driven landscape?

## Project Overview

This repository explores the limits of market efficiency by using a combination of public and proprietary daily market data to build predictive models. Our goal is to:

- **Predict excess returns** using machine learning
- **Design a trading strategy** that stays within a specified risk limit
- **Evaluate the validity of EMH** in the context of modern financial data

Your work will directly test the Efficient Market Hypothesis and challenge longstanding assumptions about market efficiency.

## Key Features

- Data ingestion and preprocessing from multiple market sources
- Feature engineering for financial time series
- Model training and evaluation (machine learning methods)
- Backtesting trading strategies with risk controls
- EMH hypothesis testing and analysis

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/hk220394/finance_emh.git
    ```
2. Install dependencies:
    ```bash
    # Example for Python projects
    pip install -r requirements.txt
    ```
3. Explore the notebooks and scripts for data analysis and modeling.

## Contributing

Contributions are welcome! Please open issues or pull requests to suggest improvements or report bugs.

## License

This project is licensed under the MIT License.

## Disclaimer

This repository is for research and educational purposes only. No content here should be interpreted as financial advice.