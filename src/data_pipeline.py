"""
Data pipeline for fetching stock data and computing financial metrics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL",
    # Finance
    "JPM", "GS", "BAC",
    # Energy
    "XOM", "CVX", "COP",
    # Healthcare
    "JNJ", "PFE", "UNH",
    # Consumer
    "PG", "KO", "WMT",
]

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


def fetch_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily adjusted closing prices via yfinance."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    # yf.download returns MultiIndex columns (Price, Ticker) — extract Close
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"]
    else:
        prices = df
    # Ensure column order matches input tickers
    prices = prices[tickers]
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns, drop first NaN row, forward-fill remaining."""
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.iloc[1:]  # drop first NaN row
    log_returns = log_returns.ffill()
    return log_returns


def compute_financial_metrics(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute annualized mu, sigma (covariance), and rho (correlation)."""
    mu = returns.mean().values * 252
    sigma = returns.cov().values * 252
    # Correlation from covariance: rho_ij = sigma_ij / sqrt(sigma_ii * sigma_jj)
    std = np.sqrt(np.diag(sigma))
    rho = sigma / np.outer(std, std)
    return mu, sigma, rho


def plot_correlation_heatmap(
    rho: np.ndarray, tickers: list[str], save_path: str
) -> None:
    """Plot and save a correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        rho,
        annot=True,
        fmt=".2f",
        xticklabels=tickers,
        yticklabels=tickers,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Stock Correlation Matrix (2022-2024)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {save_path}")


if __name__ == "__main__":
    import os

    print("Fetching stock data...")
    prices = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    # print(prices)
    print(f"Prices shape: {prices.shape}")

    print("Computing log returns...")
    returns = compute_log_returns(prices)
    print(returns)
    print(f"Returns shape: {returns.shape}")

    print("Computing financial metrics...")
    mu, sigma, rho = compute_financial_metrics(returns)
    print(f"Expected returns (annualized): {mu}")
    print(f"Covariance matrix shape: {sigma.shape}")

    save_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(rho, TICKERS, save_path)
