# Distributed Portfolio Optimization via Divide-and-Conquer QAOA on CUDA-Q

This project implements a divide-and-conquer approach to Markowitz portfolio optimization using the Quantum Approximate Optimization Algorithm (QAOA). It fetches real S&P 500 stock data, formulates the mean-variance optimization as a QUBO problem, clusters correlated stocks into smaller subproblems, and solves each subproblem with QAOA. The architecture targets the NVIDIA CUDA-Q MQPU backend for distributed multi-GPU execution, using cudaq.kernel, cudaq.observe, and cudaq.sample as the primary quantum runtime. A numpy statevector simulator is retained as a fallback for environments without CUDA-Q.

## Current Progress

Phases 0 through 4 are complete:
- **Phase 0**: Project scaffold and dependency setup
- **Phase 2**: Data pipeline (yfinance fetch, log returns, covariance/correlation) and QUBO formulation (Markowitz to QUBO to Ising mapping with brute-force verification)
- **Phase 3**: Hierarchical stock clustering (Ward linkage on correlation distance), proportional budget allocation, per-cluster subproblem generation
- **Phase 4**: Cold-start QAOA implemented with CUDA-Q kernels and validated on a 3-qubit energy sector cluster

## Setup

```bash
pip install -r requirements.txt
```

CUDA-Q must be installed separately (`pip install cudaq`). For GPU-accelerated execution, ensure NVIDIA drivers and CUDA toolkit are installed. See the [NVIDIA CUDA-Q documentation](https://nvidia.github.io/cuda-quantum/) for details. The qpp-cpu simulator works without GPU hardware, and a numpy statevector fallback is available when cudaq is not installed.

## Project Structure

```
src/
├── __init__.py          # Package init
├── data_pipeline.py     # Stock data fetching, log returns, covariance, correlation heatmap
├── qubo.py              # Markowitz QUBO formulation, QUBO-to-Ising conversion, cudaq Hamiltonian builder
├── clustering.py        # Hierarchical/spectral clustering, dendrogram, subproblem generation
└── qaoa_cold.py         # Cold-start QAOA via cudaq.kernel/observe/sample, with numpy fallback
```

## Usage

Fetch data and compute financial metrics:

```python
from src.data_pipeline import fetch_stock_data, compute_log_returns, compute_financial_metrics, TICKERS, START_DATE, END_DATE

prices = fetch_stock_data(TICKERS, START_DATE, END_DATE)
returns = compute_log_returns(prices)
mu, sigma, rho = compute_financial_metrics(returns)
```

Cluster stocks and build subproblems:

```python
from src.clustering import cluster_stocks, build_subproblems

labels = cluster_stocks(rho, n_clusters=4)
subproblems = build_subproblems(mu, sigma, labels, budget=6, q=0.5, penalty=10.0, tickers=TICKERS)
```

Run QAOA on a cluster subproblem:

```python
from src.qaoa_cold import run_qaoa, evaluate_portfolio

sp = subproblems[0]
result = run_qaoa(sp["qubo"], qubit_count=sp["qubo"].shape[0], layer_count=2)
portfolio = evaluate_portfolio(result["best_bitstring"], sp["mu"], sp["sigma"], q=0.5)
print(portfolio)
```

Run tests:

```bash
python -m pytest tests/
```
