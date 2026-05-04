# Distributed Portfolio Optimization via Divide-and-Conquer QAOA on CUDA-Q

This project implements a divide-and-conquer approach to Markowitz portfolio
optimization using the Quantum Approximate Optimization Algorithm (QAOA).
It fetches real S&P 500 data, formulates the mean-variance problem as a QUBO,
clusters correlated stocks into smaller subproblems, solves each subproblem
with QAOA on a separate (simulated) QPU, and merges the per-cluster solutions
into a global portfolio. Both **cold-start** and **warm-start** QAOA pipelines
are supported, the latter initialised from a Goemans-Williamson SDP relaxation
and using the Egger et al. modified mixer.

The runtime targets the NVIDIA CUDA-Q **MQPU** backend so that the
per-cluster sampling step runs in parallel across multiple GPUs (validated on
4 × NVIDIA L40S, 46 GB each). When the MQPU target is unavailable the code
falls back to the `qpp-cpu` simulator with sequential dispatch — same math,
slower wall-clock.

## Pipeline

```
yfinance prices
   └─ log returns ─ μ, Σ, ρ
       └─ Markowitz QUBO ─ Ising mapping
           └─ Hierarchical clustering (Ward linkage on √(2(1-ρ)))
               ├─ per-cluster sub-QUBO (proportional budget)
               │   ├─ Cold path:  QAOA (Hadamard init, Rx mixer)
               │   └─ Warm path:  GW SDP → Ry init → Egger mixer
               └─ MQPU dispatch (cudaq.sample_async, qpu_id round-robin)
                   └─ Merge sub-bitstrings → global x
                       └─ Greedy 1-swap local search (full Σ)
                           └─ Final portfolio (return, risk, Sharpe, AR)
```

## What's implemented

- **Data pipeline** — yfinance fetch for 15 stocks across 5 sectors, log
  returns, annualised μ/Σ/ρ, correlation/covariance heatmaps.
- **QUBO formulation** — Markowitz mean-variance + budget penalty,
  closed-form QUBO ↔ Ising mapping, brute-force verifier.
- **Clustering** — Ward-linkage hierarchical clustering on the correlation
  distance, dendrogram + clustered heatmap, proportional budget allocation
  with largest-remainder rounding, per-cluster sub-QUBO/Ising builder, and a
  cross-cluster covariance loss metric.
- **Cold-start QAOA** (`src/qaoa_cold.py`) — `@cudaq.kernel` circuit
  (H init → ZZ via CNOT-Rz-CNOT → Rx mixer), `cudaq.observe` objective,
  COBYLA optimiser, `cudaq.sample` final readout. Includes a
  `_sample_result_to_dict` workaround for the cudaq 0.14.0 bug where
  `dict(SampleResult)` fails for ≥ 3 qubits.
- **Goemans-Williamson SDP** (`src/gw.py`) — pure-classical baseline:
  CVXPY/SCS solves the rank-relaxed PSD program, then random-hyperplane
  rounding recovers a binary solution. The auxiliary-spin trick encodes the
  linear `h_i` term as an interaction with a clamped `z_0 = +1`.
- **Warm-start QAOA** (`src/qaoa_warm.py`) — Egger et al. (Quantum 2021):
  per-qubit Ry rotations bias the initial state toward the GW solution
  (`θ = 2 arcsin √(1-ε)` for selected qubits, `2 arcsin √ε` otherwise),
  and the mixer becomes `Ry(θ)·Rz(-2β)·Ry(-θ)` so that the warm-start state
  remains a +1 eigenstate at β = 0.
- **Distributed dispatch** (`src/distributed.py`) — `setup_mqpu()` picks
  `nvidia/mqpu` if available, else `qpp-cpu`. `run_distributed_cold_qaoa`
  and `run_distributed_ws_qaoa` optimise sequentially per cluster, then fan
  out the final sampling across QPUs via `cudaq.sample_async` with
  round-robin `qpu_id` assignment.
- **Merge & local search** (`src/merge.py`) — `merge_subportfolios` maps
  cluster-local bitstrings back to global stock ordering, `evaluate_full_portfolio`
  scores against the *full* Σ (cross-cluster terms included), and `local_search`
  performs greedy 1-in/1-out swaps under the global QUBO to repair losses
  introduced by the block-diagonal decomposition.

## Project Structure

```
src/
├── __init__.py            # Package init
├── data_pipeline.py       # yfinance fetch, log returns, μ/Σ/ρ, correlation heatmap
├── qubo.py                # Markowitz QUBO, QUBO↔Ising, cudaq SpinOperator builder
├── clustering.py          # Ward linkage, dendrogram, sub-QUBO builder, cross-cluster loss
├── qaoa_cold.py           # Cold-start QAOA (CUDA-Q kernel + COBYLA)
├── gw.py                  # GW SDP relaxation + random-hyperplane rounding
├── qaoa_warm.py           # Warm-start QAOA (GW init + Egger mixer)
├── distributed.py         # MQPU setup + distributed cold/warm dispatch
└── merge.py               # Sub-portfolio merge + greedy local search

notebooks/
└── demo.ipynb             # End-to-end walkthrough (14 sections, all figures)

tests/
└── test_qubo.py           # 4-stock toy QUBO ↔ Ising equivalence test

results/
└── figures/               # Output PNGs (heatmaps, dendrogram, convergence curves)
```

## Setup

```bash
pip install -r requirements.txt
pip install cudaq            # required — no numpy fallback in this build
```

For multi-GPU MQPU execution you need NVIDIA drivers and the CUDA toolkit
matching your `cudaq` build (validated against cudaq 0.14.0). Without GPUs
the code automatically falls back to the `qpp-cpu` simulator. See the
[NVIDIA CUDA-Q docs](https://nvidia.github.io/cuda-quantum/) for backend
details.

## Usage

Fetch data and compute financial metrics:

```python
from src.data_pipeline import (
    fetch_stock_data, compute_log_returns, compute_financial_metrics,
    TICKERS, START_DATE, END_DATE,
)

prices = fetch_stock_data(TICKERS, START_DATE, END_DATE)
returns = compute_log_returns(prices)
mu, sigma, rho = compute_financial_metrics(returns)
```

Cluster stocks and build per-cluster subproblems:

```python
from src.clustering import cluster_stocks, build_subproblems, compute_cross_cluster_loss

labels = cluster_stocks(rho, n_clusters=4)
subproblems = build_subproblems(mu, sigma, labels, budget=6, q=0.5, penalty=10.0, tickers=TICKERS)
loss = compute_cross_cluster_loss(sigma, labels)
```

Distributed cold-start QAOA across all clusters:

```python
from src.distributed import setup_mqpu, run_distributed_cold_qaoa

info = setup_mqpu()                        # picks nvidia/mqpu if available
cold_results = run_distributed_cold_qaoa(subproblems, layer_count=2, shots=10000)
```

Distributed warm-start QAOA (GW + Egger mixer):

```python
from src.distributed import run_distributed_ws_qaoa

warm_results = run_distributed_ws_qaoa(
    subproblems, layer_count=2, shots=10000, epsilon=0.1,
)
```

Merge sub-portfolios into a global selection and refine with local search:

```python
from src.merge import merge_subportfolios, evaluate_full_portfolio, local_search
from src.qubo import build_qubo

x_global = merge_subportfolios(warm_results, subproblems, n_total=len(TICKERS))
Q_full = build_qubo(mu, sigma, q=0.5, budget=6, penalty=10.0)
x_refined = local_search(x_global, Q_full, budget=6)

portfolio = evaluate_full_portfolio(x_refined, mu, sigma, q=0.5)
print(portfolio)
```

Run a single-cluster cold-start QAOA directly:

```python
from src.qaoa_cold import run_qaoa, evaluate_portfolio

sp = subproblems[0]
result = run_qaoa(sp["qubo"], qubit_count=sp["qubit_count"], layer_count=2)
portfolio = evaluate_portfolio(result["best_bitstring"], sp["mu"], sp["sigma"], q=0.5)
```

Run tests:

```bash
python -m pytest tests/
```

## Notebook

`notebooks/demo.ipynb` is the canonical end-to-end walkthrough. It is
organised into 14 sections covering: introduction, Markowitz formulation,
data pipeline, QUBO/Ising mapping, clustering with cross-cluster loss
visualisation, distributed cold-start QAOA, sub-portfolio merging,
Goemans-Williamson, warm-start QAOA, distributed warm-start, a five-method
comparison (brute force vs GW vs cold vs warm vs warm + local search),
discussion, and references.

> **Note on plotting.** `data_pipeline.py` forces the matplotlib `Agg`
> backend, which prevents `plt.show()` from rendering inline. The notebook
> uses the `fig.savefig(...) → plt.close(fig) → IPython.display.Image(...)`
> pattern throughout so figures display correctly under any kernel.

## Results (15-stock S&P 500 universe, budget = 6)

After local-search refinement, both the cold and warm distributed pipelines
recover the brute-force optimum `[AAPL, JPM, GS, XOM, KO, WMT]` with
approximation ratio 1.000 and Sharpe ≈ 0.93. The warm-start pipeline
converges in fewer optimiser iterations on each sub-problem and is more
robust to the cross-cluster covariance loss (~61 % for the 4-cluster Ward
partition) before local search is applied.

## References

Detailed citations are in Section 14 of the notebook. Key sources:

- Buonaiuto et al., *Scientific Reports* (2023) — Markowitz → QUBO mapping.
- Egger et al., *Quantum* 5, 479 (2021) — warm-start QAOA, modified mixer.
- Goemans & Williamson, *J. ACM* (1995) — SDP relaxation + hyperplane rounding.
- NVIDIA CUDA-Q documentation — MQPU backend, `sample_async`.

