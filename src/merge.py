"""
Merge distributed sub-portfolio results into a global portfolio.

Maps cluster-local bitstrings back to original stock ordering,
evaluates the full portfolio (including cross-cluster covariance),
and optionally refines via greedy local search.
"""

import numpy as np


def merge_subportfolios(
    sub_results: list[dict],
    subproblems: list[dict],
    n_total: int,
) -> np.ndarray:
    """
    Map cluster bitstrings back to original stock ordering.

    Parameters
    ----------
    sub_results  : list of dicts with 'cluster_id' and 'best_bitstring'.
    subproblems  : list of dicts with 'cluster_id' and 'stock_indices'.
    n_total      : total number of stocks.

    Returns
    -------
    x_global : (n_total,) binary array in original stock ordering.
    """
    x_global = np.zeros(n_total)

    sp_map = {sp["cluster_id"]: sp for sp in subproblems}

    for res in sub_results:
        cid = res["cluster_id"]
        sp = sp_map[cid]
        bitstring = res["best_bitstring"]
        indices = sp["stock_indices"]

        for local_idx, global_idx in enumerate(indices):
            x_global[global_idx] = float(bitstring[local_idx])

    return x_global


def evaluate_full_portfolio(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float,
    rf: float = 0.045,
) -> dict:
    """
    Evaluate a portfolio using the FULL covariance matrix (cross-cluster terms included).

    Parameters
    ----------
    x     : (n,) binary selection vector.
    mu    : (n,) expected return vector.
    sigma : (n, n) full covariance matrix.
    q     : risk-aversion parameter.
    rf    : risk-free rate for Sharpe ratio.

    Returns
    -------
    dict with keys: selected_indices, n_selected, expected_return, risk,
                    sharpe_ratio, qubo_objective, bitstring
    """
    selected = np.where(x == 1)[0].tolist()
    n_selected = int(x.sum())

    expected_return = float(mu @ x)
    portfolio_var = float(x @ sigma @ x)
    risk = float(np.sqrt(portfolio_var)) if portfolio_var > 0 else 0.0
    sharpe = (expected_return - rf) / risk if risk > 1e-10 else 0.0

    markowitz_obj = -expected_return + q * portfolio_var

    bitstring = "".join(str(int(b)) for b in x)

    return {
        "selected_indices": selected,
        "n_selected": n_selected,
        "expected_return": expected_return,
        "risk": risk,
        "sharpe_ratio": sharpe,
        "markowitz_obj": markowitz_obj,
        "bitstring": bitstring,
    }


def local_search(
    x: np.ndarray,
    Q: np.ndarray,
    budget: int,
    max_swaps: int = 100,
) -> np.ndarray:
    """
    Greedy swap-one-in/swap-one-out local search to improve QUBO objective.

    At each step, try all (swap_out, swap_in) pairs and accept the best
    improving swap. Stop when no improving swap exists or max_swaps reached.

    Parameters
    ----------
    x         : (n,) initial binary solution.
    Q         : (n, n) FULL QUBO matrix (15x15 for global problem).
    budget    : required sum(x).
    max_swaps : maximum number of swap iterations.

    Returns
    -------
    x_improved : (n,) binary solution after local search.
    """
    x = x.copy()
    n = len(x)
    current_obj = float(x @ Q @ x)

    for _ in range(max_swaps):
        selected = [i for i in range(n) if x[i] == 1]
        unselected = [i for i in range(n) if x[i] == 0]

        best_delta = 0.0
        best_swap = None

        for i_out in selected:
            for i_in in unselected:
                x[i_out] = 0
                x[i_in] = 1
                new_obj = float(x @ Q @ x)
                delta = new_obj - current_obj
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (i_out, i_in)
                x[i_out] = 1
                x[i_in] = 0

        if best_swap is None:
            break

        i_out, i_in = best_swap
        x[i_out] = 0
        x[i_in] = 1
        current_obj += best_delta

    return x
