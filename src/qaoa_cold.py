"""
Standard cold-start QAOA for portfolio optimization.

Primary implementation uses CUDA-Q (cudaq.kernel, cudaq.observe, cudaq.sample).
Falls back to a numpy statevector simulator when cudaq is unavailable.
"""

import numpy as np
from scipy.optimize import minimize as scipy_minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qubo import qubo_to_ising
#from src.qubo import qubo_to_ising

try:
    import cudaq
    from cudaq import spin
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ising term extraction (pure math — used by both backends)
# ---------------------------------------------------------------------------

def extract_ising_terms(Q: np.ndarray) -> dict:
    """
    Parse QUBO matrix into Ising ZZ and single-Z interaction terms.

    Returns dict with:
        edges_src, edges_tgt, coeffs       – ZZ terms (i < j)
        single_qubit_indices, single_qubit_coeffs – Z terms
        offset   – constant from QUBO-to-Ising mapping
        n_qubits – number of qubits
    """
    J, h, offset = qubo_to_ising(Q)
    n = Q.shape[0]

    edges_src, edges_tgt, coeffs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-15:
                edges_src.append(i)
                edges_tgt.append(j)
                coeffs.append(J[i, j])

    sq_indices, sq_coeffs = [], []
    for i in range(n):
        if abs(h[i]) > 1e-15:
            sq_indices.append(i)
            sq_coeffs.append(h[i])

    return {
        "edges_src": np.array(edges_src, dtype=int),
        "edges_tgt": np.array(edges_tgt, dtype=int),
        "coeffs": np.array(coeffs),
        "single_qubit_indices": np.array(sq_indices, dtype=int),
        "single_qubit_coeffs": np.array(sq_coeffs),
        "offset": offset,
        "n_qubits": n,
    }


# ---------------------------------------------------------------------------
# CUDA-Q QAOA kernel and driver
# ---------------------------------------------------------------------------

if CUDAQ_AVAILABLE:

    @cudaq.kernel
    def kernel_qaoa(qubit_count: int, layer_count: int,
                    thetas: list[float],
                    edges_src: list[int], edges_tgt: list[int],
                    coeffs: list[float],
                    sq_indices: list[int], sq_coeffs: list[float]):
        qubits = cudaq.qvector(qubit_count)
        # Initial state: uniform superposition
        h(qubits)
        # QAOA layers
        for p in range(layer_count):
            gamma = thetas[2 * p]
            beta = thetas[2 * p + 1]
            # Problem unitary: ZZ terms via CNOT-Rz-CNOT
            for k in range(len(edges_src)):
                x.ctrl(qubits[edges_src[k]], qubits[edges_tgt[k]])
                rz(2.0 * gamma * coeffs[k], qubits[edges_tgt[k]])
                x.ctrl(qubits[edges_src[k]], qubits[edges_tgt[k]])
            # Problem unitary: single Z terms
            for k in range(len(sq_indices)):
                rz(2.0 * gamma * sq_coeffs[k], qubits[sq_indices[k]])
            # Mixer: Rx on all qubits
            for i in range(qubit_count):
                rx(2.0 * beta, qubits[i])


def _sample_result_to_dict(counts, qubit_count: int) -> dict[str, int]:
    """Convert cudaq SampleResult to {bitstring: count} dict safely."""
    result = {}
    for i in range(1 << qubit_count):
        bs = format(i, f"0{qubit_count}b")
        c = counts.count(bs)
        if c > 0:
            result[bs] = c
    return result


def _cudaq_run_qaoa(
    Q: np.ndarray,
    qubit_count: int,
    layer_count: int,
    seed: int = 42,
    shots: int = 10000,
    maxiter: int = 200,
) -> dict:
    """Run cold-start QAOA using cudaq.observe and cudaq.sample."""
    from src.qubo import qubo_to_cudaq_hamiltonian

    hamiltonian = qubo_to_cudaq_hamiltonian(Q)
    ising = extract_ising_terms(Q)

    edges_src = ising["edges_src"].tolist()
    edges_tgt = ising["edges_tgt"].tolist()
    coeffs = ising["coeffs"].tolist()
    sq_indices = ising["single_qubit_indices"].tolist()
    sq_coeffs = ising["single_qubit_coeffs"].tolist()

    cudaq.set_random_seed(seed)
    np.random.seed(seed)

    n_params = 2 * layer_count
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.initial_parameters = np.random.uniform(0, np.pi, size=n_params).tolist()
    optimizer.max_iterations = maxiter

    convergence = []

    def objective(thetas):
        exp_val = cudaq.observe(
            kernel_qaoa, hamiltonian,
            qubit_count, layer_count, thetas,
            edges_src, edges_tgt, coeffs,
            sq_indices, sq_coeffs
        ).expectation()
        convergence.append(exp_val)
        return exp_val

    optimal_val, optimal_params = optimizer.optimize(
        dimensions=n_params, function=objective)

    counts = cudaq.sample(
        kernel_qaoa,
        qubit_count, layer_count,
        optimal_params,
        edges_src, edges_tgt, coeffs,
        sq_indices, sq_coeffs,
        shots_count=shots)

    counts_dict = _sample_result_to_dict(counts, qubit_count)
    # cudaq returns MSB-first bitstrings; convert to LSB-first for consistency
    counts_lsb = {bs[::-1]: c for bs, c in counts_dict.items()}
    best_bitstring = max(counts_lsb, key=counts_lsb.get)

    return {
        "optimal_energy": optimal_val,
        "optimal_params": list(optimal_params),
        "best_bitstring": best_bitstring,
        "counts": counts_lsb,
        "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# Numpy statevector fallback
# ---------------------------------------------------------------------------

def _precompute_energies(ising_terms: dict) -> np.ndarray:
    """Compute Ising energy (without offset) for every computational basis state."""
    n = ising_terms["n_qubits"]
    N = 1 << n
    energies = np.zeros(N)

    for k in range(N):
        z = np.array([1 - 2 * ((k >> i) & 1) for i in range(n)], dtype=float)
        for s, t, c in zip(
            ising_terms["edges_src"],
            ising_terms["edges_tgt"],
            ising_terms["coeffs"],
        ):
            energies[k] += c * z[s] * z[t]
        for idx, c in zip(
            ising_terms["single_qubit_indices"],
            ising_terms["single_qubit_coeffs"],
        ):
            energies[k] += c * z[idx]

    return energies


def _qaoa_circuit(
    n_qubits: int,
    layer_count: int,
    thetas: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Simulate the QAOA circuit via numpy statevector evolution (fallback)."""
    N = 1 << n_qubits
    psi = np.ones(N, dtype=complex) / np.sqrt(N)

    for layer in range(layer_count):
        gamma = thetas[2 * layer]
        beta = thetas[2 * layer + 1]

        psi *= np.exp(-1j * gamma * energies)

        cb = np.cos(beta)
        sb = np.sin(beta)
        for j in range(n_qubits):
            mask = 1 << j
            for k in range(N):
                if (k & mask) == 0:
                    k2 = k | mask
                    a0, a1 = psi[k], psi[k2]
                    psi[k] = cb * a0 - 1j * sb * a1
                    psi[k2] = -1j * sb * a0 + cb * a1

    return psi


def _numpy_run_qaoa(
    Q: np.ndarray,
    qubit_count: int,
    layer_count: int,
    seed: int = 42,
    shots: int = 10000,
    maxiter: int = 200,
) -> dict:
    """Run cold-start QAOA using numpy statevector simulation (fallback)."""
    rng = np.random.default_rng(seed)
    ising = extract_ising_terms(Q)
    energies = _precompute_energies(ising)
    offset = ising["offset"]

    n_params = 2 * layer_count
    init_params = rng.uniform(0, np.pi, size=n_params)

    convergence = []

    def objective(thetas):
        psi = _qaoa_circuit(qubit_count, layer_count, thetas, energies)
        probs = np.abs(psi) ** 2
        exp_val = float(np.dot(probs, energies)) + offset
        convergence.append(exp_val)
        return exp_val

    result = scipy_minimize(
        objective,
        init_params,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )

    optimal_params = result.x
    optimal_energy = result.fun

    psi = _qaoa_circuit(qubit_count, layer_count, optimal_params, energies)
    probs = np.abs(psi) ** 2
    N = 1 << qubit_count
    samples = rng.choice(N, size=shots, p=probs)

    counts: dict[str, int] = {}
    for s in samples:
        bs = format(s, f"0{qubit_count}b")[::-1]
        counts[bs] = counts.get(bs, 0) + 1

    best_bitstring = max(counts, key=counts.get)

    return {
        "optimal_energy": optimal_energy,
        "optimal_params": optimal_params,
        "best_bitstring": best_bitstring,
        "counts": counts,
        "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# Public API — dispatches to cudaq or numpy fallback
# ---------------------------------------------------------------------------

def run_qaoa(
    Q: np.ndarray,
    qubit_count: int,
    layer_count: int,
    optimizer: str = "COBYLA",
    seed: int = 42,
    shots: int = 10000,
    maxiter: int = 200,
) -> dict:
    """
    Run cold-start QAOA on a QUBO matrix.

    Uses cudaq kernels when available, numpy statevector fallback otherwise.

    Returns
    -------
    dict with keys:
        optimal_energy  – best expectation value found (Ising scale + offset)
        optimal_params  – optimized [gamma_1, beta_1, ..., gamma_p, beta_p]
        best_bitstring  – most-probable bitstring (LSB-first, char i = x_i)
        counts          – {bitstring: count} from shot sampling
        convergence     – list of expectation values at each optimizer evaluation
    """
    if CUDAQ_AVAILABLE:
        return _cudaq_run_qaoa(Q, qubit_count, layer_count, seed, shots, maxiter)
    else:
        return _numpy_run_qaoa(Q, qubit_count, layer_count, seed, shots, maxiter)


# ---------------------------------------------------------------------------
# Portfolio evaluation (pure math — no quantum)
# ---------------------------------------------------------------------------

def evaluate_portfolio(
    bitstring: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float,
    rf: float = 0.045,
) -> dict:
    """
    Evaluate portfolio metrics from a QAOA bitstring.

    Parameters
    ----------
    bitstring : LSB-first string where character i = x_i (1 = selected)
    mu        : expected return vector (sub-problem scale)
    sigma     : covariance matrix (sub-problem scale)
    q         : risk-aversion parameter
    rf        : risk-free rate for Sharpe ratio (default 4.5%)
    """
    x = np.array([int(b) for b in bitstring], dtype=float)
    selected = np.where(x == 1)[0].tolist()
    n_selected = int(x.sum())

    expected_return = float(mu @ x)
    portfolio_var = float(x @ sigma @ x)
    risk = float(np.sqrt(portfolio_var)) if portfolio_var > 0 else 0.0
    sharpe = (expected_return - rf) / risk if risk > 1e-10 else 0.0

    return {
        "selected_indices": selected,
        "n_selected": n_selected,
        "expected_return": expected_return,
        "risk": risk,
        "sharpe_ratio": sharpe,
        "bitstring": bitstring,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from src.data_pipeline import (
        fetch_stock_data,
        compute_log_returns,
        compute_financial_metrics,
        TICKERS,
        START_DATE,
        END_DATE,
    )
    from src.clustering import cluster_stocks, build_subproblems
    from src.qubo import brute_force_qubo

    print(f"Backend: {'cudaq' if CUDAQ_AVAILABLE else 'numpy (fallback)'}")

    # ---- Data & clustering (reproduce Phase 3) ----
    print("Fetching stock data...")
    prices = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    returns = compute_log_returns(prices)
    mu, sigma, rho = compute_financial_metrics(returns)

    N_CLUSTERS = 4
    BUDGET = 6
    Q_RISK = 0.5
    PENALTY = 10.0

    labels = cluster_stocks(rho, N_CLUSTERS)
    subproblems = build_subproblems(
        mu, sigma, labels, BUDGET, Q_RISK, PENALTY, TICKERS
    )

    # Pick smallest cluster
    sp = min(subproblems, key=lambda s: len(s["stock_indices"]))
    Q_sub = sp["qubo"]
    sub_mu = sp["mu"]
    sub_sigma = sp["sigma"]
    n = Q_sub.shape[0]

    print(f"\nValidation cluster {sp['cluster_id']}: {sp['tickers']}")
    print(f"  Qubits: {n}, Budget: {sp['budget']}")

    # ---- Brute force reference ----
    bf_x, bf_val = brute_force_qubo(Q_sub, budget=sp["budget"])
    bf_bs = "".join(str(int(b)) for b in bf_x)
    bf_port = evaluate_portfolio(bf_bs, sub_mu, sub_sigma, Q_RISK)

    print(f"\nBrute-force optimum:")
    print(f"  x = {bf_x.astype(int)},  QUBO energy = {bf_val:.6f}")
    print(
        f"  Return = {bf_port['expected_return']:.4f}, "
        f"Risk = {bf_port['risk']:.4f}, "
        f"Sharpe = {bf_port['sharpe_ratio']:.4f}"
    )

    # ---- QAOA at p = 1, 2, 3 ----
    print("\n" + "=" * 70)
    print(f"QAOA Results  (cold-start, {'cudaq' if CUDAQ_AVAILABLE else 'numpy fallback'})")
    print("=" * 70)

    results = {}
    for p in [1, 2, 3]:
        res = run_qaoa(Q_sub, n, p, seed=42, shots=10000, maxiter=200)
        qaoa_x = np.array([int(b) for b in res["best_bitstring"]], dtype=float)
        sampled_energy = float(qaoa_x @ Q_sub @ qaoa_x)
        port = evaluate_portfolio(res["best_bitstring"], sub_mu, sub_sigma, Q_RISK)

        budget_ok = port["n_selected"] == sp["budget"]
        if budget_ok and abs(sampled_energy) > 1e-10 and sampled_energy < 0:
            ar = bf_val / sampled_energy
        else:
            ar = float("nan")

        results[p] = {
            **res,
            "portfolio": port,
            "sampled_energy": sampled_energy,
            "approx_ratio": ar,
        }

        print(f"\np = {p}:")
        print(f"  Expectation energy : {res['optimal_energy']:.6f}")
        print(f"  Best bitstring     : {res['best_bitstring']}  ->  "
              f"{[sp['tickers'][i] for i in port['selected_indices']]}")
        print(f"  Sampled QUBO energy: {sampled_energy:.6f}")
        print(f"  Approx ratio       : {ar:.4f}")
        print(f"  Return = {port['expected_return']:.4f}, "
              f"Risk = {port['risk']:.4f}, "
              f"Sharpe = {port['sharpe_ratio']:.4f}")
        print(f"  Budget check       : {port['n_selected']}/{sp['budget']} "
              f"{'OK' if port['n_selected'] == sp['budget'] else 'VIOLATED'}")

    # ---- Convergence plot ----
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for p in [1, 2, 3]:
        conv = results[p]["convergence"]
        ax.plot(range(len(conv)), conv, color=colors[p], label=f"p={p}", alpha=0.8)
    ax.axhline(
        y=bf_val, color="black", linestyle="--", linewidth=1.5,
        label=f"Brute force  ({bf_val:.4f})",
    )
    ax.set_xlabel("Function Evaluation")
    ax.set_ylabel("QUBO Energy (expectation value)")
    ax.set_title(
        f"Cold-Start QAOA Convergence — "
        f"Cluster {sp['cluster_id']} ({', '.join(sp['tickers'])})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "qaoa_convergence.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nConvergence plot saved to {save_path}")
