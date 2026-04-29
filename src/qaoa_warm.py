"""
Warm-start QAOA for portfolio optimization (CUDA-Q backend).

Uses a Goemans-Williamson (GW) classical solution to initialize the QAOA
circuit with biased Ry rotations instead of uniform superposition, and
applies the Egger et al. modified mixer.

Reference: Egger et al., "Warm-starting quantum optimization" (Quantum, 2021).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cudaq
from cudaq import spin

from src.qubo import qubo_to_ising, brute_force_qubo
from src.qaoa_cold import (
    extract_ising_terms,
    evaluate_portfolio,
    _sample_result_to_dict,
)
from src.gw import gw_solve


# ---------------------------------------------------------------------------
# Warm-start angle computation (Egger et al.)
# ---------------------------------------------------------------------------

def compute_ws_angles(x_gw: np.ndarray, epsilon: float = 0.1) -> list[float]:
    """
    Compute warm-start rotation angles from a GW binary solution.

    For each qubit i:
        if x_gw[i] == 1:  theta_i = 2 * arcsin(sqrt(1 - epsilon))
        if x_gw[i] == 0:  theta_i = 2 * arcsin(sqrt(epsilon))

    This biases the initial state toward the GW solution while maintaining
    some exploration (controlled by epsilon).
    """
    thetas = []
    for xi in x_gw:
        if xi == 1:
            theta = 2.0 * np.arcsin(np.sqrt(1.0 - epsilon))
        else:
            theta = 2.0 * np.arcsin(np.sqrt(epsilon))
        thetas.append(theta)
    return thetas


# ---------------------------------------------------------------------------
# CUDA-Q warm-start QAOA kernel
# ---------------------------------------------------------------------------

@cudaq.kernel
def kernel_ws_qaoa(qubit_count: int, layer_count: int,
                   thetas: list[float],
                   ws_angles: list[float],
                   edges_src: list[int], edges_tgt: list[int],
                   coeffs: list[float],
                   sq_indices: list[int], sq_coeffs: list[float]):
    qubits = cudaq.qvector(qubit_count)
    # Warm-start initial state: Ry rotations instead of Hadamard
    for i in range(qubit_count):
        ry(ws_angles[i], qubits[i])
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
        # Egger modified mixer: Ry(θ_i) Rz(-2β) Ry(-θ_i)
        for i in range(qubit_count):
            ry(ws_angles[i], qubits[i])
            rz(-2.0 * beta, qubits[i])
            ry(-ws_angles[i], qubits[i])


def _cudaq_run_ws_qaoa(
    Q: np.ndarray,
    qubit_count: int,
    layer_count: int,
    ws_angles: list[float],
    seed: int = 42,
    shots: int = 10000,
    maxiter: int = 200,
) -> dict:
    """Run warm-start QAOA using cudaq.observe and cudaq.sample."""
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
    optimizer.initial_parameters = np.random.uniform(0, 0.5, size=n_params).tolist()
    optimizer.max_iterations = maxiter

    convergence = []

    def objective(thetas):
        exp_val = cudaq.observe(
            kernel_ws_qaoa, hamiltonian,
            qubit_count, layer_count, thetas,
            ws_angles,
            edges_src, edges_tgt, coeffs,
            sq_indices, sq_coeffs
        ).expectation()
        convergence.append(exp_val)
        return exp_val

    optimal_val, optimal_params = optimizer.optimize(
        dimensions=n_params, function=objective)

    counts = cudaq.sample(
        kernel_ws_qaoa,
        qubit_count, layer_count,
        optimal_params,
        ws_angles,
        edges_src, edges_tgt, coeffs,
        sq_indices, sq_coeffs,
        shots_count=shots)

    counts_dict = _sample_result_to_dict(counts, qubit_count)
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
# Public API
# ---------------------------------------------------------------------------

def run_ws_qaoa(
    Q: np.ndarray,
    qubit_count: int,
    layer_count: int,
    ws_angles: list[float],
    use_ws_mixer: bool = True,
    optimizer: str = "COBYLA",
    seed: int = 42,
    shots: int = 10000,
    maxiter: int = 200,
) -> dict:
    """
    Run warm-start QAOA on a QUBO matrix using CUDA-Q.

    Returns
    -------
    dict with keys: optimal_energy, optimal_params, best_bitstring,
                    counts, convergence.
    """
    return _cudaq_run_ws_qaoa(
        Q, qubit_count, layer_count, ws_angles, seed, shots, maxiter
    )


# ---------------------------------------------------------------------------
# Validation: compare cold-start vs warm-start vs GW-only
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
    from src.qaoa_cold import run_qaoa

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

    sp = min(subproblems, key=lambda s: len(s["stock_indices"]))
    Q_sub = sp["qubo"]
    sub_mu = sp["mu"]
    sub_sigma = sp["sigma"]
    n = Q_sub.shape[0]

    print(f"\nValidation cluster {sp['cluster_id']}: {sp['tickers']}")
    print(f"  Qubits: {n}, Budget: {sp['budget']}")

    bf_x, bf_val = brute_force_qubo(Q_sub, budget=sp["budget"])
    bf_bs = "".join(str(int(b)) for b in bf_x)
    print(f"\nBrute-force optimum:")
    print(f"  x = {bf_x.astype(int)},  QUBO energy = {bf_val:.6f}")

    print("\n" + "=" * 70)
    print("Goemans-Williamson SDP Relaxation + Rounding")
    print("=" * 70)

    gw_x, gw_obj = gw_solve(Q_sub, budget=sp["budget"], num_trials=1000, seed=42)
    gw_bs = "".join(str(int(b)) for b in gw_x)
    gw_port = evaluate_portfolio(gw_bs, sub_mu, sub_sigma, Q_RISK)

    gw_ar = bf_val / gw_obj if abs(gw_obj) > 1e-10 and gw_obj < 0 else float("nan")
    print(f"  GW solution: x = {gw_x.astype(int)}")
    print(f"  GW QUBO energy: {gw_obj:.6f}")
    print(f"  GW approx ratio: {gw_ar:.4f}")
    print(f"  GW portfolio: {[sp['tickers'][i] for i in gw_port['selected_indices']]}")

    ws_angles = compute_ws_angles(gw_x, epsilon=0.1)
    print(f"\nWarm-start angles (epsilon=0.1):")
    for i, (t, xi) in enumerate(zip(ws_angles, gw_x)):
        print(f"  qubit {i} ({sp['tickers'][i]}): x_gw={int(xi)}, theta={t:.4f}")

    print("\n" + "=" * 70)
    print("Cold-Start QAOA")
    print("=" * 70)

    cold_results = {}
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
        cold_results[p] = {"ar": ar, "energy": sampled_energy, "bitstring": res["best_bitstring"], "budget_ok": budget_ok}
        print(f"  p={p}: bitstring={res['best_bitstring']}, energy={sampled_energy:.4f}, AR={ar:.4f}, budget={'OK' if budget_ok else 'VIOLATED'}")

    print("\n" + "=" * 70)
    print("Warm-Start QAOA (Egger et al. modified mixer)")
    print("=" * 70)

    warm_results = {}
    for p in [1, 2, 3]:
        res = run_ws_qaoa(
            Q_sub, n, p, ws_angles,
            use_ws_mixer=True, seed=42, shots=10000, maxiter=200,
        )
        qaoa_x = np.array([int(b) for b in res["best_bitstring"]], dtype=float)
        sampled_energy = float(qaoa_x @ Q_sub @ qaoa_x)
        port = evaluate_portfolio(res["best_bitstring"], sub_mu, sub_sigma, Q_RISK)
        budget_ok = port["n_selected"] == sp["budget"]
        if budget_ok and abs(sampled_energy) > 1e-10 and sampled_energy < 0:
            ar = bf_val / sampled_energy
        else:
            ar = float("nan")
        warm_results[p] = {"ar": ar, "energy": sampled_energy, "bitstring": res["best_bitstring"], "budget_ok": budget_ok}
        print(f"  p={p}: bitstring={res['best_bitstring']}, energy={sampled_energy:.4f}, AR={ar:.4f}, budget={'OK' if budget_ok else 'VIOLATED'}")

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    ar_values = []
    colors = []

    methods.append("GW-only")
    ar_values.append(gw_ar if not np.isnan(gw_ar) else 0.0)
    colors.append("#9467bd")

    for p in [1, 2, 3]:
        methods.append(f"Cold p={p}")
        ar_values.append(cold_results[p]["ar"] if not np.isnan(cold_results[p]["ar"]) else 0.0)
        colors.append({"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}[str(p)])

    for p in [1, 2, 3]:
        methods.append(f"Warm p={p}")
        ar_values.append(warm_results[p]["ar"] if not np.isnan(warm_results[p]["ar"]) else 0.0)
        colors.append({"1": "#aec7e8", "2": "#ffbb78", "3": "#98df8a"}[str(p)])

    bars = ax.bar(methods, ar_values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, ar_values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2, 0.02,
                "budget\nviolated", ha="center", va="bottom", fontsize=7,
                color="red", fontweight="bold",
            )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Exact optimum")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title(
        f"Cold vs Warm-Start QAOA vs GW — "
        f"Cluster {sp['cluster_id']} ({', '.join(sp['tickers'])})"
    )
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "ws_vs_cold.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nComparison chart saved to {save_path}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<16} {'Bitstring':<10} {'QUBO Energy':<14} {'Approx Ratio':<14} {'Budget'}")
    print("-" * 70)
    print(f"{'Brute force':<16} {bf_bs:<10} {bf_val:<14.4f} {'1.0000':<14} {'OK'}")
    print(f"{'GW-only':<16} {gw_bs:<10} {gw_obj:<14.4f} {gw_ar:<14.4f} {'OK'}")
    for p in [1, 2, 3]:
        cr = cold_results[p]
        status = "OK" if cr["budget_ok"] else "VIOLATED"
        ar_str = f"{cr['ar']:.4f}" if not np.isnan(cr["ar"]) else "N/A"
        print(f"{'Cold p=' + str(p):<16} {cr['bitstring']:<10} {cr['energy']:<14.4f} {ar_str:<14} {status}")
    for p in [1, 2, 3]:
        wr = warm_results[p]
        status = "OK" if wr["budget_ok"] else "VIOLATED"
        ar_str = f"{wr['ar']:.4f}" if not np.isnan(wr["ar"]) else "N/A"
        print(f"{'Warm p=' + str(p):<16} {wr['bitstring']:<10} {wr['energy']:<14.4f} {ar_str:<14} {status}")
