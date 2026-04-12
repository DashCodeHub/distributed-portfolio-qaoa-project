"""
Test QUBO formulation on a 4-stock toy problem.

Build QUBO (q=0.5, budget=2, penalty=10), brute force it,
convert to Ising, and verify the ground state energy matches.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qubo import build_qubo, qubo_to_ising, brute_force_qubo


def test_4stock_toy_problem():
    """4-stock toy problem with synthetic data."""
    np.random.seed(42)

    # Synthetic financial metrics for 4 stocks
    mu = np.array([0.12, 0.10, 0.08, 0.15])  # annualized returns
    sigma = np.array([
        [0.04, 0.006, 0.002, 0.005],
        [0.006, 0.03, 0.004, 0.003],
        [0.002, 0.004, 0.025, 0.001],
        [0.005, 0.003, 0.001, 0.05],
    ])  # annualized covariance

    q = 0.5
    budget = 2
    penalty = 10.0

    # Build QUBO
    Q = build_qubo(mu, sigma, q, budget, penalty)
    n = len(mu)

    print("=" * 60)
    print("4-Stock Toy Problem")
    print("=" * 60)
    print(f"Expected returns (mu): {mu}")
    print(f"Risk aversion (q): {q}")
    print(f"Budget: {budget}")
    print(f"Penalty: {penalty}")
    print(f"\nQUBO matrix Q ({n}x{n}):")
    print(Q)

    # Verify QUBO matrix structure
    assert Q.shape == (n, n), f"Expected ({n},{n}), got {Q.shape}"

    # Check diagonal: Q_ii = -mu_i + q*sigma_ii + penalty*(1 - 2*budget)
    for i in range(n):
        expected_diag = -mu[i] + q * sigma[i, i] + penalty * (1 - 2 * budget)
        assert np.isclose(Q[i, i], expected_diag), (
            f"Q[{i},{i}] = {Q[i, i]}, expected {expected_diag}"
        )

    # Check off-diagonal: Q_ij = q*sigma_ij + penalty
    for i in range(n):
        for j in range(i + 1, n):
            expected_offdiag = q * sigma[i, j] + penalty
            assert np.isclose(Q[i, j], expected_offdiag), (
                f"Q[{i},{j}] = {Q[i, j]}, expected {expected_offdiag}"
            )

    # Check symmetry
    assert np.allclose(Q, Q.T), "QUBO matrix is not symmetric"
    print("\n[PASS] QUBO matrix structure verified")

    # Brute force solve
    best_x, best_val = brute_force_qubo(Q, budget=budget)
    print(f"\nBrute-force optimal portfolio: {best_x.astype(int)}")
    print(f"Brute-force optimal QUBO value: {best_val:.6f}")

    # Verify the solution satisfies budget constraint
    assert int(best_x.sum()) == budget, (
        f"Budget violated: sum(x) = {int(best_x.sum())}, expected {budget}"
    )
    print("[PASS] Budget constraint satisfied")

    # Convert to Ising
    J, h, offset = qubo_to_ising(Q)
    print(f"\nIsing J matrix:\n{J}")
    print(f"Ising h vector: {h}")
    print(f"Ising offset: {offset:.6f}")

    # Verify ground state energy: convert best_x to spins, compute Ising energy
    best_z = 1 - 2 * best_x  # x=0 -> z=+1, x=1 -> z=-1

    # Ising energy = sum_{i<j} J_ij z_i z_j + sum_i h_i z_i + offset
    ising_energy = offset
    for i in range(n):
        ising_energy += h[i] * best_z[i]
        for j in range(i + 1, n):
            ising_energy += J[i, j] * best_z[i] * best_z[j]

    print(f"\nIsing energy of optimal solution: {ising_energy:.6f}")
    print(f"QUBO energy of optimal solution:  {best_val:.6f}")

    assert np.isclose(ising_energy, best_val, atol=1e-10), (
        f"Energy mismatch: Ising={ising_energy:.10f}, QUBO={best_val:.10f}"
    )
    print("[PASS] Ising ground state energy matches QUBO optimum")

    # Also verify by brute-forcing all spin configurations
    best_ising_energy = np.inf
    best_spin = None
    for bits in np.ndindex(*([2] * n)):
        z = np.array([1 - 2 * b for b in bits], dtype=float)
        x = (1 - z) / 2
        if budget is not None and int(x.sum()) != budget:
            continue
        e = offset
        for i in range(n):
            e += h[i] * z[i]
            for j in range(i + 1, n):
                e += J[i, j] * z[i] * z[j]
        if e < best_ising_energy:
            best_ising_energy = e
            best_spin = z.copy()

    best_x_from_ising = ((1 - best_spin) / 2).astype(int)
    print(f"\nIsing brute-force optimal spins: {best_spin.astype(int)}")
    print(f"Corresponding portfolio:         {best_x_from_ising}")
    print(f"Ising brute-force energy:        {best_ising_energy:.6f}")

    assert np.isclose(best_ising_energy, best_val, atol=1e-10), (
        f"Ising brute-force mismatch: {best_ising_energy} vs {best_val}"
    )
    assert np.array_equal(best_x_from_ising, best_x.astype(int)), (
        "Ising and QUBO brute-force found different optimal portfolios"
    )
    print("[PASS] Ising brute-force matches QUBO brute-force")

    # Print portfolio interpretation
    tickers = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]
    selected = [t for t, xi in zip(tickers, best_x) if xi == 1]
    print(f"\n{'=' * 60}")
    print(f"Optimal portfolio selects: {selected}")
    print(f"Return of selected: {mu[best_x.astype(bool)].sum():.4f}")
    print(f"{'=' * 60}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_4stock_toy_problem()
