"""
QUBO formulation for Markowitz portfolio optimization.

Follows Buonaiuto et al. (Scientific Reports, 2023) for the
mean-variance -> QUBO -> Ising mapping.
"""

import itertools
import numpy as np


def build_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float,
    budget: int,
    penalty: float,
) -> np.ndarray:
    """
    Build QUBO matrix for Markowitz mean-variance portfolio optimization.

    Objective:  -mu^T x  +  q * x^T Sigma x
    Penalty:    penalty * (sum(x) - budget)^2

    Combined QUBO matrix Q where cost = x^T Q x:
        Q_ii = -mu_i + q*sigma_ii + penalty*(1 - 2*budget)
        Q_ij = q*sigma_ij + penalty   (i != j)
    """
    n = len(mu)
    Q = np.zeros((n, n))

    for i in range(n):
        Q[i, i] = -mu[i] + q * sigma[i, i] + penalty * (1 - 2 * budget)

    for i in range(n):
        for j in range(i + 1, n):
            val = q * sigma[i, j] + penalty
            Q[i, j] = val
            Q[j, i] = val

    return Q


def qubo_to_ising(
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO (binary x in {0,1}) to Ising (spin z in {-1,+1}).

    Substitution: x_i = (1 - z_i) / 2

    x^T Q x = sum_ij Q_ij * x_i * x_j
            = sum_ij Q_ij * (1-z_i)/2 * (1-z_j)/2

    After expansion:
        J_ij = Q_ij / 4               (i != j)
        h_i  = -(1/2) * (Q_ii + sum_{j!=i} Q_ij / 2)
        offset = sum_ii Q_ii / 4 + sum_{i<j} Q_ij / 2  +  ... (constant)

    Detailed derivation:
        x_i x_j = (1-z_i)(1-z_j)/4 = (1 - z_i - z_j + z_i z_j)/4

    So: x^T Q x = (1/4) sum_ij Q_ij (1 - z_i - z_j + z_i z_j)
        = (1/4) sum_ij Q_ij z_i z_j
          - (1/4) sum_ij Q_ij z_i
          - (1/4) sum_ij Q_ij z_j
          + (1/4) sum_ij Q_ij

    Collecting terms:
        J_ij (i<j) = (Q_ij + Q_ji) / 4  = Q_ij / 2  (since Q is symmetric for off-diag)
        But diagonal z_i*z_i = 1, so diagonal Q_ii z_i z_i = Q_ii (constant).

    Let's do this carefully with the standard mapping.
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = 0.0

    # Expand x_i = (1 - z_i)/2
    # x_i * x_j = (1 - z_i)(1 - z_j)/4
    #           = (1 - z_i - z_j + z_i*z_j) / 4
    #
    # For i == j: x_i^2 = x_i (binary), and (1-z_i)^2/4 = (1-2z_i+1)/4 = (1-z_i)/2
    #   So x_i = (1-z_i)/2, which gives:
    #   Q_ii * x_i = Q_ii/2 - Q_ii*z_i/2
    #
    # For i != j:
    #   Q_ij * x_i * x_j = Q_ij * (1 - z_i - z_j + z_i*z_j) / 4

    # Diagonal terms: Q_ii * x_i = Q_ii*(1 - z_i)/2
    for i in range(n):
        offset += Q[i, i] / 2.0
        h[i] += -Q[i, i] / 2.0

    # Off-diagonal terms: Q_ij * (1 - z_i - z_j + z_i*z_j)/4
    for i in range(n):
        for j in range(i + 1, n):
            Qij = Q[i, j] + Q[j, i]  # sum both triangle entries
            offset += Qij / 4.0
            h[i] += -Qij / 4.0
            h[j] += -Qij / 4.0
            J[i, j] += Qij / 4.0

    # Make J symmetric
    J = J + J.T

    return J, h, offset


def qubo_to_cudaq_hamiltonian(Q: np.ndarray):
    """
    Convert QUBO to a cudaq.SpinOperator Hamiltonian via the Ising mapping.

    Returns a cudaq.SpinOperator: H = offset*I + sum_i h_i*Z_i + sum_{i<j} J_ij*Z_i*Z_j
    """
    from cudaq import spin

    J, h, offset = qubo_to_ising(Q)
    n = Q.shape[0]

    hamiltonian = offset * spin.i(0)

    for i in range(n):
        if abs(h[i]) > 1e-10:
            hamiltonian += h[i] * spin.z(i)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-10:
                hamiltonian += J[i, j] * spin.z(i) * spin.z(j)

    return hamiltonian


def brute_force_qubo(
    Q: np.ndarray, budget: int | None = None
) -> tuple[np.ndarray, float]:
    """
    Brute-force solve QUBO by enumerating all 2^n binary vectors.

    Minimizes x^T Q x. If budget is given, only considers vectors
    where sum(x) == budget.
    """
    n = Q.shape[0]
    best_x = None
    best_val = np.inf

    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        if budget is not None and int(x.sum()) != budget:
            continue
        val = x @ Q @ x
        if val < best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val

