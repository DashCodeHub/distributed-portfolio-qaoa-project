"""
Goemans-Williamson SDP relaxation and randomized rounding for QUBO.

Solves the SDP relaxation of min x^T Q x (binary x) by lifting to
Y = xx^T (PSD, diag=1), then recovers binary solutions via random
hyperplane rounding.

Reference: Goemans & Williamson (1995); adapted for portfolio QUBO.
"""

import numpy as np
import cvxpy as cp


def solve_sdp_relaxation(Q: np.ndarray) -> np.ndarray:
    """
    Solve the SDP relaxation of min x^T Q x  s.t. x in {-1,+1}^n.

    Relaxation:
        minimize   trace(Q @ Y)
        subject to Y[i,i] == 1  for all i
                   Y >> 0       (positive semidefinite)

    Parameters
    ----------
    Q : (n, n) symmetric matrix (Ising-scale or raw — caller decides).

    Returns
    -------
    Y_optimal : (n, n) PSD matrix with unit diagonal.
    """
    n = Q.shape[0]
    Y = cp.Variable((n, n), symmetric=True)

    constraints = [Y >> 0]
    for i in range(n):
        constraints.append(Y[i, i] == 1)

    prob = cp.Problem(cp.Minimize(cp.trace(Q @ Y)), constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP solver returned status: {prob.status}")

    return np.array(Y.value)


def gw_rounding(Y: np.ndarray, num_trials: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Goemans-Williamson random hyperplane rounding.

    Decomposes Y = V V^T via eigenvalue decomposition, then samples
    random hyperplanes r and rounds x = sign(V @ r).

    Parameters
    ----------
    Y         : (n, n) PSD matrix from SDP relaxation (unit diagonal).
    num_trials: number of random hyperplane trials.
    seed      : random seed for reproducibility.

    Returns
    -------
    best_x : (n,) array in {-1, +1}^n giving the best rounding found.
    """
    n = Y.shape[0]
    rng = np.random.default_rng(seed)

    # Eigenvalue decomposition; clamp negative eigenvalues to 0
    eigvals, eigvecs = np.linalg.eigh(Y)
    eigvals = np.maximum(eigvals, 0.0)
    # V such that Y ≈ V V^T
    V = eigvecs * np.sqrt(eigvals)[np.newaxis, :]  # (n, n)

    best_x = None
    best_obj = np.inf

    for _ in range(num_trials):
        r = rng.standard_normal(n)
        r /= np.linalg.norm(r)
        x = np.sign(V @ r)
        # Replace any zeros with +1
        x[x == 0] = 1.0
        obj = float(x @ Y @ x)  # proxy; caller evaluates true objective
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()

    return best_x


def gw_solve(
    Q: np.ndarray,
    budget: int | None = None,
    num_trials: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Solve QUBO approximately via GW SDP relaxation + rounding.

    The QUBO is in binary {0,1} form: min x^T Q x.
    We convert to spin {-1,+1} for SDP, round, then convert back.

    If budget is given, after rounding we enforce the budget constraint
    by selecting the `budget` variables with highest preference for x_i=1
    (based on the SDP embedding distances).

    Parameters
    ----------
    Q          : (n, n) QUBO matrix.
    budget     : if not None, enforce sum(x) == budget.
    num_trials : number of GW rounding trials.
    seed       : random seed.

    Returns
    -------
    best_x   : (n,) binary {0,1} solution.
    best_obj : QUBO objective value x^T Q x.
    """
    from src.qubo import qubo_to_ising

    J, h, offset = qubo_to_ising(Q)
    n = Q.shape[0]

    # Build Ising cost matrix for SDP: E(z) = z^T W z + h^T z + offset
    # where W_ij = J_ij for i!=j, W_ii = 0
    # SDP relaxation of z^T W z: trace(W @ Y)
    # For the linear term h^T z, we introduce an auxiliary spin z_0 = +1
    # and encode h_i as interaction between z_0 and z_i.
    # Extended matrix W_ext of size (n+1) x (n+1):
    #   W_ext[0,0] = 0
    #   W_ext[0,i] = W_ext[i,0] = h[i] / 2  (since h_i * z_i = h_i * z_0 * z_i with z_0=1)
    #   W_ext[i,j] = J[i,j] for i,j >= 1

    n_ext = n + 1
    W_ext = np.zeros((n_ext, n_ext))
    W_ext[1:, 1:] = J
    for i in range(n):
        W_ext[0, i + 1] = h[i] / 2.0
        W_ext[i + 1, 0] = h[i] / 2.0

    # Solve SDP
    Y_ext = solve_sdp_relaxation(W_ext)

    # GW rounding in extended space
    rng = np.random.default_rng(seed)
    eigvals, eigvecs = np.linalg.eigh(Y_ext)
    eigvals = np.maximum(eigvals, 0.0)
    V = eigvecs * np.sqrt(eigvals)[np.newaxis, :]

    best_x_binary = None
    best_obj = np.inf

    for _ in range(num_trials):
        r = rng.standard_normal(n_ext)
        r /= np.linalg.norm(r)
        z_ext = np.sign(V @ r)
        z_ext[z_ext == 0] = 1.0

        # Align so that z_0 = +1 (the auxiliary spin)
        if z_ext[0] < 0:
            z_ext = -z_ext

        z = z_ext[1:]  # spin variables {-1, +1}

        # Convert spin -> binary: x_i = (1 - z_i) / 2
        x_binary = ((1 - z) / 2).astype(int).astype(float)

        # Enforce budget constraint if needed
        if budget is not None:
            selected = int(x_binary.sum())
            if selected != budget:
                # Use SDP embedding to rank qubits by preference for x_i=1 (z_i=-1)
                # The correlation with auxiliary spin z_0: Y_ext[0, i+1]
                # More negative => more likely z_i = -1 => x_i = 1
                scores = -Y_ext[0, 1:]  # higher score = prefer x_i=1
                ranked = np.argsort(scores)[::-1]  # descending
                x_binary = np.zeros(n)
                for idx in ranked[:budget]:
                    x_binary[idx] = 1.0

        obj = float(x_binary @ Q @ x_binary)
        if obj < best_obj:
            best_obj = obj
            best_x_binary = x_binary.copy()

    return best_x_binary, best_obj
