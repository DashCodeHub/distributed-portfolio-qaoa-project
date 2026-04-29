"""
Stock clustering and subproblem generation for distributed portfolio optimization.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import SpectralClustering

from src.qubo import build_qubo, qubo_to_ising, qubo_to_cudaq_hamiltonian
from src.qaoa_cold import extract_ising_terms


def correlation_to_distance(rho: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d_ij = sqrt(2*(1 - rho_ij))."""
    return np.sqrt(np.maximum(0.0, 2.0 * (1.0 - rho)))


def cluster_stocks(
    rho: np.ndarray, n_clusters: int, method: str = "hierarchical"
) -> np.ndarray:
    """
    Cluster stocks based on correlation matrix.

    Parameters
    ----------
    rho : correlation matrix (n x n)
    n_clusters : number of clusters
    method : 'hierarchical' (ward linkage) or 'spectral'

    Returns
    -------
    labels : cluster label array of length n (0-indexed)
    """
    if method == "hierarchical":
        dist = correlation_to_distance(rho)
        # Convert full distance matrix to condensed form for linkage
        n = dist.shape[0]
        condensed = dist[np.triu_indices(n, k=1)]
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # 0-indexed
    elif method == "spectral":
        # Spectral clustering uses affinity; shift correlation to [0, 1]
        affinity = (rho + 1.0) / 2.0
        np.fill_diagonal(affinity, 1.0)
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
            assign_labels="kmeans",
        )
        labels = sc.fit_predict(affinity)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hierarchical' or 'spectral'.")
    return labels


def plot_dendrogram(
    rho: np.ndarray, tickers: list[str], save_path: str, n_clusters: int = 4
) -> None:
    """Plot hierarchical clustering dendrogram colored by cluster."""
    dist = correlation_to_distance(rho)
    n = dist.shape[0]
    condensed = dist[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="ward")

    # Find the distance threshold that yields n_clusters
    # fcluster with maxclust uses a threshold internally; we compute it
    # for coloring by finding the merge distance between cluster n_clusters and n_clusters-1
    if n_clusters > 1 and len(Z) >= n_clusters:
        color_threshold = (Z[-(n_clusters - 1), 2] + Z[-n_clusters, 2]) / 2.0
    else:
        color_threshold = 0

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z,
        labels=tickers,
        color_threshold=color_threshold,
        above_threshold_color="grey",
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax,
    )
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_ylabel("Distance d = sqrt(2(1 - rho))")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Dendrogram saved to {save_path}")


def plot_clustered_heatmap(
    rho: np.ndarray,
    labels: np.ndarray,
    tickers: list[str],
    save_path: str,
) -> None:
    """Plot correlation heatmap reordered by cluster with cluster boundaries."""
    # Sort indices by cluster label
    order = np.argsort(labels, kind="stable")
    rho_ordered = rho[np.ix_(order, order)]
    tickers_ordered = [tickers[i] for i in order]
    labels_ordered = labels[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        rho_ordered,
        annot=True,
        fmt=".2f",
        xticklabels=tickers_ordered,
        yticklabels=tickers_ordered,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    # Draw cluster boundary lines
    unique_labels = np.unique(labels_ordered)
    pos = 0
    for lbl in unique_labels[:-1]:
        count = np.sum(labels_ordered == lbl)
        pos += count
        ax.axhline(y=pos, color="black", linewidth=2)
        ax.axvline(x=pos, color="black", linewidth=2)

    ax.set_title("Clustered Correlation Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Clustered heatmap saved to {save_path}")


def _allocate_sub_budgets(labels: np.ndarray, budget: int) -> dict[int, int]:
    """Allocate budget proportionally to clusters, ensuring sum == budget."""
    unique_labels = np.unique(labels)
    n = len(labels)
    cluster_sizes = {k: int(np.sum(labels == k)) for k in unique_labels}

    # Proportional allocation with largest-remainder rounding
    raw = {k: budget * cluster_sizes[k] / n for k in unique_labels}
    floored = {k: int(np.floor(v)) for k, v in raw.items()}
    remainders = {k: raw[k] - floored[k] for k in unique_labels}

    allocated = sum(floored.values())
    deficit = budget - allocated

    # Distribute remaining budget to clusters with largest remainders
    for k in sorted(remainders, key=remainders.get, reverse=True):
        if deficit <= 0:
            break
        floored[k] += 1
        deficit -= 1

    # Ensure every cluster gets at least 1 if possible
    for k in unique_labels:
        if floored[k] == 0 and budget >= len(unique_labels):
            # Steal from the cluster with the largest allocation
            donor = max(floored, key=floored.get)
            if floored[donor] > 1:
                floored[donor] -= 1
                floored[k] = 1

    return floored


def build_subproblems(
    mu: np.ndarray,
    sigma: np.ndarray,
    labels: np.ndarray,
    budget: int,
    q: float,
    penalty: float,
    tickers: list[str] | None = None,
) -> list[dict]:
    """
    Build per-cluster subproblems.

    For each cluster: extract sub-mu, sub-sigma, assign proportional sub-budget,
    build QUBO, convert to Ising Hamiltonian (J, h, offset).

    Returns list of dicts with keys:
        cluster_id, stock_indices, tickers, mu, sigma, budget, qubo, hamiltonian
    """
    unique_labels = np.unique(labels)
    sub_budgets = _allocate_sub_budgets(labels, budget)
    subproblems = []

    for k in unique_labels:
        indices = np.where(labels == k)[0]
        sub_mu = mu[indices]
        sub_sigma = sigma[np.ix_(indices, indices)]
        sub_budget = sub_budgets[k]

        sub_qubo = build_qubo(sub_mu, sub_sigma, q, sub_budget, penalty)
        J, h, offset = qubo_to_ising(sub_qubo)

        # Extract Ising terms for QAOA kernels
        ising = extract_ising_terms(sub_qubo)
        n_qubits = sub_qubo.shape[0]

        sub_tickers = (
            [tickers[i] for i in indices] if tickers is not None else None
        )

        sp = {
            "cluster_id": int(k),
            "stock_indices": indices.tolist(),
            "tickers": sub_tickers,
            "mu": sub_mu,
            "sigma": sub_sigma,
            "budget": sub_budget,
            "qubo": sub_qubo,
            "hamiltonian": {"J": J, "h": h, "offset": offset},
            "qubit_count": n_qubits,
            "edges_src": ising["edges_src"].tolist(),
            "edges_tgt": ising["edges_tgt"].tolist(),
            "coeffs": ising["coeffs"].tolist(),
            "single_qubit_indices": ising["single_qubit_indices"].tolist(),
            "single_qubit_coeffs": ising["single_qubit_coeffs"].tolist(),
        }

        sp["cudaq_hamiltonian"] = qubo_to_cudaq_hamiltonian(sub_qubo)

        subproblems.append(sp)

    return subproblems


def compute_cross_cluster_loss(sigma: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the fraction of total off-diagonal covariance that falls in
    cross-cluster blocks (i.e., what the block-diagonal decomposition ignores).

    Returns a value in [0, 1].
    """
    n = sigma.shape[0]
    total = 0.0
    cross = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = abs(sigma[i, j])
            total += val
            if labels[i] != labels[j]:
                cross += val
    return cross / total if total > 0 else 0.0


if __name__ == "__main__":
    import os
    from src.data_pipeline import (
        fetch_stock_data, compute_log_returns, compute_financial_metrics, TICKERS,
        START_DATE, END_DATE,
    )

    N_CLUSTERS = 4
    BUDGET = 6
    Q_RISK = 0.5
    PENALTY = 10.0

    # --- Fetch data ---
    print("Fetching stock data...")
    prices = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    returns = compute_log_returns(prices)
    mu, sigma, rho = compute_financial_metrics(returns)

    # --- Cluster ---
    print(f"\nClustering {len(TICKERS)} stocks into {N_CLUSTERS} groups...")
    labels = cluster_stocks(rho, N_CLUSTERS, method="hierarchical")

    print("\nCluster assignments:")
    for k in range(N_CLUSTERS):
        members = [TICKERS[i] for i in range(len(TICKERS)) if labels[i] == k]
        print(f"  Cluster {k}: {members}")

    # --- Plots ---
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_dendrogram(rho, TICKERS, os.path.join(fig_dir, "dendrogram.png"), N_CLUSTERS)
    plot_clustered_heatmap(rho, labels, TICKERS, os.path.join(fig_dir, "clustered_heatmap.png"))

    # --- Subproblems ---
    print(f"\nBuilding subproblems (budget={BUDGET}, q={Q_RISK}, penalty={PENALTY})...")
    subproblems = build_subproblems(mu, sigma, labels, BUDGET, Q_RISK, PENALTY, TICKERS)

    print("\nSubproblem summary:")
    for sp in subproblems:
        print(f"  Cluster {sp['cluster_id']}: "
              f"stocks={sp['tickers']}, "
              f"budget={sp['budget']}, "
              f"QUBO size={sp['qubo'].shape[0]}x{sp['qubo'].shape[0]}")

    # --- Cross-cluster loss ---
    loss = compute_cross_cluster_loss(sigma, labels)
    print(f"\nCross-cluster covariance loss: {loss:.4f} ({loss*100:.2f}%)")
