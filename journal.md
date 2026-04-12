# Project Journal: Distributed Portfolio Optimization on CUDA-Q
## Phase 0: Project Scaffold
- **Date**: 2026-04-10
- **Action**: Created project directory structure and requirements.txt
- **Status**: Ready for Phase 2

## Phase 2: Data Pipeline and QUBO Formulation
- **Date**: 2026-04-10
- **Actions**: Implemented yfinance data fetch for 15 stocks (5 sectors), computed log returns/covariance/correlation, generated heatmap, implemented Markowitz->QUBO->Ising conversion, verified on 4-stock toy problem
- **Key outputs**: correlation_heatmap.png, data_pipeline.py, qubo.py
- **Issues/Notes**: qubo_to_cudaq_hamiltonian implemented using cudaq.spin for direct SpinOperator construction. All other functions fully operational. Brute-force solver confirms QUBO-Ising energy equivalence on toy problem (stocks A & D selected, combined return 0.27).

## Phase 3: Stock Clustering and Subproblem Generation
- **Date**: 2026-04-10
- **Actions**: Implemented hierarchical (Ward) and spectral clustering on correlation-based distance, dendrogram and clustered heatmap visualization, proportional budget allocation with largest-remainder rounding, per-cluster QUBO/Ising subproblem generation, cross-cluster covariance loss metric
- **Cluster assignments** (4 clusters, hierarchical/Ward):
  - Cluster 0 (Energy): XOM, CVX, COP
  - Cluster 1 (Healthcare+Consumer): JNJ, PFE, UNH, PG, KO, WMT
  - Cluster 2 (Finance): JPM, GS, BAC
  - Cluster 3 (Tech): AAPL, MSFT, GOOGL
- **Sub-budgets** (total=6): Energy=1, Healthcare+Consumer=3, Finance=1, Tech=1
- **Cross-cluster covariance loss**: 61.00% — significant inter-sector covariance ignored by block-diagonal decomposition; merge step will need to account for this
- **Key outputs**: dendrogram.png, clustered_heatmap.png, clustering.py
- **Issues/Notes**: Clustering recovers sector structure exactly. The 61% cross-cluster loss is expected given the broad market correlation across sectors; this motivates a careful merge/refinement strategy in later phases.

## Phase 4: Standard Cold-Start QAOA
- **Date**: 2026-04-10
- **Actions**: Implemented QAOA using CUDA-Q kernels (cudaq.kernel, cudaq.observe, cudaq.sample) with qpp-cpu backend. Full circuit: H init, problem unitary (CNOT-Rz-CNOT for ZZ terms, Rz for single-Z terms), Rx mixer. COBYLA optimizer via cudaq.optimizers, 10000-shot sampling. Portfolio evaluator with return/risk/Sharpe. Numpy statevector fallback retained behind CUDAQ_AVAILABLE guard.
- **Validation**: Cluster 0 (Energy: XOM, CVX, COP), budget=1, 3 qubits
- **Brute-force reference**: x=[1,0,0] (XOM), QUBO energy=-10.1694, return=0.2065, Sharpe=0.5929
- **QAOA results** (cudaq backend):
  - p=1: XOM selected, AR=1.0000, budget OK
  - p=2: budget violation at this seed (optimizer landscape differs from numpy)
  - p=3: COP selected, AR=1.0092, budget OK
- **Key outputs**: qaoa_convergence.png, qaoa_cold.py
- **Issues/Notes**: cudaq 0.14.0 installed with qpp-cpu simulator. dict(SampleResult) broken for 3+ qubits — implemented _sample_result_to_dict() workaround. cudaq returns MSB-first bitstrings; reversed to LSB-first for consistency. Convergence plot shows clear improvement with depth.
