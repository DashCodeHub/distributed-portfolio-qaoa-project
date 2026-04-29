"""
Distributed QAOA execution across multiple QPUs (CUDA-Q backend).

Primary path uses cudaq MQPU for true multi-GPU parallelism via
cudaq.sample_async with qpu_id-based routing. Falls back to sequential
qpp-cpu when GPUs are unavailable.
"""

import os
import time
import numpy as np

import cudaq

from src.qaoa_cold import run_qaoa, kernel_qaoa, _sample_result_to_dict
from src.qaoa_warm import run_ws_qaoa, compute_ws_angles, kernel_ws_qaoa
from src.gw import gw_solve


# ---------------------------------------------------------------------------
# QPU setup
# ---------------------------------------------------------------------------

_BACKEND = None
_NUM_QPUS = 1


def setup_mqpu(num_qpus: int | None = None) -> dict:
    """
    Configure the multi-QPU backend.

    Attempts cudaq targets in order:
      1. "nvidia" with option="mqpu" (multi-GPU parallel sampling)
      2. "qpp-cpu" (CPU simulator, sequential)
    """
    global _BACKEND, _NUM_QPUS

    if num_qpus:
        os.environ["CUDAQ_MQPU_NGPUS"] = str(num_qpus)

    try:
        cudaq.set_target("nvidia", option="mqpu")
        _BACKEND = "mqpu"
        _NUM_QPUS = num_qpus or max(1, cudaq.num_available_gpus())
        return {
            "backend": "mqpu",
            "num_qpus": _NUM_QPUS,
            "note": "NVIDIA MQPU backend active — parallel sampling across GPUs",
        }
    except Exception:
        pass

    cudaq.set_target("qpp-cpu")
    _BACKEND = "qpp-cpu"
    _NUM_QPUS = num_qpus or 1
    return {
        "backend": "qpp-cpu",
        "num_qpus": _NUM_QPUS,
        "note": "CPU simulator (qpp-cpu). Sequential dispatch, "
                "mathematically identical to MQPU.",
    }


# ---------------------------------------------------------------------------
# Distributed cold-start QAOA
# ---------------------------------------------------------------------------

def run_distributed_cold_qaoa(
    subproblems: list[dict],
    layer_count: int = 2,
    seed: int = 42,
    shots: int = 10000,
    maxiter: int = 200,
) -> list[dict]:
    """
    Run cold-start QAOA on all subproblems with distributed dispatch.

    With cudaq MQPU: optimize parameters sequentially, then dispatch
    final sampling in parallel across QPUs via sample_async.
    With qpp-cpu: sequential execution (identical math).
    """
    global _BACKEND, _NUM_QPUS
    if _BACKEND is None:
        setup_mqpu()

    results = []
    optimal_params_list = []

    for c, sp in enumerate(subproblems):
        Q = sp["qubo"]
        n = Q.shape[0]
        qpu_id = c % _NUM_QPUS

        t0 = time.perf_counter()
        res = run_qaoa(
            Q, n, layer_count,
            seed=seed + c, shots=shots, maxiter=maxiter,
        )
        elapsed = time.perf_counter() - t0

        qaoa_x = np.array([int(b) for b in res["best_bitstring"]], dtype=float)
        sampled_energy = float(qaoa_x @ Q @ qaoa_x)

        optimal_params_list.append(res["optimal_params"])
        results.append({
            "cluster_id": sp["cluster_id"],
            "best_bitstring": res["best_bitstring"],
            "sampled_energy": sampled_energy,
            "optimal_energy": res["optimal_energy"],
            "optimal_params": res["optimal_params"],
            "counts": res["counts"],
            "timing_s": elapsed,
            "qpu_id": qpu_id,
        })

    if _BACKEND == "mqpu":
        futures = []
        for c, sp in enumerate(subproblems):
            qpu_id = c % _NUM_QPUS
            future = cudaq.sample_async(
                kernel_qaoa,
                sp["qubit_count"], layer_count,
                list(optimal_params_list[c]),
                sp["edges_src"], sp["edges_tgt"], sp["coeffs"],
                sp["single_qubit_indices"], sp["single_qubit_coeffs"],
                shots_count=shots, qpu_id=qpu_id)
            futures.append(future)

        for i, future in enumerate(futures):
            counts = future.get()
            sp = subproblems[i]
            counts_dict = _sample_result_to_dict(counts, sp["qubit_count"])
            counts_lsb = {bs[::-1]: c for bs, c in counts_dict.items()}
            best_bs = max(counts_lsb, key=counts_lsb.get)
            results[i]["best_bitstring"] = best_bs
            results[i]["counts"] = counts_lsb
            qaoa_x = np.array([int(b) for b in best_bs], dtype=float)
            results[i]["sampled_energy"] = float(qaoa_x @ sp["qubo"] @ qaoa_x)

    return results


# ---------------------------------------------------------------------------
# Distributed warm-start QAOA
# ---------------------------------------------------------------------------

def run_distributed_ws_qaoa(
    subproblems: list[dict],
    layer_count: int = 2,
    seed: int = 42,
    shots: int = 10000,
    maxiter: int = 200,
    epsilon: float = 0.1,
) -> list[dict]:
    """
    Run warm-start QAOA on all subproblems with distributed dispatch.

    For each cluster:
      1. Solve GW SDP relaxation + rounding
      2. Compute warm-start angles from GW solution
      3. Run WS-QAOA with Egger modified mixer
    """
    global _BACKEND, _NUM_QPUS
    if _BACKEND is None:
        setup_mqpu()

    results = []
    optimal_params_list = []
    ws_angles_list = []

    for c, sp in enumerate(subproblems):
        Q = sp["qubo"]
        n = Q.shape[0]
        qpu_id = c % _NUM_QPUS

        t0 = time.perf_counter()

        gw_x, gw_obj = gw_solve(Q, budget=sp["budget"], num_trials=1000, seed=seed + c)
        gw_bs = "".join(str(int(b)) for b in gw_x)

        ws_angles = compute_ws_angles(gw_x, epsilon=epsilon)
        ws_angles_list.append(ws_angles)

        res = run_ws_qaoa(
            Q, n, layer_count, ws_angles,
            use_ws_mixer=True,
            seed=seed + c, shots=shots, maxiter=maxiter,
        )
        elapsed = time.perf_counter() - t0

        qaoa_x = np.array([int(b) for b in res["best_bitstring"]], dtype=float)
        sampled_energy = float(qaoa_x @ Q @ qaoa_x)

        optimal_params_list.append(res["optimal_params"])
        results.append({
            "cluster_id": sp["cluster_id"],
            "best_bitstring": res["best_bitstring"],
            "sampled_energy": sampled_energy,
            "optimal_energy": res["optimal_energy"],
            "optimal_params": res["optimal_params"],
            "counts": res["counts"],
            "timing_s": elapsed,
            "qpu_id": qpu_id,
            "gw_bitstring": gw_bs,
            "gw_energy": gw_obj,
            "ws_angles": ws_angles,
        })

    if _BACKEND == "mqpu":
        futures = []
        for c, sp in enumerate(subproblems):
            qpu_id = c % _NUM_QPUS
            future = cudaq.sample_async(
                kernel_ws_qaoa,
                sp["qubit_count"], layer_count,
                list(optimal_params_list[c]),
                ws_angles_list[c],
                sp["edges_src"], sp["edges_tgt"], sp["coeffs"],
                sp["single_qubit_indices"], sp["single_qubit_coeffs"],
                shots_count=shots, qpu_id=qpu_id)
            futures.append(future)

        for i, future in enumerate(futures):
            counts = future.get()
            sp = subproblems[i]
            counts_dict = _sample_result_to_dict(counts, sp["qubit_count"])
            counts_lsb = {bs[::-1]: c for bs, c in counts_dict.items()}
            best_bs = max(counts_lsb, key=counts_lsb.get)
            results[i]["best_bitstring"] = best_bs
            results[i]["counts"] = counts_lsb
            qaoa_x = np.array([int(b) for b in best_bs], dtype=float)
            results[i]["sampled_energy"] = float(qaoa_x @ sp["qubo"] @ qaoa_x)

    return results
