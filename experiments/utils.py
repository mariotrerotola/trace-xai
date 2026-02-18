"""Shared utilities: JSON encoding, results I/O, and parallel execution."""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return list(o)
        return super().default(o)


def save_results(results, filename):
    os.makedirs("results", exist_ok=True)
    path = f"results/{filename}"
    with open(path, "w") as f:
        json.dump(results, f, cls=NpEncoder, indent=2)
    print(f"Results saved to {path}")


def run_parallel(
    fn: Callable[..., Any],
    tasks: Sequence[dict],
    *,
    max_workers: int | None = None,
    desc: str = "tasks",
    use_processes: bool = False,
) -> list[dict]:
    """Execute *fn* on each task dict in parallel.

    By default uses threads (safe because sklearn releases the GIL for
    heavy C/Cython computation). Set ``use_processes=True`` for CPU-bound
    pure-Python workloads — but note that ``fn`` must be importable
    (top-level in a module) for pickling.

    Parameters
    ----------
    fn : callable
        Function that accepts keyword arguments matching task dict keys
        and returns a dict (or list of dicts) of results.
    tasks : list of dict
        Each dict is unpacked as ``fn(**task)``.
    max_workers : int or None
        Number of parallel workers. ``None`` lets the executor choose.
    desc : str
        Label printed in progress messages.
    use_processes : bool
        If True, use ProcessPoolExecutor instead of ThreadPoolExecutor.

    Returns
    -------
    list of dict
        Collected results (order may differ from input).
    """
    if not tasks:
        return []

    n = len(tasks)
    results: list[dict] = []
    done = 0
    t0 = time.perf_counter()

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with Executor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, **task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            done += 1
            try:
                res = future.result()
                if isinstance(res, list):
                    results.extend(res)
                else:
                    results.append(res)
            except Exception as exc:
                idx = futures[future]
                print(f"  [{desc}] Task {idx} failed: {exc}")
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (n - done) if done else 0
            print(
                f"  [{desc}] {done}/{n} done  "
                f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)"
            )

    return results
