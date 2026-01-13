from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set


def _parse_cpu_affinity(spec: str) -> Set[int]:
    cpus: Set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(f"Invalid cpu range: {part!r}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("Empty cpu_affinity")
    return cpus


def _set_cpu_affinity_all_threads(cpus: Set[int]) -> None:
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("os.sched_setaffinity is not available on this platform")
    task_dir = Path("/proc/self/task")
    if task_dir.exists():
        for tid in task_dir.iterdir():
            if tid.name.isdigit():
                os.sched_setaffinity(int(tid.name), cpus)
        return
    os.sched_setaffinity(0, cpus)


def apply_cpu_affinity(cpu_affinity: Optional[str]) -> Optional[Sequence[int]]:
    """Pin current process (and its threads) to a subset of CPU cores.

    Args:
        cpu_affinity: e.g. "0-15" or "0-7,16-23".
    Returns:
        Sorted CPU ids applied, or None if not requested.
    """
    if cpu_affinity is None:
        return None
    cpus = _parse_cpu_affinity(cpu_affinity)
    _set_cpu_affinity_all_threads(cpus)
    return sorted(cpus)


def apply_thread_env(*, omp_threads: Optional[int] = None, mkl_threads: Optional[int] = None) -> None:
    if omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(int(omp_threads))
    if mkl_threads is not None:
        os.environ["MKL_NUM_THREADS"] = str(int(mkl_threads))


def apply_torch_threads(*, torch_threads: Optional[int] = None, interop_threads: Optional[int] = None) -> None:
    if torch_threads is None and interop_threads is None:
        return
    import torch

    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    if interop_threads is not None:
        torch.set_num_interop_threads(int(interop_threads))


def apply_cuda_visible_devices(gpu_id: Optional[int]) -> None:
    """Restrict CUDA to a single physical GPU by id (before CUDA init)."""
    if gpu_id is None:
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))


@dataclass(frozen=True)
class RuntimeConfigResult:
    cpu_affinity: Optional[Sequence[int]]


def configure_runtime(
    *,
    gpu_id: Optional[int] = None,
    cpu_affinity: Optional[str] = None,
    omp_threads: Optional[int] = None,
    mkl_threads: Optional[int] = None,
    torch_threads: Optional[int] = None,
    torch_interop_threads: Optional[int] = None,
) -> RuntimeConfigResult:
    apply_cuda_visible_devices(gpu_id)
    applied_cpus = apply_cpu_affinity(cpu_affinity)
    apply_thread_env(omp_threads=omp_threads, mkl_threads=mkl_threads)
    apply_torch_threads(torch_threads=torch_threads, interop_threads=torch_interop_threads)
    return RuntimeConfigResult(cpu_affinity=applied_cpus)

