"""
Phase 0 — Resource instrumentation harness for the Warp-vs-CalculiX benchmark.

Measures, around an arbitrary code region:
  - wall-clock time
  - aggregate CPU utilization (%, 0..100), sampled over time
  - per-core CPU utilization (to catch a single hot core hiding under a low average)
  - GPU utilization and peak memory (via NVML, optional)

Purpose: turn the supervisor's two requirements into *objective numbers* instead
of trusting an agent's claim that "it only uses the GPU":

  Req 1 : during the GPU solve, aggregate CPU util must stay < 20%   -> result.cpu_under(20)
  Req 2 : GPU wall-clock <= full-CPU (ccx) wall-clock                -> compare_walltime(...)

Dependencies:
  pip install psutil          # required
  pip install nvidia-ml-py    # optional, for GPU metrics (module name: pynvml)
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import psutil
except ImportError as e:  # pragma: no cover
    raise ImportError("psutil is required: pip install psutil") from e

# GPU metrics are optional; degrade gracefully if NVML is unavailable.
try:
    import pynvml  # provided by the `nvidia-ml-py` package
    _HAVE_NVML = True
except ImportError:
    _HAVE_NVML = False


@dataclass
class Sample:
    t: float                       # seconds since region start
    cpu_total: float               # aggregate CPU %, 0..100 (mean of per-core)
    cpu_per_core: List[float]      # per-core %
    gpu_util: Optional[float]      # %
    gpu_mem_mb: Optional[float]    # MiB used


@dataclass
class MeasureResult:
    name: str
    wall_s: float
    cpu_avg: float                 # time-averaged aggregate CPU %
    cpu_max: float                 # peak aggregate CPU %
    per_core_max: float            # peak of any single core
    gpu_util_avg: Optional[float]
    gpu_mem_peak_mb: Optional[float]
    n_samples: int
    samples: List[Sample] = field(default_factory=list, repr=False)

    # --- Req 1: CPU must stay under a threshold during the region ---
    def cpu_under(self, threshold: float = 20.0) -> bool:
        return self.cpu_avg < threshold

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "wall_s": self.wall_s,
            "cpu_avg": self.cpu_avg,
            "cpu_max": self.cpu_max,
            "per_core_max": self.per_core_max,
            "gpu_util_avg": self.gpu_util_avg,
            "gpu_mem_peak_mb": self.gpu_mem_peak_mb,
            "n_samples": self.n_samples,
        }

    def summary(self, cpu_threshold: float = 20.0) -> str:
        gpu = (
            f"gpu_util_avg={self.gpu_util_avg:5.1f}%  "
            f"gpu_mem_peak={self.gpu_mem_peak_mb:8.1f} MiB"
            if self.gpu_util_avg is not None
            else "gpu: (NVML unavailable)"
        )
        verdict = "PASS" if self.cpu_under(cpu_threshold) else "FAIL"
        return (
            f"[{self.name}] wall={self.wall_s:8.3f} s   "
            f"cpu_avg={self.cpu_avg:5.1f}%  cpu_max={self.cpu_max:5.1f}%  "
            f"core_max={self.per_core_max:5.1f}%   {gpu}\n"
            f"    Req1 (cpu_avg < {cpu_threshold:.0f}%): {verdict}   "
            f"(samples={self.n_samples})"
        )


class ResourceMonitor:
    """Background sampler. Polls CPU (psutil) and GPU (NVML) at a fixed interval."""

    def __init__(self, interval: float = 0.05, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self._samples: List[Sample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        if _HAVE_NVML:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._handle = None  # keep going without GPU metrics

    def _read_gpu(self):
        if self._handle is None:
            return None, None
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            m = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return float(u.gpu), m.used / (1024.0 ** 2)
        except Exception:
            return None, None

    def _run(self):
        # Prime psutil so the first real reading covers a proper interval
        # (the first call after import returns a meaningless 0.0 / instant value).
        psutil.cpu_percent(percpu=True)
        t0 = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval)
            per = psutil.cpu_percent(percpu=True)          # single call -> no state clash
            total = sum(per) / len(per) if per else 0.0    # aggregate = mean of cores
            gutil, gmem = self._read_gpu()
            self._samples.append(
                Sample(time.perf_counter() - t0, total, per, gutil, gmem)
            )

    def start(self):
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[Sample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self._samples


def _summarize(name: str, wall: float, samples: List[Sample]) -> MeasureResult:
    if not samples:
        return MeasureResult(name, wall, 0.0, 0.0, 0.0, None, None, 0, [])
    cpu = [s.cpu_total for s in samples]
    cpu_avg = sum(cpu) / len(cpu)
    cpu_max = max(cpu)
    per_core_max = max(
        (max(s.cpu_per_core) for s in samples if s.cpu_per_core), default=0.0
    )
    gutil = [s.gpu_util for s in samples if s.gpu_util is not None]
    gmem = [s.gpu_mem_mb for s in samples if s.gpu_mem_mb is not None]
    return MeasureResult(
        name=name,
        wall_s=wall,
        cpu_avg=cpu_avg,
        cpu_max=cpu_max,
        per_core_max=per_core_max,
        gpu_util_avg=(sum(gutil) / len(gutil)) if gutil else None,
        gpu_mem_peak_mb=max(gmem) if gmem else None,
        n_samples=len(samples),
        samples=samples,
    )


@contextmanager
def measure(name: str, interval: float = 0.05, gpu_index: int = 0, verbose: bool = True):
    """
    Wrap a code region and collect timing + CPU/GPU utilization.

        with measure("warp_solve") as m:
            solve_on_gpu(...)
        res = m["result"]
        assert res.cpu_under(20.0)   # Req 1

    Keep `interval` small (0.02-0.10 s) for short solves so you get enough samples.
    """
    mon = ResourceMonitor(interval=interval, gpu_index=gpu_index)
    holder: dict = {}
    mon.start()
    t0 = time.perf_counter()
    try:
        yield holder
    finally:
        wall = time.perf_counter() - t0
        samples = mon.stop()
        result = _summarize(name, wall, samples)
        holder["result"] = result
        if verbose:
            print(result.summary())


def compare_walltime(gpu: MeasureResult, cpu: MeasureResult) -> dict:
    """
    Req 2: GPU wall-clock must be <= full-CPU (ccx) wall-clock.
    Returns a small verdict dict (also handy to dump to JSON for the report).
    """
    speedup = cpu.wall_s / gpu.wall_s if gpu.wall_s > 0 else float("inf")
    verdict = {
        "gpu_wall_s": gpu.wall_s,
        "cpu_wall_s": cpu.wall_s,
        "speedup_cpu_over_gpu": speedup,       # >= 1.0 means Req 2 satisfied
        "req2_pass": gpu.wall_s <= cpu.wall_s,
    }
    print(
        f"[Req2] gpu={gpu.wall_s:.3f}s  cpu={cpu.wall_s:.3f}s  "
        f"speedup={speedup:.2f}x  -> {'PASS' if verdict['req2_pass'] else 'FAIL'}"
    )
    return verdict


def save_results(path: str, **named_results) -> None:
    """Dump one or more MeasureResult (and/or plain dicts) to a JSON file."""
    out = {}
    for k, v in named_results.items():
        out[k] = v.as_dict() if isinstance(v, MeasureResult) else v
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    # Smoke test: a CPU-bound loop should light up the CPU; sleeping should not.
    with measure("busy_cpu", interval=0.05) as m:
        x = 0
        for _ in range(20_000_000):
            x += 1
    with measure("idle_sleep", interval=0.05) as m2:
        time.sleep(1.0)
    print("\nInstrumentation harness OK.")
