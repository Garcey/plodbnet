"""Run telemetry: the per-update metrics file, the heartbeat, and the 1 Hz
CPU / RAM / GPU resource sampler."""

from __future__ import annotations

import collections
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import torch


def _jsonable(x: object) -> object:
    """Strict-JSON values: NaN/Inf -> None, numpy/torch scalars -> Python,
    anything else unknown -> str."""
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and callable(x.item):
        try:
            return _jsonable(x.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(x)


class MetricsWriter:
    """runs/<stem>.metrics.jsonl (2026-09-28, ML-005): ONE JSON object per
    update -- every PPOStats field, the critic's value health, timings, rows,
    lr and entropy coefficients, per-tier F/T/R, pool, rollbacks, VRAM -- so
    tools read numbers instead of regex-parsing the log (whose layout changes).
    Appends and flushes one line per update; strict JSON (NaN -> null).
    `scripts/plot_metrics.py` charts it."""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        except OSError as e:  # a full disk must not kill the run from here
            print(f"[metrics] could not append to {self.path}: {e!r}")


def read_metrics(path: "str | Path") -> list[dict]:
    """Every record of a metrics JSONL file (a torn last line is skipped)."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


class Heartbeat:
    """runs/<stem>.heartbeat (2026-09-28, ML-024): rewritten atomically after
    every update with the update number, the time and the count of
    consecutive rolled-back updates. A guardian that watches only the PID
    cannot tell a hung trainer or a rollback livelock from a working one; a
    stale heartbeat (or `consecutive_rollbacks` climbing) can."""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def beat(self, **fields: object) -> None:
        rec = {"time": time.time(), "pid": os.getpid(), **fields}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(_jsonable(rec), sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[heartbeat] could not write {self.path}: {e!r}")


class _ResourceSampler:
    """Background 1 Hz samples of CPU / RAM / GPU util / VRAM during a run.

    Writes JSONL to `out_path` (train.py: runs/<stem>.resources.jsonl -- one
    file per run, not one shared by every trainer on the pod). Phase labels via
    set_phase so each sample is attributable to rollout vs optimize. Uses
    psutil when available; falls back to /proc + nvidia-smi. Keeps at most
    `max_samples` rows in memory for `summarize` (it used to keep every row
    for the life of the run: ~100 MB of RAM a day at 1 Hz).
    """

    def __init__(
        self,
        out_path: str | Path = "runs/profile_resources.jsonl",
        interval_s: float = 1.0,
        max_samples: int = 21_600,
    ) -> None:
        self.out_path = Path(out_path)
        self.interval_s = float(interval_s)
        self._phase = "init"
        self._update = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil = None
        self._proc = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception:
            self._psutil = None
            self._proc = None
        self._n_logical = os.cpu_count() or 1
        self._cgroup_quota_cpus = self._read_cgroup_quota_cpus()
        self.samples: "collections.deque[dict]" = collections.deque(maxlen=int(max_samples))

    @staticmethod
    def _read_cgroup_quota_cpus() -> float | None:
        for path in ("/sys/fs/cgroup/cpu.max",):
            try:
                raw = Path(path).read_text().strip().split()
                if len(raw) >= 2 and raw[0] != "max":
                    quota, period = float(raw[0]), float(raw[1])
                    if period > 0:
                        return quota / period
            except (OSError, ValueError):
                pass
        try:
            q = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0 and p > 0:
                return q / p
        except (OSError, ValueError):
            pass
        return None

    def set_phase(self, phase: str, update: int | None = None) -> None:
        self._phase = phase
        if update is not None:
            self._update = int(update)

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("")
        self._thread = threading.Thread(
            target=self._loop, name="resource-sampler", daemon=True
        )
        self._thread.start()
        print(
            f"[profile-resources] sampling every {self.interval_s:.1f}s -> "
            f"{self.out_path}  "
            f"(cgroup_quota_cpus={self._cgroup_quota_cpus!s}, "
            f"logical_cpus={self._n_logical})"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2.0)
            self._thread = None

    def _sample_gpu(self) -> tuple[float | None, float | None, float | None]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2.0,
            ).strip().splitlines()
            if not out:
                return None, None, None
            parts = [p.strip() for p in out[0].split(",")]
            return float(parts[0]), float(parts[1]), float(parts[2])
        except Exception:
            return None, None, None

    def _sample_cpu_ram(self) -> dict:
        row: dict = {}
        if self._psutil is not None and self._proc is not None:
            proc_cpu_1core = float(self._proc.cpu_percent(None))
            host_cpu = float(self._psutil.cpu_percent(None))
            row["proc_cpu_pct_1core"] = round(proc_cpu_1core, 2)
            row["proc_cpu_pct_of_logical"] = round(
                proc_cpu_1core / max(1, self._n_logical), 2
            )
            if self._cgroup_quota_cpus and self._cgroup_quota_cpus > 0:
                row["proc_cpu_pct_of_quota"] = round(
                    proc_cpu_1core / self._cgroup_quota_cpus, 2
                )
            row["host_cpu_pct"] = round(host_cpu, 2)
            mem = self._proc.memory_info()
            row["proc_rss_gb"] = round(mem.rss / (1024 ** 3), 3)
            vm = self._psutil.virtual_memory()
            row["host_ram_used_gb"] = round(vm.used / (1024 ** 3), 3)
            row["host_ram_total_gb"] = round(vm.total / (1024 ** 3), 3)
            row["host_ram_pct"] = round(vm.percent, 2)
        else:
            try:
                with open("/proc/self/status", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = float(line.split()[1])
                            row["proc_rss_gb"] = round(kb / (1024 ** 2), 3)
                            break
            except OSError:
                pass
        if self._cgroup_quota_cpus is not None:
            row["cgroup_quota_cpus"] = round(self._cgroup_quota_cpus, 2)
        row["logical_cpus"] = self._n_logical
        return row

    def _sample_torch_vram(self) -> dict:
        row: dict = {}
        if torch.cuda.is_available():
            try:
                row["torch_alloc_gb"] = round(
                    torch.cuda.memory_allocated() / (1024 ** 3), 3
                )
                row["torch_reserved_gb"] = round(
                    torch.cuda.memory_reserved() / (1024 ** 3), 3
                )
            except Exception:
                pass
        return row

    def _loop(self) -> None:
        t0 = time.time()
        while not self._stop.is_set():
            ts = time.time()
            gpu_util, gpu_used, gpu_total = self._sample_gpu()
            row = {
                "t": round(ts - t0, 3),
                "wall": ts,
                "phase": self._phase,
                "update": self._update,
            }
            row.update(self._sample_cpu_ram())
            row.update(self._sample_torch_vram())
            if gpu_util is not None:
                row["gpu_util_pct"] = gpu_util
            if gpu_used is not None:
                row["gpu_mem_used_mib"] = gpu_used
            if gpu_total is not None:
                row["gpu_mem_total_mib"] = gpu_total
            self.samples.append(row)
            try:
                with self.out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass
            self._stop.wait(self.interval_s)

    def summarize(self) -> None:
        if not self.samples:
            print("[profile-resources] no samples collected")
            return
        by_phase: dict[str, list[dict]] = {}
        for s in self.samples:
            by_phase.setdefault(str(s.get("phase", "?")), []).append(s)

        def _stats(vals: list[float]) -> str:
            if not vals:
                return "n/a"
            vals = sorted(vals)
            n = len(vals)
            mean = sum(vals) / n
            p50 = vals[n // 2]
            p95 = vals[min(n - 1, int(n * 0.95))]
            return f"mean={mean:5.1f}  p50={p50:5.1f}  p95={p95:5.1f}  n={n}"

        print("\n===== resource sampler summary (per phase) =====")
        for phase, rows in by_phase.items():
            print(f"  phase={phase!r}  samples={len(rows)}")
            for key, label in (
                ("proc_cpu_pct_of_quota", "CPU % of cgroup quota"),
                ("proc_cpu_pct_of_logical", "CPU % of logical cores"),
                ("host_cpu_pct", "host CPU %"),
                ("gpu_util_pct", "GPU util %"),
                ("gpu_mem_used_mib", "GPU mem MiB"),
                ("torch_alloc_gb", "torch alloc GiB"),
                ("torch_reserved_gb", "torch reserved GiB"),
                ("proc_rss_gb", "proc RSS GiB"),
            ):
                vals = [
                    float(r[key])
                    for r in rows
                    if key in r and r[key] is not None
                ]
                if vals:
                    print(f"    {label:28s}  {_stats(vals)}")
        print(f"[profile-resources] raw JSONL -> {self.out_path}")
