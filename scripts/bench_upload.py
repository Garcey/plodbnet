#!/usr/bin/env python
"""Break rollout._PinnedStepH2D.upload_rows (the learner's per-step packed
upload) into its pieces at the vMin shape: 7,333 tables, ~5,000 learner rows.
Diagnostic only; needs CUDA.

    .venv/bin/python scripts/bench_upload.py
"""

from __future__ import annotations

import os
import resource
import time

os.environ.setdefault("PLO5_RUST_ENCODER", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from plo5bp.compact_obs import MINIMAL_LAYOUT, pack_rows_into  # noqa: E402
from plo5bp.compact_obs import unpack as unpack_compact  # noqa: E402
from plo5bp.config import GameConfig  # noqa: E402
from plo5bp.env_batched import BatchedBombPotEnv  # noqa: E402
from plo5bp.rollout import _PinnedStepH2D  # noqa: E402


def main() -> None:
    dev = torch.device("cuda")
    n = 7333
    env = BatchedBombPotEnv(n, GameConfig(num_seats=6, starting_stack=60 * 10_000,
                                          ante=30_000, bb=10_000),
                            opp_outcome_mc=0, obs_mode="minimal")
    rng = np.random.default_rng(0)
    env.reset_batch(rng.integers(0, 2**62, size=n, dtype=np.uint64),
                    rng.integers(0, 6, size=n).astype(np.uint8))
    rows = np.nonzero(~env._dones & (rng.random(n) < 0.7))[0].astype(np.int64)
    sizing = np.zeros((n, 4), dtype=np.int64)
    h2d = _PinnedStepH2D(n, env.obs_dim, dev, n_slots=3, layout=MINIMAL_LAYOUT)
    lay = MINIMAL_LAYOUT
    k = rows.size
    print(f"{k} rows")
    for _ in range(20):
        h2d.upload_rows(env._obs, rows, env._gate_mask, sizing, slot=0)
    torch.cuda.synchronize()
    reps = 300
    t = {"pack": 0.0, "gm_sz": 0.0, "h2d_launch": 0.0, "unpack_launch": 0.0, "sync": 0.0}
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    for _ in range(reps):
        a = time.perf_counter()
        pack_rows_into(env._obs, rows, lay, h2d.bits_h[0].numpy(), h2d.real_h[0].numpy(), 0)
        b = time.perf_counter()
        h2d.gm_h[0].numpy()[:k] = env._gate_mask[rows]
        h2d.sz_h[0].numpy()[:k] = sizing[rows]
        c = time.perf_counter()
        bits_t = h2d.bits_h[0, :k].to(dev, non_blocking=True)
        real_t = h2d.real_h[0, :k].to(dev, non_blocking=True)
        m_t = h2d.gm_h[0, :k].to(dev, non_blocking=True)
        b_t = h2d.sz_h[0, :k].to(dev, non_blocking=True)
        d = time.perf_counter()
        o_t = unpack_compact(bits_t, real_t, lay)
        e = time.perf_counter()
        torch.cuda.synchronize()
        f = time.perf_counter()
        for key, v in zip(t, (b - a, c - b, d - c, e - d, f - e)):
            t[key] += v
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    print("  ".join(f"{key} {1e3 * v / reps:.3f} ms" for key, v in t.items()))
    print(f"kernel CPU {1e3 * (ru1.ru_stime - ru0.ru_stime) / reps:.3f} ms/iter, "
          f"user CPU {1e3 * (ru1.ru_utime - ru0.ru_utime) / reps:.3f} ms/iter")
    a = time.perf_counter()
    for _ in range(reps):
        h2d.upload_rows(env._obs, rows, env._gate_mask, sizing, slot=0)
    torch.cuda.synchronize()
    print(f"upload_rows total {1e3 * (time.perf_counter() - a) / reps:.3f} ms")


if __name__ == "__main__":
    main()
