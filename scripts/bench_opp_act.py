#!/usr/bin/env python
"""Where does the batched rollout's opponent step spend its time? Times the
stacked (vmapped) forward and the sampling tail of rollout._StackedOpponents
separately at a realistic batch (8 pool snapshots of CKPT's actor, ~2,000
opponent rows over 7,333 tables), and lists the aten ops the vmapped forward
dispatches. Diagnostic only.

    .venv/bin/python scripts/bench_opp_act.py checkpoints/vMin2_19.pt
"""

from __future__ import annotations

import copy
import os
import sys
import time

os.environ.setdefault("PLO5_RUST_ENCODER", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from plo5bp.config import GameConfig  # noqa: E402
from plo5bp.env_batched import BatchedBombPotEnv  # noqa: E402
from plo5bp.network import build_actor_from_state_dict  # noqa: E402
from plo5bp.rollout import _StackedOpponents  # noqa: E402


def main() -> None:
    torch.distributions.Distribution.set_default_validate_args(False)  # as train.py
    path = sys.argv[1]
    dev = torch.device("cuda")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    base = build_actor_from_state_dict(ck["model"], int(cfg["hidden_dim"]), int(cfg["num_layers"]))
    models = []
    g = torch.Generator().manual_seed(0)
    for i in range(8):
        m = copy.deepcopy(base)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn(p.shape, generator=g) * 1e-3 * i)
        models.append(m.to(dev).eval())
    stk = _StackedOpponents(models)

    n = 7333
    env = BatchedBombPotEnv(n, GameConfig(num_seats=6, starting_stack=60 * 10_000,
                                          ante=30_000, bb=10_000),
                            opp_outcome_mc=0, obs_mode=cfg.get("obs_mode", "minimal"))
    rng = np.random.default_rng(0)
    env.reset_batch(rng.integers(0, 2**62, size=n, dtype=np.uint64),
                    rng.integers(0, 6, size=n).astype(np.uint8))
    rows = np.nonzero(~env._dones & (rng.random(n) < 0.3))[0]
    snap = rng.integers(0, 8, size=rows.size)
    order = np.argsort(snap, kind="stable")
    rows, snap = rows[order], snap[order]
    counts = np.bincount(snap, minlength=8)
    j = np.arange(rows.size) - (np.cumsum(counts) - counts)[snap]
    safe = np.where(env._actors >= 0, env._actors, 0).astype(np.intp)
    to_call = np.maximum(env._bet_to_call.astype(np.int64)
                         - env._street_commit[np.arange(n), safe].astype(np.int64), 0)
    sizing = np.stack([env._min_raise.astype(np.int64), env._max_raise.astype(np.int64),
                       env._pot.astype(np.int64), to_call], axis=-1)
    o = torch.from_numpy(env._obs[rows]).to(dev)
    m_ = torch.from_numpy(env._gate_mask[rows]).to(dev)
    b = torch.from_numpy(sizing[rows]).to(dev)
    g_t = torch.from_numpy(snap).to(dev)
    j_t = torch.from_numpy(j).to(dev)
    n_max = int(counts.max())
    print(f"{rows.size} opponent rows over 8 snapshots, n_max {n_max}")

    def fwd():
        obs_pad = o.new_zeros((stk.n, n_max, o.shape[-1]))
        obs_pad[g_t, j_t] = o
        gm_pad = m_.new_zeros((stk.n, n_max, m_.shape[-1]))
        gm_pad[g_t, j_t] = m_
        return stk._vforward(stk.params, stk.buffers, obs_pad, gm_pad)

    with torch.inference_mode():
        for _ in range(20):
            heads = fwd()
            stk.template._act_from_heads(*(h[g_t, j_t] for h in heads), b)
        torch.cuda.synchronize()
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            heads = fwd()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sel = [h[g_t, j_t] for h in heads]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(reps):
            out = stk.template._act_from_heads(*sel, b)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        for _ in range(reps):
            base_out = models[0].act(o, m_, b)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
    print(f"vmapped forward     {1e3 * (t1 - t0) / reps:.3f} ms")
    print(f"sampling tail       {1e3 * (t3 - t2) / reps:.3f} ms")
    print(f"plain act (1 model) {1e3 * (t4 - t3) / reps:.3f} ms  (forward + tail, for scale)")

    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
        heads = fwd()
    ops = [e for e in prof.events() if e.name.startswith("aten::")]
    print(f"vmapped forward dispatched {len(ops)} aten ops (incl. nested):")
    seen = {}
    for e in ops:
        seen[e.name] = seen.get(e.name, 0) + 1
    for name, c in sorted(seen.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {c:4d} {name}")
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
        out = stk.template._act_from_heads(*sel, b)
    ops = [e for e in prof.events() if e.name.startswith("aten::")]
    print(f"sampling tail dispatched {len(ops)} aten ops (incl. nested)")


if __name__ == "__main__":
    main()
