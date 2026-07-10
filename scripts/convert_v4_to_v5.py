"""Convert a v4 (ordinal-logistic head) checkpoint into a v5 (mixture)
checkpoint — the sanctioned warm-start path for the v5 stem.

Two function-preserving transforms compose (V5_DESIGN.md §2.4 / §3.3):

1. HEAD: v4's (mu_raw, s_raw) rows become mixture component 0; minor
   components get ZERO weights with bias-driven spread (mu_raw ±1 →
   locations near the min / pot regions) and mixture logits (w0_logit,
   -1.5, ...) → w ≈ (0.92, 0.04, 0.04) after the ε-floor. The converted
   policy is v4 with a ~8% smear from the spread minor components — an
   entropy re-warm, not a policy break.
2. OBS WIDTH: first-layer weight matrices gain zero-initialized columns
   for any observation dims added since the checkpoint was trained
   (appended at the obs tail, so actor columns append at the end and
   critic columns insert BEFORE its 260-wide opponent-multihot block).
   Zero columns contribute exactly nothing → identical function at
   step 0.

The critic also gains the zero-init dueling `adv_head` (Q(s,a) = V +
A, q_actions = 2 + anchor_count) so the v5 stem's Q-auxiliary loss can
run without a later checkpoint break.

Training-state metadata that must NOT carry into a new stem is
stripped: anneal state (v5 re-seeds entropy high — the anneal is
one-way down), update counter (fresh stem numbering + LR warmup), and
pool_member_updates (pool siblings are v4 files; convert each with
this script if you want the prior pool seeded, e.g.
`for f in checkpoints/vFour4_9*.pt; do ... convert_v4_to_v5 $f ...`).

Usage:
    .venv/Scripts/python scripts/convert_v4_to_v5.py \
        checkpoints/vFour4_915.pt checkpoints/vFive1_seed.pt \
        [--mixture-k 3] [--w0-logit 3.0]

train.py's head_version guard stays strict on purpose: this script is
the only v4→v5 bridge, and the emitted file is a fully valid v5
checkpoint (`--sizing-head mixture --load-checkpoint <out>` just works).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from plo5bp.encoding import OBS_DIM  # noqa: E402
from plo5bp.network import (  # noqa: E402
    ActorCriticV4,
    ActorCriticV5,
    build_actor_from_state_dict,
    state_dict_anchor_count,
    state_dict_obs_dim,
)

_OPP_DIM = 5 * 52


def _first_layer_key(sd: dict, prefix: str = "") -> str:
    for k in (f"{prefix}torso.0.0.weight", f"{prefix}torso.0.weight"):
        if k in sd:
            return k
    raise SystemExit(f"no first-layer weight under '{prefix}torso.0' — not a plo5bp net")


def _pad_columns(sd: dict, key: str, insert_at: int, pad: int) -> None:
    """Insert `pad` zero columns into sd[key] at column `insert_at`."""
    w = sd[key]
    zeros = torch.zeros(w.shape[0], pad, dtype=w.dtype)
    sd[key] = torch.cat([w[:, :insert_at], zeros, w[:, insert_at:]], dim=1)


def convert(
    ckpt: dict, mixture_k: int = 3, w0_logit: float = 3.0
) -> tuple[dict, list[str]]:
    """Return (converted checkpoint dict, log lines). Pure — no I/O."""
    log: list[str] = []
    if int(ckpt.get("head_version", 1)) != ActorCriticV4.head_version:
        raise SystemExit(
            f"expected a v4 checkpoint (head_version="
            f"{ActorCriticV4.head_version}), got head_version="
            f"{ckpt.get('head_version')!r}"
        )
    sd = {k: v.clone() for k, v in ckpt["model"].items()}
    if "size_head.weight" not in sd:
        raise SystemExit("checkpoint model has no size_head — not a v4 actor")

    # --- 1. obs-width padding (actor: obs-only input, append at end) ---
    old_obs = state_dict_obs_dim(sd)
    if old_obs > OBS_DIM:
        raise SystemExit(
            f"checkpoint obs width {old_obs} exceeds current OBS_DIM {OBS_DIM}"
        )
    if old_obs < OBS_DIM:
        _pad_columns(sd, _first_layer_key(sd), old_obs, OBS_DIM - old_obs)
        log.append(f"actor obs width padded {old_obs} -> {OBS_DIM} (zero columns)")

    # --- 2. head conversion ---
    k = int(mixture_k)
    size_w, size_b = sd.pop("size_head.weight"), sd.pop("size_head.bias")
    hidden = size_w.shape[1]
    mix_w = torch.zeros(3 * k, hidden, dtype=size_w.dtype)
    mix_b = torch.zeros(3 * k, dtype=size_b.dtype)
    mix_w[0], mix_b[0] = size_w[0], size_b[0]          # component-0 mu = v4 mu
    mix_w[k], mix_b[k] = size_w[1], size_b[1]          # component-0 s  = v4 s
    if k > 1:
        # Minor components: zero weights, mu_raw biases spread across the
        # ladder (tanh(±1) ≈ ±0.76 → near the min / pot regions for K=3),
        # s_raw bias 0 (mid scale), logits (w0, -1.5, ...).
        mix_b[1:k] = torch.linspace(-1.0, 1.0, k - 1)
        mix_b[2 * k] = float(w0_logit)
        mix_b[2 * k + 1: 3 * k] = -1.5
    sd["mix_head.weight"], sd["mix_head.bias"] = mix_w, mix_b
    log.append(
        f"size_head -> mix_head (K={k}, w0_logit={w0_logit}): component 0 = v4"
    )

    # --- 3. critic: obs-width padding (insert BEFORE the opp block) +
    #        zero-init dueling adv_head -------------------------------
    out = dict(ckpt)
    csd = {k2: v.clone() for k2, v in ckpt["critic"].items()}
    ckey = _first_layer_key(csd)
    c_old_obs = csd[ckey].shape[1] - _OPP_DIM
    if c_old_obs != old_obs:
        raise SystemExit(
            f"critic obs width {c_old_obs} != actor obs width {old_obs}"
        )
    if c_old_obs < OBS_DIM:
        _pad_columns(csd, ckey, c_old_obs, OBS_DIM - c_old_obs)
        log.append(f"critic obs width padded {c_old_obs} -> {OBS_DIM}")
    anchor_count = state_dict_anchor_count(sd)
    if anchor_count is None:
        raise SystemExit("could not sniff anchor count from refine_head")
    q_actions = 2 + anchor_count
    if "adv_head.weight" not in csd:
        c_hidden = csd[ckey].shape[0]
        csd["adv_head.weight"] = torch.zeros(q_actions, c_hidden)
        csd["adv_head.bias"] = torch.zeros(q_actions)
        log.append(f"critic adv_head added (zero-init, q_actions={q_actions})")

    # --- 4. metadata ---
    out["model"] = sd
    out["critic"] = csd
    out["head_version"] = ActorCriticV5.head_version
    for stale in (
        "anneal_tier_ent", "anneal_baseline", "anneal_block_acc",
        "update_counter", "pool_member_updates",
        # model_ema is the v4-shaped (size_head, old obs width) EMA
        # reference — loading it into a v5 magnet reference would
        # shape-crash. A fresh v5 stem re-inits its EMA from the loaded
        # weights and ramps in.
        "model_ema",
    ):
        if out.pop(stale, None) is not None:
            log.append(f"stripped {stale} (fresh v5 stem state)")
    return out, log


def _verify(orig_model_sd: dict, out: dict, hidden: int, layers: int) -> None:
    """Function-preservation probe: gate/value must match exactly on the
    original obs prefix; the anchor marginal deviates only by the minor-
    component smear. Random obs + a few sizing configs, CPU."""
    from plo5bp.sizing import anchor_grid_torch

    old = build_actor_from_state_dict(orig_model_sd, hidden, layers).eval()
    new = build_actor_from_state_dict(out["model"], hidden, layers).eval()
    old_obs_dim = state_dict_obs_dim(orig_model_sd)
    torch.manual_seed(0)
    obs_new = torch.randn(64, state_dict_obs_dim(out["model"]))
    obs_old = obs_new[:, :old_obs_dim]
    gm = torch.ones(64, 3, dtype=torch.bool)
    sizing = torch.tensor([[10000, 180000, 180000, 0]] * 64, dtype=torch.int64)
    with torch.no_grad():
        g_old, a_old, r_old, v_old = old(obs_old, gm)
        g_new, a_new, r_new, v_new = new(obs_new, gm)
        grid = anchor_grid_torch(sizing, new.anchor_spec)
        p_old = old._anchor_dist(a_old, grid).probs
        p_new = new._anchor_dist(a_new, grid).probs
    g_dev = float((g_old - g_new).abs().max())
    v_dev = float((v_old - v_new).abs().max())
    r_dev = float((r_old - r_new).abs().max())
    a_dev = float((p_old - p_new).abs().max())
    print(
        f"[verify] gate dev {g_dev:.2e}  value dev {v_dev:.2e}  "
        f"refine dev {r_dev:.2e}  anchor-marginal max dev {a_dev:.4f} "
        f"(expected ~= the minor-component smear)"
    )
    # Zero columns are exact in real arithmetic, but widening the matmul's
    # reduction dim (991 -> current OBS_DIM) changes the BLAS tiling and
    # therefore the f32 summation ORDER of the same nonzero products —
    # ~1e-5-scale noise at hidden 2048, not a conversion defect. Thresholds
    # allow kernel noise while still catching real wiring mistakes (a
    # mis-inserted column shows up at 1e-1+). Values are bb-scale, hence
    # the looser absolute bound there.
    if g_dev > 1e-3 or v_dev > 1e-2 or r_dev > 1e-3 or a_dev > 0.15:
        raise SystemExit("function preservation FAILED on gate/value/refine")
    print("[verify] function preservation OK (within f32 kernel noise)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="v4 checkpoint (.pt)")
    ap.add_argument("dst", help="output v5 checkpoint (.pt)")
    ap.add_argument("--mixture-k", type=int, default=3)
    ap.add_argument(
        "--w0-logit", type=float, default=3.0,
        help="component-0 mixture logit; higher = smaller smear "
        "(3.0 -> w0 ~= 0.92 after the 0.03 floor)",
    )
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    out, log = convert(ckpt, mixture_k=args.mixture_k, w0_logit=args.w0_logit)
    for line in log:
        print(f"[convert] {line}")
    if not args.no_verify:
        cfg = ckpt.get("config") or {}
        _verify(
            ckpt["model"], out,
            int(cfg.get("hidden_dim", 128)), int(cfg.get("num_layers", 2)),
        )
    torch.save(out, args.dst)
    print(f"[convert] wrote {args.dst} (head_version={out['head_version']})")


if __name__ == "__main__":
    main()
