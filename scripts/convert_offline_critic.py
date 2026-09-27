#!/usr/bin/env python
"""Offline-trained critic (scripts/critic_offline.py --save-dir) -> a
CentralCritic state dict train.py can start from (--critic-init), 2026-09-26.

The offline `Critic` (inp [+ in_norm] -> act -> residual blocks -> 51-bin head)
is exactly CentralCritic(act, in_norm, torso_layernorm=True, value_bins=51);
the dueling adv_head (absent offline) is added at zero init (Q == V), and V is
read out as the raw-space mean (v_raw) unless --symlog-readout.

    .venv/bin/python scripts/convert_offline_critic.py /root/critics/silu-2048x3.pt \
        checkpoints/critic_silu2048x3_u1290.pt [--q-actions 3]
"""

from __future__ import annotations

import argparse

import torch

from plo5bp.network import CentralCritic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--q-actions", type=int, default=3)
    ap.add_argument("--act", default=None, help="default: from the candidate name (silu-/relu-)")
    ap.add_argument("--symlog-readout", action="store_true")
    args = ap.parse_args()
    src = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = src["state_dict"]
    name = str(src.get("name", ""))
    act = args.act or ("silu" if name.startswith("silu") else "relu")
    in_norm = "in_norm.weight" in sd
    hidden, in_dim = sd["inp.weight"].shape
    blocks = len({k.split(".")[1] for k in sd if k.startswith("lins.")})
    obs_dim = in_dim - 5 * 52
    if src.get("extra"):
        raise SystemExit("candidates with extra (strength) inputs are not CentralCritic-shaped")
    cc = CentralCritic(
        obs_dim=obs_dim, hidden_dim=hidden, num_blocks=blocks, q_actions=args.q_actions,
        torso_layernorm=True, value_bins=int(sd["head.weight"].shape[0]),
        act=act, in_norm=in_norm, v_raw=not args.symlog_readout,
    )
    out = cc.state_dict()
    m = {"torso.0.0.weight": "inp.weight", "torso.0.0.bias": "inp.bias",
         "value_head.weight": "head.weight", "value_head.bias": "head.bias",
         "_value_centers": "centers", "_value_edges": "edges"}
    if in_norm:
        m.update({"torso.0.1.weight": "in_norm.weight", "torso.0.1.bias": "in_norm.bias"})
    for k in range(blocks):
        m.update({
            f"torso.{k + 1}.norm.weight": f"norms.{k}.weight", f"torso.{k + 1}.norm.bias": f"norms.{k}.bias",
            f"torso.{k + 1}.linear.weight": f"lins.{k}.weight", f"torso.{k + 1}.linear.bias": f"lins.{k}.bias",
        })
    for dst, s in m.items():
        assert out[dst].shape == sd[s].shape, (dst, out[dst].shape, sd[s].shape)
        out[dst] = sd[s].clone()
    cc.load_state_dict(out)
    torch.save({"critic": cc.state_dict(), "converted_from": args.src,
                "held_out": src.get("held_out"), "arch": {"act": act, "in_norm": in_norm,
                "hidden": hidden, "blocks": blocks, "v_raw": not args.symlog_readout}}, args.out)
    print(f"wrote {args.out}: {act} in_norm={in_norm} {hidden}x{blocks}, q_actions {args.q_actions}")


if __name__ == "__main__":
    main()
