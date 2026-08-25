from pathlib import Path

tp = Path("scripts/train.py")
t = tp.read_text(encoding="utf-8")

# import OBS_DIM_MINIMAL
if "OBS_DIM_MINIMAL" not in t:
    t = t.replace(
        "from plo5bp.encoding import OBS_DIM\n",
        "from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL\n",
    )

# add CLI flag after --num-layers
if "--obs-mode" not in t:
    t = t.replace(
        '    parser.add_argument("--num-layers", type=int, default=4)\n',
        '    parser.add_argument("--num-layers", type=int, default=4)\n'
        '    parser.add_argument(\n'
        '        "--obs-mode",\n'
        '        choices=["full", "minimal"],\n'
        '        default="full",\n'
        '        help="Observation layout. full=OBS_DIM 1171 (default). '\
        'minimal=bare table-visible 796 (cards, street, active/all-in, '\
        'stacks, pot/to_call/min/max, commits, seat-exists, button, history). '\
        'Cold-start only; no warm-start from full-obs checkpoints. '\
        'Skips opp-outcome MC for speed.",\n'
        '    )\n',
    )

# TrainingConfig field
if "obs_mode=args.obs_mode" not in t:
    t = t.replace(
        "        hidden_dim=args.hidden_dim,\n"
        "        num_layers=args.num_layers,\n",
        "        hidden_dim=args.hidden_dim,\n"
        "        num_layers=args.num_layers,\n"
        "        obs_mode=args.obs_mode,\n",
    )

# obs_dim selection
old_obs = "    obs_dim = OBS_DIM_NLH if is_nlh else OBS_DIM\n"
new_obs = (
    "    if is_nlh and args.obs_mode == \"minimal\":\n"
    "        raise SystemExit(\"error: --obs-mode minimal is PLO-only\")\n"
    "    if is_nlh:\n"
    "        obs_dim = OBS_DIM_NLH\n"
    "    elif args.obs_mode == \"minimal\":\n"
    "        obs_dim = OBS_DIM_MINIMAL\n"
    "    else:\n"
    "        obs_dim = OBS_DIM\n"
)
if old_obs not in t:
    raise SystemExit("obs_dim line not found")
t = t.replace(old_obs, new_obs)

# print obs_mode in head line
t = t.replace(
    'f"obs_dim={obs_dim} anchors={anchor_spec.count} ({anchor_spec.name})"',
    'f"obs_dim={obs_dim} obs_mode={args.obs_mode} '\
    'anchors={anchor_spec.count} ({anchor_spec.name})"',
)

# warm-start: refuse obs_mode mismatch — after variant check
guard = '''        ckpt_obs_mode = str(
            (ckpt.get("config") or {}).get("obs_mode", "full")
            if isinstance(ckpt.get("config"), dict)
            else ckpt.get("obs_mode", "full")
        )
        if ckpt_obs_mode != args.obs_mode:
            raise SystemExit(
                f"obs_mode mismatch: checkpoint={ckpt_obs_mode!r} vs "
                f"--obs-mode={args.obs_mode!r}. Minimal/full layouts are not "
                "warm-start compatible (different obs width + features)."
            )
'''
# insert after variant mismatch block ends - find unique spot
marker = '            raise SystemExit(\n                f"variant mismatch: checkpoint={ckpt_variant} vs "\n'
# find the end of that SystemExit
idx = t.find('Cross-variant warm-starts are "\n                "refused: each variant trains from scratch."\n            )\n')
if idx < 0:
    # try other formatting
    idx = t.find("each variant trains from scratch.")
    if idx < 0:
        raise SystemExit("variant exit not found")
    # find closing paren after
    end = t.find(")", idx)
    end = t.find("\n", end) + 1
else:
    end = idx + len('Cross-variant warm-starts are "\n                "refused: each variant trains from scratch."\n            )\n')
if "obs_mode mismatch" not in t:
    t = t[:end] + guard + t[end:]
    print("warm-start guard inserted")
else:
    print("guard already present")

# stamp obs_mode into checkpoint saves — look for torch.save dicts with "variant"
# Add obs_mode next to variant in save payloads
import re
# simpler: ensure config dict already has obs_mode via TrainingConfig dataclass dump
# Check how config is saved
if '"obs_mode"' not in t and "obs_mode" in t:
    # TrainingConfig is saved as whole - if it's a dataclass asdict it should include new field
    pass

tp.write_text(t, encoding="utf-8")
print("train.py patched")

# Verify TrainingConfig has field
from plo5bp.config import TrainingConfig
import dataclasses
fields = {f.name for f in dataclasses.fields(TrainingConfig)}
assert "obs_mode" in fields, fields
print("TrainingConfig.obs_mode present, default", TrainingConfig().obs_mode)
