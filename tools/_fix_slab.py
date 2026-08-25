from pathlib import Path
rp = Path("python/plo5bp/rollout.py")
r = rp.read_text(encoding="utf-8")
# import OBS_DIM_MINIMAL
if "OBS_DIM_MINIMAL" not in r:
    r = r.replace(
        "from plo5bp.encoding import OBS_DIM\n",
        "from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL\n",
    )
old = '        obs_dim = OBS_DIM_NLH if configs[0].variant == "nlh_single" else OBS_DIM\n'
new = (
    '        if configs[0].variant == "nlh_single":\n'
    '            obs_dim = OBS_DIM_NLH\n'
    '        elif str(getattr(train_config, "obs_mode", "full")) == "minimal":\n'
    '            obs_dim = OBS_DIM_MINIMAL\n'
    '        else:\n'
    '            obs_dim = OBS_DIM\n'
)
if old not in r:
    raise SystemExit("obs_dim assign not found in multiconfig")
r = r.replace(old, new)
rp.write_text(r, encoding="utf-8")
print("rollout multiconfig obs_dim fixed")
