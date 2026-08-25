from pathlib import Path
rp = Path("python/plo5bp/rollout.py")
r = rp.read_text(encoding="utf-8")
r = r.replace(
    "_opp_mc = 0 if _obs_mode == \"minimal\" else TRAIN_OPP_OUTCOME_MC",
    # Engine requires >=1; 1 is the cheapest legal budget (features unused in minimal).
    "_opp_mc = 1 if _obs_mode == \"minimal\" else TRAIN_OPP_OUTCOME_MC",
)
rp.write_text(r, encoding="utf-8")
print("rollout opp_mc fixed")
