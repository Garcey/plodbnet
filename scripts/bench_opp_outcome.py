"""Microbench: cost share of opp_outcome_fractions within a postflop
observation build. Throwaway diagnostic (scoop/quarter feature review)."""
import time
import numpy as np
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.encoding import encode_observation


def land_postflop(seed: int):
    env = BombPotEnv(GameConfig())
    obs, info = env.reset(seed=seed, button=0)
    # Bomb pot: first node is the flop. Confirm fractions are live.
    fr = np.asarray(env._rs.opp_outcome_fractions(), dtype=np.float32)
    return env, info, fr


def timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    env, info, fr = land_postflop(12345)
    print("sample fractions [k=2,3,4]x[scoopO,qO,scoopH,qH]:")
    print("  ", np.round(fr, 4).tolist())
    assert fr.any(), "expected a live postflop node"

    rs = env._rs
    N = 4000

    t_feat = timeit(lambda: rs.opp_outcome_fractions(), N)
    t_1024 = timeit(lambda: rs.opp_outcome_fractions_mc(1024), N)
    t_256 = timeit(lambda: rs.opp_outcome_fractions_mc(256), N)
    t_full = timeit(lambda: rs.observation_dict(), N)
    raw = rs.observation_dict()
    cfg = env.config
    t_enc = timeit(lambda: encode_observation(raw, cfg), N)

    # Feature share of the engine obs build (clamp jitter that can push
    # the standalone feature time slightly above the full dict time).
    share = min(t_1024 / t_full, 0.99)
    # Amdahl: obs-build speedup if only the feature part is sped up 2.85x.
    obs_speedup = 1.0 / (share * (t_256 / t_1024) + (1.0 - share))

    print(f"\nper-call (flop node, single-threaded), avg of {N}:")
    print(f"  opp_outcome_fractions (1024): {t_1024*1e6:8.1f} us")
    print(f"  opp_outcome_fractions  (256): {t_256*1e6:8.1f} us")
    print(f"  observation_dict(full, 1024): {t_full*1e6:8.1f} us")
    print(f"  encode_observation          : {t_enc*1e6:8.1f} us")
    print(f"\n  feature speedup 1024->256   = {t_1024/t_256:5.2f}x")
    print(f"  feature share of obs_dict    ~ {100*share:4.0f}%")
    print(f"  => obs-build speedup         ~ {obs_speedup:4.2f}x")


if __name__ == "__main__":
    main()
