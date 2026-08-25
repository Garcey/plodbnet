from pathlib import Path

# Fix scatter assignment to project sub-batch first
eb_path = Path("python/plo5bp/env_batched.py")
eb = eb_path.read_text(encoding="utf-8")
old = '''                else:
                    obs_sub = encode_observation_batch(
                        sub, cat_a_sub, cat_b_sub, self.config
                    )
                if self._obs.shape != (self.n, self._obs_dim):
                    self._obs = np.zeros((self.n, self._obs_dim), dtype=np.float32)
                else:
                    self._obs.fill(0.0)
                self._obs[idx] = obs_sub
'''
new = '''                else:
                    obs_sub = encode_observation_batch(
                        sub, cat_a_sub, cat_b_sub, self.config
                    )
                if getattr(self, "_obs_mode", "full") == "minimal":
                    obs_sub = project_obs_minimal(obs_sub)
                if self._obs.shape != (self.n, self._obs_dim):
                    self._obs = np.zeros((self.n, self._obs_dim), dtype=np.float32)
                else:
                    self._obs.fill(0.0)
                self._obs[idx] = obs_sub
'''
if old not in eb:
    raise SystemExit("scatter block not found")
eb_path.write_text(eb.replace(old, new), encoding="utf-8")
print("scatter fixed")

# rollout.py: pass obs_mode
rp = Path("python/plo5bp/rollout.py")
r = rp.read_text(encoding="utf-8")
# serial
r = r.replace(
    "envs = [BombPotEnv(game_config, ev_runout_samples=train_config.ev_runout_samples)\n            for _ in range(n_envs)]",
    "envs = [BombPotEnv(\n"
    "                game_config,\n"
    "                ev_runout_samples=train_config.ev_runout_samples,\n"
    "                obs_mode=str(getattr(train_config, \"obs_mode\", \"full\")),\n"
    "            )\n"
    "            for _ in range(n_envs)]",
)
# batched construct
old_b = '''            env = BatchedBombPotEnv(
                n_envs,
                game_config,
                ev_runout_samples=train_config.ev_runout_samples,
                opp_outcome_mc=TRAIN_OPP_OUTCOME_MC,
            )
'''
new_b = '''            _obs_mode = str(getattr(train_config, "obs_mode", "full"))
            # Minimal obs drops opp-outcome features; skip the expensive MC.
            _opp_mc = 0 if _obs_mode == "minimal" else TRAIN_OPP_OUTCOME_MC
            env = BatchedBombPotEnv(
                n_envs,
                game_config,
                ev_runout_samples=train_config.ev_runout_samples,
                opp_outcome_mc=_opp_mc,
                obs_mode=_obs_mode,
            )
'''
if old_b not in r:
    raise SystemExit("batched env construct not found")
r = r.replace(old_b, new_b)
# cache key must include obs_mode
r = r.replace(
    "cache_key = (n_envs, game_config.num_seats, game_config.variant)",
    "cache_key = (\n"
    "            n_envs, game_config.num_seats, game_config.variant,\n"
    "            str(getattr(train_config, \"obs_mode\", \"full\")),\n"
    "        )",
)
rp.write_text(r, encoding="utf-8")
print("rollout patched")

# config.py TrainingConfig
cp = Path("python/plo5bp/config.py")
c = cp.read_text(encoding="utf-8")
if "obs_mode" not in c:
    # add field near hidden_dim
    needle = "    hidden_dim: int = 128\n"
    if needle not in c:
        # try other default
        needle = "    hidden_dim: int = 2048\n"
    if needle not in c:
        raise SystemExit("hidden_dim field not found in config")
    c = c.replace(
        needle,
        needle
        + "    # Observation layout: 'full' = OBS_DIM 1171; 'minimal' = bare\n"
        + "    # table-visible 796 (cards/history/stacks/commits/...). Cold-start only.\n"
        + "    obs_mode: str = \"full\"\n",
    )
    cp.write_text(c, encoding="utf-8")
    print("config patched")
else:
    print("config already has obs_mode")

print("stage3 ok")
