from pathlib import Path

# --- encoding.py ---
enc_path = Path("python/plo5bp/encoding.py")
enc = enc_path.read_text(encoding="utf-8")
if "OBS_DIM_MINIMAL" in enc:
    print("encoding.py already has minimal")
else:
    needle = "return np.ascontiguousarray(vec[..., :OBS_DIM_V2])\n"
    idx = enc.find(needle)
    if idx < 0:
        raise SystemExit("needle not found in encoding.py")
    pos = idx + len(needle)
    while pos < len(enc) and enc[pos] in "\r\n":
        pos += 1
    insert = '''
# ---- bare-visibility / "minimal" obs mode (experiment stem) ----------------
# Table-visible state only: cards, street, who's in / all-in, stacks, pot
# pricing scalars, commits, seat structure, button, action history.
# NO SPR/odds/categories/draws/blockers/opp-outcome MC/v2-v7 engineered tails.
# Used by `--obs-mode minimal` (cold-start only). Gather is exact.
_MINIMAL_RANGES: tuple[tuple[int, int], ...] = (
    (_HOLE_OFF, _BOARD_B_OFF + 52),          # hole + board A + board B (156)
    (_STREET_OFF, _STREET_OFF + 4),          # street one-hot (4)
    (_ACTIVE_OFF, _ACTIVE_OFF + 8),          # active mask (8)
    (_ALLIN_OFF, _ALLIN_OFF + 8),            # all-in mask (8)
    (_STACKS_OFF, _STACKS_OFF + 8),          # stacks/bb (8)
    (_SCALARS_OFF, _SCALARS_OFF + 4),        # pot, to_call, min_bet, max_bet (4)
    (_HISTORY_OFF, _HISTORY_OFF + _HISTORY_DEPTH * _HISTORY_SLOT_DIM),  # 576
    (_SEAT_EXISTS_OFF, _SEAT_EXISTS_OFF + 8),       # seat-exists (8)
    (_TOTAL_COMMIT_OFF, _TOTAL_COMMIT_OFF + 8),     # hand total commit (8)
    (_STREET_COMMIT_OFF, _STREET_COMMIT_OFF + 8),   # street commit (8)
    (_HERO_BTN_DIST_OFF, _HERO_BTN_DIST_OFF + 8),   # button vs hero (8)
)
_MINIMAL_INDEX: np.ndarray = np.concatenate(
    [np.arange(a, b, dtype=np.int64) for a, b in _MINIMAL_RANGES]
)
OBS_DIM_MINIMAL: int = int(_MINIMAL_INDEX.shape[0])
assert OBS_DIM_MINIMAL == 796, f"unexpected minimal width {_MINIMAL_INDEX.shape[0]}"


def project_obs_minimal(vec: np.ndarray) -> np.ndarray:
    """Project full-layout obs `(..., OBS_DIM)` -> bare-visibility `(..., 796)`.

    Exact gather of table-visible dims only. Safe on vectors and batches.
    """
    return np.ascontiguousarray(vec[..., _MINIMAL_INDEX])


'''
    enc_path.write_text(enc[:pos] + insert + enc[pos:], encoding="utf-8")
    print("encoding.py patched, OBS_DIM_MINIMAL ok")

# --- env.py ---
env_path = Path("python/plo5bp/env.py")
env = env_path.read_text(encoding="utf-8")
if "obs_mode" in env and "OBS_DIM_MINIMAL" in env:
    print("env.py already has obs_mode")
else:
    env = env.replace(
        "from plo5bp.encoding import OBS_DIM, encode_observation\n",
        "from plo5bp.encoding import (\n"
        "    OBS_DIM,\n"
        "    OBS_DIM_MINIMAL,\n"
        "    encode_observation,\n"
        "    project_obs_minimal,\n"
        ")\n",
    )
    old_init = '''    def __init__(
        self,
        config: GameConfig | None = None,
        ev_runout_samples: int = 0,
    ):
        self.config = config or GameConfig()
        stacks = np.asarray(self.config.resolved_stacks, dtype=np.uint64)
        self._rs = _RustGameState(
            num_seats=self.config.num_seats,
            starting_stack=0,
            ante=self.config.ante,
            bb=self.config.bb,
            starting_stacks=stacks,
            variant=self.config.variant,
            sb=self.config.sb,
        )
        # Per-variant observation layout: PLO5 991 dims, NLH 995.
        if self.config.variant == VARIANT_NLH:
            self._obs_dim = OBS_DIM_NLH
            self._encode = encode_observation_nlh
        else:
            self._obs_dim = OBS_DIM
            self._encode = encode_observation
        self._last_obs_vec = np.zeros(self._obs_dim, dtype=np.float32)
'''
    new_init = '''    def __init__(
        self,
        config: GameConfig | None = None,
        ev_runout_samples: int = 0,
        obs_mode: str = "full",
    ):
        self.config = config or GameConfig()
        mode = str(obs_mode or "full").strip().lower()
        if mode not in ("full", "minimal"):
            raise ValueError(
                f"obs_mode must be 'full' or 'minimal', got {obs_mode!r}"
            )
        if mode == "minimal" and self.config.variant == VARIANT_NLH:
            raise ValueError("obs_mode=minimal is PLO-only (not NLH)")
        self._obs_mode = mode
        stacks = np.asarray(self.config.resolved_stacks, dtype=np.uint64)
        self._rs = _RustGameState(
            num_seats=self.config.num_seats,
            starting_stack=0,
            ante=self.config.ante,
            bb=self.config.bb,
            starting_stacks=stacks,
            variant=self.config.variant,
            sb=self.config.sb,
        )
        # Per-variant observation layout: PLO full 1171 / minimal 796, NLH 995.
        if self.config.variant == VARIANT_NLH:
            self._obs_dim = OBS_DIM_NLH
            self._encode = encode_observation_nlh
        elif mode == "minimal":
            self._obs_dim = OBS_DIM_MINIMAL
            self._encode = encode_observation
        else:
            self._obs_dim = OBS_DIM
            self._encode = encode_observation
        self._last_obs_vec = np.zeros(self._obs_dim, dtype=np.float32)
'''
    if old_init not in env:
        raise SystemExit("env.py init block not found")
    env = env.replace(old_init, new_init)
    # project after encode in _pack_obs — find encode call site
    # look for: obs_vec = self._encode(
    # and after assignment project if minimal
    env_path.write_text(env, encoding="utf-8")
    print("env.py init patched")

print("done stage1")
