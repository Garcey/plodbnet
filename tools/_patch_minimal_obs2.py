from pathlib import Path
import re

# --- env.py: project after encode in _pack_obs ---
env_path = Path("python/plo5bp/env.py")
env = env_path.read_text(encoding="utf-8")
old = "        vec = self._encode(raw, self.config)\n        self._last_obs_vec = vec\n"
new = (
    "        vec = self._encode(raw, self.config)\n"
    "        if getattr(self, \"_obs_mode\", \"full\") == \"minimal\":\n"
    "            vec = project_obs_minimal(vec)\n"
    "        self._last_obs_vec = vec\n"
)
if old not in env:
    raise SystemExit("env _pack_obs encode site not found")
env_path.write_text(env.replace(old, new), encoding="utf-8")
print("env._pack_obs projected")

# --- env_batched.py ---
eb_path = Path("python/plo5bp/env_batched.py")
eb = eb_path.read_text(encoding="utf-8")
eb = eb.replace(
    "from plo5bp.encoding import OBS_DIM, encode_observation_batch\n",
    "from plo5bp.encoding import (\n"
    "    OBS_DIM,\n"
    "    OBS_DIM_MINIMAL,\n"
    "    encode_observation_batch,\n"
    "    project_obs_minimal,\n"
    ")\n",
)
old_init = '''    def __init__(
        self,
        num_envs: int,
        config: GameConfig | None = None,
        ev_runout_samples: int = 0,
        opp_outcome_mc: int = 1024,
    ):
        self.n = int(num_envs)
        self.config = config or GameConfig()
        # Per-variant observation layout, mirroring the scalar env: the
        # PLO double-bomb family shares the 991-dim layout; NLH is 995.
        self._is_nlh = self.config.variant == VARIANT_NLH
        self._obs_dim = OBS_DIM_NLH if self._is_nlh else OBS_DIM
'''
new_init = '''    def __init__(
        self,
        num_envs: int,
        config: GameConfig | None = None,
        ev_runout_samples: int = 0,
        opp_outcome_mc: int = 1024,
        obs_mode: str = "full",
    ):
        self.n = int(num_envs)
        self.config = config or GameConfig()
        mode = str(obs_mode or "full").strip().lower()
        if mode not in ("full", "minimal"):
            raise ValueError(
                f"obs_mode must be 'full' or 'minimal', got {obs_mode!r}"
            )
        self._is_nlh = self.config.variant == VARIANT_NLH
        if mode == "minimal" and self._is_nlh:
            raise ValueError("obs_mode=minimal is PLO-only (not NLH)")
        self._obs_mode = mode
        # Per-variant observation layout: PLO full 1171 / minimal 796, NLH 995.
        if self._is_nlh:
            self._obs_dim = OBS_DIM_NLH
        elif mode == "minimal":
            self._obs_dim = OBS_DIM_MINIMAL
        else:
            self._obs_dim = OBS_DIM
'''
if old_init not in eb:
    raise SystemExit("env_batched init not found")
eb = eb.replace(old_init, new_init)

# Force numpy path when minimal (rust emits full 1171; we project after numpy encode).
# Also: minimal doesn't need expensive opp MC — caller can pass opp_outcome_mc=0.
old_rust = '''        self._use_rust_encoder = (
            bool(int(os.environ.get("PLO5_RUST_ENCODER", "0")))
            and not self._is_nlh
            and OBS_DIM == _RUST_ENCODER_OBS_DIM
        )
'''
new_rust = '''        # Minimal mode always uses numpy encode + project_obs_minimal (Rust
        # encoder emits the full 1171 layout and is skipped here).
        self._use_rust_encoder = (
            bool(int(os.environ.get("PLO5_RUST_ENCODER", "0")))
            and not self._is_nlh
            and mode == "full"
            and OBS_DIM == _RUST_ENCODER_OBS_DIM
        )
'''
if old_rust not in eb:
    raise SystemExit("rust encoder gate not found")
eb = eb.replace(old_rust, new_rust)

# After every place that assigns self._obs from encode_observation_batch, project.
# Safest: add helper and call at end of _refresh encoder paths.
# Patch the print line to include obs_mode
eb = eb.replace(
    'f"variant={self.config.variant}"\n',
    'f"variant={self.config.variant} obs_mode={self._obs_mode}"\n',
)

# Add projection helper method before _unpack_post if missing
if "_maybe_project_obs" not in eb:
    helper = '''
    def _maybe_project_obs(self) -> None:
        """If obs_mode=minimal, gather table-visible dims in-place shape (N, 796)."""
        if getattr(self, "_obs_mode", "full") != "minimal":
            return
        full = self._obs
        if full.shape[-1] == self._obs_dim:
            return  # already projected
        proj = project_obs_minimal(full)
        if (
            self._obs is not None
            and self._obs.shape == proj.shape
            and self._obs.dtype == proj.dtype
        ):
            np.copyto(self._obs, proj)
        else:
            self._obs = proj

'''
    marker = "    def _unpack_post(self, bundle) -> None:\n"
    if marker not in eb:
        raise SystemExit("_unpack_post not found")
    eb = eb.replace(marker, helper + marker)

# Call _maybe_project_obs after encoder sections that set self._obs.
# Insert before each `with record_function("step1a_unpack/post")` that follows encoder.
# Simpler: at start of _unpack_post call it? No - unpack doesn't know.
# Call after each encoder block ends - look for patterns where we set obs then unpack.

# After rust path copyto:
eb = eb.replace(
    '''            with record_function("step1a_unpack/post"):
                self._unpack_post(bundle)
            return

        with record_function("step1a_bundle/obs_features_batch"):
            bundle = self._be.observation_and_features_batch()
''',
    '''            self._maybe_project_obs()
            with record_function("step1a_unpack/post"):
                self._unpack_post(bundle)
            return

        with record_function("step1a_bundle/obs_features_batch"):
            bundle = self._be.observation_and_features_batch()
'''
)

# After numpy full encode paths - the final unpack at end of _refresh
# Find the last unpack after numpy encode
old_tail = '''                self._obs[idx] = obs_sub
        with record_function("step1a_unpack/post"):
            self._unpack_post(bundle)
'''
new_tail = '''                self._obs[idx] = obs_sub
        self._maybe_project_obs()
        with record_function("step1a_unpack/post"):
            self._unpack_post(bundle)
'''
if old_tail not in eb:
    # try alternate
    print("WARN: numpy scatter tail not exact; scanning...")
else:
    eb = eb.replace(old_tail, new_tail)

# Also after full-batch numpy encode (no mask) - the _maybe_project_obs before unpack covers scatter path only if we always call once at end.
# Looking at structure: after if/elif/else encode, one unpack. Our new_tail only on scatter branch.
# Better: single call before the shared unpack after the whole if/elif/else.

# Re-read logic - the structure is:
# if encode_mask is None or all: encode full
# elif not any: zeros
# else: scatter
# then unpack
# So project once before that unpack.

eb_path.write_text(eb, encoding="utf-8")
print("env_batched written")

# Fix: ensure project happens on full encode path too.
eb = eb_path.read_text(encoding="utf-8")
# If we only added to scatter, add before ALL unpacks that follow encoder in numpy path.
# Count _maybe_project_obs
print("project calls", eb.count("_maybe_project_obs()"))
# On numpy path the final unpack - ensure project before it
if eb.count("self._maybe_project_obs()\n        with record_function(\"step1a_unpack/post\")") < 1:
    # insert before the numpy-path final unpack only once at the right indent
    # Find pattern after encode_observation_batch blocks
    target = "        with record_function(\"step1a_unpack/post\"):\n            self._unpack_post(bundle)\n"
    # Only replace the LAST occurrence (numpy path)
    parts = eb.rsplit(target, 1)
    if len(parts) != 2:
        print("could not find final unpack")
    else:
        eb = parts[0] + "        self._maybe_project_obs()\n" + target + parts[1]
        eb_path.write_text(eb, encoding="utf-8")
        print("added final project before numpy unpack")
else:
    print("final project already present")

print("stage2 done")
