"""Bit-exactness check for the training pipeline (owner rule: "remains bit exact").

Train a tiny recipe with two versions of the Python code -- a git revision
(default HEAD) and the working tree -- and compare the SHA-256 of EVERY tensor
of every checkpoint they write, the rolling ``<stem>.optim.pt`` sidecar (Adam
moments + l2-init references) included. Identical digests = the change is exact
for that recipe; any difference names the first tensors that moved.

CLI: ``scripts/exactness_check.py`` (see its --help). Test:
``tests/python/training/test_exactness.py`` runs the ``smoke`` recipe on every pytest
session (HEAD vs the working tree when training code has uncommitted changes,
else the working tree twice = determinism).

Scope and rules (docs/training.md "Second efficiency pass"):

- Only TENSORS (and numpy arrays) are compared. Metadata that may legitimately
  differ between two runs (wall-clock, provenance, log text) is ignored.
- Both sides import the Rust engine binary of THIS working tree (the reference
  tree gets a copy): the check covers Python changes. To check an ENGINE
  change, build each side's extension and pass ``engine=`` / ``--ref-engine``.
- Each side runs in a fresh working directory with its own live-control file
  (``PLO5BP_ANNEAL_CONTROL``), every inherited ``PLO5*`` variable cleared and
  the recipe's environment applied, so nothing on this machine leaks in.
- CUDA: both sides share ONE private ``TORCHINDUCTOR_CACHE_DIR`` and run one
  after the other -- Inductor autotunes the compiled PPO kernels by timing and
  caches the choice, so a cache written by other runs (or two runs autotuning
  at once) changes the numerics. On CPU nothing is compiled and the two sides
  may run in parallel.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths a reference tree needs: the package and the scripts (train.py; later
# refactors may add more entry points there).
EXPORT_PATHS = ("python", "scripts")

# Training-relevant sources: an uncommitted change under one of these can move
# a digest. (The UI / GTO / OCR / CFR-app subpackages cannot.)
TRAINING_PATHSPECS = (
    ":(glob)python/plo5bp/*.py",
    ":(glob)python/plo5bp/train/**",
    "scripts/train.py",
)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """A tiny train.py invocation. ``args`` must NOT contain --checkpoint,
    --num-updates or --device (the runner owns them)."""

    description: str
    args: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    updates: int = 3


_COMMON = ("--batched", "--seed", "1234", "--checkpoint-every", "1")

RECIPES: dict[str, Recipe] = {
    # The docs/training.md recipe ("Verify an exactness claim"): legacy v2 anchor head,
    # full observation through the default encoder (Rust since 2026-09-28,
    # ML-015; numpy before -- the two are bit-identical), 6 mixed configs per
    # update.
    "tiny": Recipe(
        "docs/training.md recipe: v2 anchor head, full obs (the default encoder), 32x3 / 32x1",
        (
            "--hidden-dim", "32", "--num-layers", "3",
            "--critic-hidden-dim", "32", "--critic-num-blocks", "1",
            "--num-envs", "480", "--rollout-length", "24000",
            "--mix-configs", "--configs-per-tier", "2", *_COMMON,
        ),
    ),
    # The vSix6 guardian's flag set at toy scale: v6 kit, rebuilt SiLU critic,
    # q-norm, extra critic-only epoch, micro-batching, f16 real columns, obs rev
    # 1 through the Rust encoder, pool opponents from update 1 on.
    "v6": Recipe(
        "vSix6 flags at toy scale (v6 kit, SiLU critic, q-norm, critic-only "
        "epoch, micro-batching, obs-real-f16, Rust encoder, obs rev 1)",
        (
            "--variant", "plo5_double_bomb", "--v6",
            "--hidden-dim", "32", "--num-layers", "3",
            "--critic-hidden-dim", "32", "--critic-num-blocks", "2",
            "--critic-act", "silu", "--critic-in-norm", "--critic-v-raw",
            "--q-base-raw", "--q-fold-zero", "--critic-q-norm",
            "--critic-extra-epochs", "1", "--critic-minibatches", "8",
            "--no-grad-checkpoint",
            "--num-envs", "480", "--rollout-length", "12000",
            "--obs-real-f16", "--micro-batch-rows", "1000",
            "--num-minibatches", "4", "--ppo-epochs", "2",
            "--mix-configs", "--configs-per-tier", "2",
            "--mix-tiers", "clubgg,clubgg_deep,deep",
            "--entropy-coef", "0.045", "--sizing-entropy-scale", "0.3",
            "--gae-lambda", "0.8", "--lr", "7.5e-5", "--clip-room-mid", "0.07",
            "--target-kl", "0.5", "--kl-hard", "10.0", "--adv-clip", "8",
            "--snapshot-every", "1", *_COMMON,
        ),
        (("PLO5_RUST_ENCODER", "1"), ("PLO5BP_OBS_REV", "1")),
    ),
    # The minimal-observation stems (vMin3's settings at toy scale).
    "minimal": Recipe(
        "vMin3 flags at toy scale (minimal obs, Rust minimal encoder)",
        (
            "--variant", "plo5_double_bomb", "--v6", "--obs-mode", "minimal",
            "--hidden-dim", "32", "--num-layers", "3",
            "--critic-hidden-dim", "32", "--critic-num-blocks", "2",
            "--num-envs", "480", "--rollout-length", "24000",
            "--micro-batch-rows", "2000", "--num-minibatches", "4",
            "--ppo-epochs", "2", "--mix-configs", "--configs-per-tier", "2",
            "--entropy-coef", "0.07", "--sizing-entropy-scale", "0.1",
            "--snapshot-every", "1", *_COMMON,
        ),
        (("PLO5_RUST_ENCODER", "1"),),
    ),
    # The pytest recipe: the v6 recipe's code paths at the smallest size that
    # still plays pool opponents (update 2) -- a few seconds per side on CPU.
    "smoke": Recipe(
        "pytest: v6 recipe's code paths at the smallest size",
        (
            "--variant", "plo5_double_bomb", "--v6",
            "--hidden-dim", "16", "--num-layers", "3",
            "--critic-hidden-dim", "16", "--critic-num-blocks", "1",
            "--critic-act", "silu", "--critic-in-norm", "--critic-v-raw",
            "--q-base-raw", "--q-fold-zero", "--critic-q-norm",
            "--critic-extra-epochs", "1", "--critic-minibatches", "4",
            "--no-grad-checkpoint",
            "--num-envs", "120", "--rollout-length", "2400",
            "--obs-real-f16", "--micro-batch-rows", "300",
            "--num-minibatches", "2", "--ppo-epochs", "2",
            "--mix-configs", "--configs-per-tier", "1",
            "--entropy-coef", "0.045", "--sizing-entropy-scale", "0.3",
            "--gae-lambda", "0.8", "--snapshot-every", "1", *_COMMON,
        ),
        (("PLO5_RUST_ENCODER", "1"), ("PLO5BP_OBS_REV", "1")),
    ),
}


# --------------------------------------------------------------- digests ---

def _array_digest(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(repr(tuple(arr.shape)).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(repr(tuple(t.shape)).encode())
    # Through a uint8 view: numpy has no bfloat16, and a byte view is exact
    # for every dtype.
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def tensor_digests(obj: object, prefix: str = "") -> dict[str, str]:
    """``{path: sha256}`` for every tensor / numpy array inside ``obj``
    (nested dicts, lists and tuples; dict keys become path parts). Anything
    else -- numbers, strings, None -- is metadata and ignored."""
    out: dict[str, str] = {}
    if torch.is_tensor(obj):
        out[prefix or "."] = _tensor_digest(obj)
    elif isinstance(obj, np.ndarray):
        out[prefix or "."] = _array_digest(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(tensor_digests(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(tensor_digests(v, f"{prefix}[{i}]"))
    return out


def checkpoint_digests(ckpt_dir: Path) -> dict[str, dict[str, str]]:
    """``{file name: tensor_digests(torch.load(file))}`` for every ``*.pt`` in
    ``ckpt_dir`` (numbered checkpoints, the final save, the optimizer sidecar)."""
    out: dict[str, dict[str, str]] = {}
    for f in sorted(Path(ckpt_dir).glob("*.pt")):
        out[f.name] = tensor_digests(torch.load(f, map_location="cpu", weights_only=False))
    return out


def _load_tensor(ckpt: Path, path: str):
    """The object at a ``tensor_digests`` path (for the max-|diff| report)."""
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    import re

    for part in re.findall(r"[^/\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            obj = obj[int(part[1:-1])]
        elif isinstance(obj, dict):
            key = next((k for k in obj if str(k) == part), part)
            obj = obj[key]
        else:
            return None
    return obj


def compare_digests(
    ref: dict[str, dict[str, str]],
    new: dict[str, dict[str, str]],
    ref_dir: Path | None = None,
    new_dir: Path | None = None,
    limit: int = 20,
) -> list[str]:
    """Human-readable differences (empty = bit-identical). With the two
    checkpoint directories, a differing float tensor also reports its max |diff|."""
    diffs: list[str] = []
    for name in sorted(set(ref) | set(new)):
        if name not in new:
            diffs.append(f"{name}: written by the reference only")
            continue
        if name not in ref:
            diffs.append(f"{name}: written by the new code only")
            continue
        a, b = ref[name], new[name]
        for key in sorted(set(a) | set(b)):
            if key not in b:
                diffs.append(f"{name}:{key}: missing in the new code")
            elif key not in a:
                diffs.append(f"{name}:{key}: new tensor (not in the reference)")
            elif a[key] != b[key]:
                msg = f"{name}:{key}: differs"
                if ref_dir is not None and new_dir is not None:
                    try:
                        ta = _load_tensor(Path(ref_dir) / name, key)
                        tb = _load_tensor(Path(new_dir) / name, key)
                        if torch.is_tensor(ta) and torch.is_tensor(tb):
                            if ta.shape != tb.shape or ta.dtype != tb.dtype:
                                msg += f" (shape/dtype {tuple(ta.shape)}/{ta.dtype} -> {tuple(tb.shape)}/{tb.dtype})"
                            elif ta.is_floating_point():
                                d = (ta.double() - tb.double()).abs().max().item()
                                msg += f" (max |diff| {d:.3e})"
                    except Exception:  # noqa: BLE001 -- the report is best-effort
                        pass
                diffs.append(msg)
            if len(diffs) >= limit:
                diffs.append("... (more differences not listed)")
                return diffs
    return diffs


# ------------------------------------------------------------ the trees ---

def engine_binaries(pkg_dir: Path) -> list[Path]:
    """The compiled extension(s) of a ``plo5bp`` package directory."""
    # Exactly the importable module (`_engine.pyd`, `_engine.cp314-win_amd64.pyd`,
    # `_engine.cpython-311-x86_64-linux-gnu.so`), not a swapped-out
    # `_engine_old_<n>.pyd` left behind by rebuild_engine.sh.
    return sorted(
        p for p in Path(pkg_dir).glob("_engine*")
        if p.name.split(".")[0] == "_engine" and p.suffix in (".pyd", ".so")
    )


def training_changes(repo: Path = REPO_ROOT, rev: str = "HEAD") -> list[str]:
    """Training-relevant paths that differ between ``rev`` and the working
    tree (modified, added, deleted or untracked). Raises when git is missing."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *TRAINING_PATHSPECS],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    changed = [ln[3:].strip() for ln in out.splitlines() if ln.strip()]
    if rev != "HEAD":
        diff = subprocess.run(
            ["git", "diff", "--name-only", rev, "--", *TRAINING_PATHSPECS],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout
        changed = sorted(set(changed) | {ln.strip() for ln in diff.splitlines() if ln.strip()})
    return changed


def export_git_tree(rev: str, dest: Path, repo: Path = REPO_ROOT,
                    engine: Path | None = None) -> Path:
    """Write ``rev``'s ``python/`` and ``scripts/`` into ``dest`` (read-only
    git: ``git archive``; the repository is not touched) and give it an engine
    binary: ``engine`` if given, else a copy of the working tree's. Returns
    ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tar_bytes = subprocess.run(
        ["git", "archive", "--format=tar", rev, "--", *EXPORT_PATHS],
        cwd=repo, capture_output=True, check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # a Python without extraction filters (< 3.12)
            tf.extractall(dest)
    pkg = dest / "python" / "plo5bp"
    sources = [Path(engine)] if engine is not None else engine_binaries(repo / "python" / "plo5bp")
    if not sources:
        raise FileNotFoundError(
            "no compiled engine (python/plo5bp/_engine*.pyd|.so) in the working "
            "tree -- build it first (maturin develop --release from the repo root)"
        )
    for src in sources:
        shutil.copy2(src, pkg / src.name)
    return dest


# ---------------------------------------------------------------- running ---

def _clean_env(recipe: Recipe, workdir: Path, tree: Path,
               inductor_cache: Path, extra_env: dict[str, str] | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PLO5")}
    env.update({
        "PYTHONPATH": str(Path(tree) / "python"),
        "PLO5BP_ANNEAL_CONTROL": str(workdir / "no_live_control.json"),
        "PLO5BP_STEP_TIMERS": "0",
        "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
        # PYTHONHASHSEED is deliberately NOT pinned: production runs do not
        # pin it, so a result that depends on str-hash order must show up as
        # a difference (--same catches it).
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.update(dict(recipe.env))
    env.update(extra_env or {})
    return env


def run_recipe(
    tree: Path,
    recipe: Recipe,
    workdir: Path,
    *,
    device: str = "cpu",
    updates: int | None = None,
    inductor_cache: Path | None = None,
    extra_args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
    timeout: float = 3600.0,
    side: str = "new",
) -> Path:
    """Train ``recipe`` with ``tree``'s code in ``workdir``; returns the
    checkpoint directory. Raises ``RunFailed`` (with the log tail) when the
    run fails."""
    workdir = Path(workdir)
    ck_dir = workdir / "ck"
    ck_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(inductor_cache) if inductor_cache is not None else workdir / "inductor"
    cmd = [
        sys.executable, "-u", str(Path(tree) / "scripts" / "train.py"),
        *recipe.args, *extra_args,
        "--device", device,
        "--num-updates", str(recipe.updates if updates is None else updates),
        "--checkpoint", str(ck_dir / "x.pt"),
    ]
    env = _clean_env(recipe, workdir, Path(tree), cache, extra_env)
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                          text=True, timeout=timeout)
    (workdir / "train.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-4000:]
        raise RunFailed(side, f"train.py exited {proc.returncode} in {workdir}:\n{tail}")
    return ck_dir


class RunFailed(RuntimeError):
    """A recipe run exited non-zero. ``side`` is "ref" or "new"."""

    def __init__(self, side: str, message: str) -> None:
        super().__init__(message)
        self.side = side


@dataclasses.dataclass
class CheckResult:
    recipe: str
    identical: bool
    differences: list[str]
    ref_dir: Path
    new_dir: Path


def check(
    recipe_name: str,
    ref_tree: Path,
    new_tree: Path,
    workdir: Path,
    *,
    device: str = "cpu",
    updates: int | None = None,
    parallel: bool | None = None,
    extra_args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
) -> CheckResult:
    """Run ``recipe_name`` with both trees and compare every tensor digest."""
    recipe = RECIPES[recipe_name]
    workdir = Path(workdir)
    cache = workdir / "inductor"  # ONE private cache for both sides
    kw = dict(device=device, updates=updates, inductor_cache=cache,
              extra_args=extra_args, extra_env=extra_env)
    ref_w, new_w = workdir / "ref", workdir / "new"
    if parallel is None:
        parallel = device == "cpu"
    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(run_recipe, ref_tree, recipe, ref_w, side="ref", **kw)
            fb = pool.submit(run_recipe, new_tree, recipe, new_w, side="new", **kw)
            # The NEW side's failure is the one that matters: report it first.
            new_ck = fb.result()
            ref_ck = fa.result()
    else:
        new_ck = run_recipe(new_tree, recipe, new_w, side="new", **kw)
        ref_ck = run_recipe(ref_tree, recipe, ref_w, side="ref", **kw)
    ref_d, new_d = checkpoint_digests(ref_ck), checkpoint_digests(new_ck)
    if not ref_d:
        raise RunFailed("ref", f"the reference run wrote no checkpoints ({ref_ck})")
    diffs = compare_digests(ref_d, new_d, ref_ck, new_ck)
    return CheckResult(recipe_name, not diffs, diffs, ref_ck, new_ck)


def scratch_dir(prefix: str = "plo5bp_exact_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
