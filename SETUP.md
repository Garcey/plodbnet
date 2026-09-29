# Setting up a machine

Three kinds of machine run this project. Each section is complete on its own.

- [A Windows PC](#a-windows-pc-development-study-tool-cfr-solver) — development, the local study tool, the CFR Solver app
- [The training pod](#the-training-pod-runpod) — RunPod GPU box that trains the models
- The production server — [ops/SERVER_SETUP.md](ops/SERVER_SETUP.md) (and [docs/ops/PRODUCTION.md](docs/ops/PRODUCTION.md) to run it)

**Versions.** Rust is the same everywhere: `rust-toolchain.toml` pins the compiler
(1.93.1) for every build in the repo, and the builds use the exact crate versions in
`Cargo.lock` (`--locked` on the server and in CI). Python differs per machine and the
engine runs on any 3.11+ (it is built for the stable ABI): the desktop 3.14
(`requirements/desktop-freeze.txt`), the pod the image's own Python (recorded in
`requirements/pod-freeze.txt`), the production server and CI 3.11 (Debian 12's).

Every machine gets its **own** SSH key. Never copy a private key from one machine to
another: a lost machine is then one line to revoke.

## A Windows PC (development, study tool, CFR Solver)

Install first: **Git for Windows** (it brings Git Bash — run the `bash` commands of
this repo there), **Python 3.11 or newer** (the desktop runs 3.14), **Rust** (rustup —
the compiler version is pinned by `rust-toolchain.toml` and rustup fetches it on the
first build), **Node.js 20+** (the browser-code tests run under Node). Optional:
Tesseract (`winget install --id UB-Mannheim.TesseractOCR`, live ClubGG capture only).

```powershell
git clone https://github.com/Garcey/plodbnet      # the repo is private: sign in when asked
cd plodbnet
py -3 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install -e ".[dev,desktop]"     # + ",ocr" for live table capture
.venv\Scripts\maturin develop --release           # builds the Rust engine — from the repo ROOT
```

- The exact versions the desktop runs are in `requirements/desktop-freeze.txt`; add
  `-c requirements/desktop-freeze.txt` to the `pip install` line to get the same.
- An NVIDIA GPU (the desktop has an RTX 3070): swap in the CUDA build of torch with
  `.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- Re-run `maturin develop --release` after pulling Rust changes. From
  `rust_engine\` it fails with "Couldn't find the symbol PyInit_plo5bp_engine". While
  any Python process has the engine loaded, Windows cannot replace it — close the UI /
  tests first. A test run that ends with "STALE ENGINE" means exactly this rebuild.
- Once per clone, turn on the commit hook that refuses secrets:
  `git config core.hooksPath scripts/hooks` (it uses gitleaks when installed:
  `winget install gitleaks.gitleaks`).

**Run things**

- Study tool (local build, live capture included): `.venv\Scripts\python -m uvicorn plo5bp.ui.server:app --port 8765`
- The public website locally: `.\run_public.ps1` — see [PUBLIC_SETUP.md](PUBLIC_SETUP.md)
- The CFR Solver app: double-click `Install CFR Solver.bat` (Desktop + Start-menu
  shortcut; without pywin32 it uses PowerShell), or
  `.venv\Scripts\python scripts\cfr_app.py --desktop` (own window) / `--browser`
  (http://127.0.0.1:8766). Solves go to `data\cfr\` (`CFR_APP_DATA_DIR` to move them);
  if the window does not appear, read `runs\cfr_app.log`. It answers on 127.0.0.1 only.

**Tests** — `bash scripts/check.sh` runs what CI runs; `bash scripts/check.sh site`,
`… training` or `… quick` run a subset (or `pytest -m homegame`, `-m training`,
`-m "not slow"`). What legitimately SKIPS on a PC like the desktop (the end-of-run
summary lists every skip): the pixel-OCR modules without OpenCV / Tesseract, the CFR
app tests whose fixtures live in the git-ignored `data/cfr/`, and anything needing a
trained checkpoint (`checkpoints/` is git-ignored — the UI then serves an untrained
placeholder). Anything else that skips or fails is real.

`cargo test` needs a Python for pyo3's build step: in Git Bash,
`export PYO3_PYTHON="$PWD/.venv/Scripts/python.exe"`, put the BASE Python folder (the
one with `python3.dll`) on PATH, then
`cargo test --manifest-path rust_engine/Cargo.toml --profile fasttest --lib`.

**Line endings**: the working tree is CRLF (Git's `core.autocrlf`) except shell
scripts, which must stay LF (they run on Linux). Some tools silently flip a file's
endings — check with a byte count, not grep. What ships to the server is exported
from git, so it is always as committed.

## The training pod (RunPod)

What has been used: an RTX PRO 6000 Blackwell (96 GB), ~233 GiB container RAM, and a
persistent network volume at `/workspace` — **a ~20 GB quota** (measure with
`du -sb /workspace`; `df` shows the whole cluster). A full volume kills every trainer
SILENTLY at its next checkpoint write: keep it under ~15 GB and move old checkpoints
to the desktop. A new pod starts with an empty home folder; only `/workspace` persists.

**Access**: add each machine's own public key under RunPod → Settings → SSH Public
Keys; connect with the "SSH over exposed TCP" line from the pod's Connect panel (the
address and port change with every pod — never write them into the repo).

**One-time: the code** (the repo is private, so the pod clones with a read-only
deploy key kept on the persistent volume):

```bash
mkdir -p /workspace/secrets && chmod 700 /workspace/secrets
ssh-keygen -t ed25519 -N '' -C plodbnet-pod-deploy -f /workspace/secrets/plodbnet_deploy
cat /workspace/secrets/plodbnet_deploy.pub
#   → GitHub: Garcey/plodbnet → Settings → Deploy keys → Add deploy key (leave "write" OFF)
GIT_SSH_COMMAND='ssh -i /workspace/secrets/plodbnet_deploy -o IdentitiesOnly=yes' \
  git clone git@github.com:Garcey/plodbnet.git /workspace/plodbnet
git -C /workspace/plodbnet config core.sshCommand "ssh -i /workspace/secrets/plodbnet_deploy -o IdentitiesOnly=yes"
```

**Every new pod** (~2 minutes; the guardians expect exactly this layout):

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && . ~/.cargo/env
cd /workspace/plodbnet && git pull
rustup toolchain install 1.93.1 --profile minimal  # = rust-toolchain.toml (the build would fetch it anyway)
python3 -m venv --system-site-packages .venv       # reuses the image's CUDA torch
.venv/bin/pip install -r requirements/pod.in
CARGO_TARGET_DIR=/root/cargo-target .venv/bin/maturin develop --release
.venv/bin/pip freeze --exclude-editable > requirements/pod-freeze.txt   # copy it back and commit it
```

**Optional, a few % faster rollouts**: the engine is built for any x86-64 CPU by
default. A pod that builds its own engine can build it for its own CPU instead —
put `RUSTFLAGS="-C target-cpu=native"` in front of the `maturin develop` / `maturin
build` commands here (every time, or cargo rebuilds everything). Measured on the
desktop (PERF-025): full-observation encoding ~3-4% faster (the board-draw features
~25%), outputs bit-identical (golden digests). Such a binary may crash on an older
CPU: never copy it to another machine, and never set this for the production server
or CI.

**Training**: each run has a guardian that relaunches the trainer and resumes from
the newest `<stem>_*.pt` (e.g. `NUM_ENVS=1760000 setsid nohup bash scripts/vSix6_guardian.sh`);
stop it with its stop file (`runs/<stem>.stop`), tune it live through its control file
(CLAUDE.md, "Training"). Always name the network size (`--hidden-dim`, `--num-layers`).
The guardians pin the trainer's CPUs to the GPU's NUMA node.

**Updating code under a running trainer**: `git pull`, then build the engine as a
wheel and swap the binary in place (the running process keeps the old file):
`CARGO_TARGET_DIR=/root/cargo-target .venv/bin/maturin build --release -o /root/wheels`,
extract `_engine*.so` from the newest wheel into `python/plo5bp/` under a temporary
name and `mv -f` it over the old one. A graceful restart = the stop file + SIGTERM:
the trainer finishes its update and saves, and the guardian relaunches it warm (Adam
state and opponent pool restored). Careful with `pgrep -f` / `pkill -f`: a pattern
can also match the ssh command that runs it.

**Checking exactness on the GPU**: the checkpoint-digest recipe in CLAUDE.md
("Second efficiency pass"), with ONE private `TORCHINDUCTOR_CACHE_DIR` for both runs.
