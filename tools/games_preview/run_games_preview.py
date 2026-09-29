"""Local preview of the PUBLIC build (home games) against a throwaway DB.

Loopback only; dev login enabled; never touches data/public.db — the SQLite
file and trainer stats live in the system temp dir. Started by the
`games_preview` entry in .claude/launch.json (port 8772). Sign in with
  http://127.0.0.1:8772/auth/dev?email=host@example.com&name=Host
(host@example.com is the admin, so it has home-game access), then use
bot.py in this folder to seat and drive scripted guests.
"""
import os
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
# Port: argv[1] or $PREVIEW_PORT (default 8772). Each port gets its own
# throwaway DB, so several previews can run side by side.
port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PREVIEW_PORT", "8772"))
scratch = Path(tempfile.gettempdir()) / ("plodbnet_games_preview" if port == 8772 else f"plodbnet_games_preview_{port}")
scratch.mkdir(exist_ok=True)
sys.path.insert(0, str(repo / "python"))
os.chdir(repo)
os.environ.update({
    "PLO5BP_PUBLIC": "1",
    "PLO5BP_DEV_LOGIN": "1",
    "PLO5BP_BASE_URL": f"http://127.0.0.1:{port}",
    # browsers share cookies across ports: one cookie name per preview server,
    # so signing in on one port no longer signs you out on another
    "PLO5BP_SESSION_COOKIE": f"wg_session_{port}",
    "PLO5BP_ADMIN_EMAILS": "host@example.com",
    "PLO5BP_DB": str(scratch / "preview.db"),
    "PLO5BP_TRAINER_STATS": str(scratch / "stats.json"),
})
import uvicorn  # noqa: E402

uvicorn.run("plo5bp.ui.server:app", host="127.0.0.1", port=port, log_level="warning")
