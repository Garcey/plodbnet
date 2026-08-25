"""Standalone desktop UI for the native NLH CFR solver (Monker/Pio-like).

Install once (creates a Desktop shortcut)::

    Install CFR Solver.bat
    # or:  .venv/Scripts/python scripts/install_cfr_desktop_shortcut.py

Then double-click **CFR Solver** on the Desktop — native window, no console,
no browser tab. The local API binds an ephemeral 127.0.0.1 port and dies
with the window.

Dev / browser mode::

    .venv/Scripts/python scripts/cfr_app.py --browser
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.2.0"
