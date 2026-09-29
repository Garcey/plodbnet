"""Live capture for the LOCAL study build: ClubGG pixel OCR + PokerNow DOM ingest.

Both sources turn table frames into mutations of the study ``Session`` so the
action log, participant mask and card spec follow the live table:

    ClubGG window --WGC--> clubgg.OcrRunner ----\\
                                                 +--> tracking (mirror, hand-start
    PokerNow tab --POST--> pokernow.PokerNowRunner/   machine, reconcilers) --> session
                                                      --> server._rebuild_env

* ``state``    — ``LiveState``: per-session debounce state (``session.live``).
* ``tracking`` — plumbing shared by both sources (see its docstring).
* ``clubgg``   — ``OcrRunner`` (capture loop + per-tick pipeline).
* ``pokernow`` — ``PokerNowRunner`` (per-payload pipeline).
* ``routes``   — ``/ocr/*`` + ``/pokernow/*`` and ``install(app)``.

The public build (``PLO5BP_PUBLIC``) never imports this package, so none of it
ships there; ``plo5bp.ui.server`` calls ``routes.install(app)`` otherwise.

Import order: every submodule reads the study core (``plo5bp.ui.server``: the
session proxy, the env rebuild, the state projection), and the core mounts
this package while it is itself being imported. Importing the core FIRST,
here, makes any import order work: a test that imports a live submodule on its
own runs the whole core, whose install step imports ``plo5bp.ui.live.routes``
while this ``__init__`` still sits on its first line — fine, because the core
only ever imports SUBMODULES of this package, never names from this file.
"""

import plo5bp.ui.server  # noqa: F401  (must stay the first import — see above)

from plo5bp.ui.live.routes import install, router  # noqa: E402

__all__ = ["install", "router"]
