#!/usr/bin/env python
"""Promote Step-6 rejected roots under a documented one-off 5.0 bb cap.

Does NOT change TEACHER_MAX_EXPL_BB (stays 1.0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_batch import _atomic_write_text  # noqa: E402

# NOTE (review 2026-09-20 F6): labels exported from roots accepted here carry
# ``teacher_max_expl_bb`` = whatever cap the EXPORT is run with. A checkpoint
# trained on a 5.0 bb teacher can never pass the GTO badge (bar: 1.0 bb).
ONE_OFF = 5.0


def main() -> None:
    out = Path("data/cfr/step6_teacher")
    strat = out / "strategies"
    strat.mkdir(exist_ok=True)
    accepted = []
    for p in sorted((out / "rejected").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        expl = d.get("exploitability_bb")
        rid = d["job_id"]
        if expl is None or float(expl) > ONE_OFF:
            print(f"[step6] skip {rid} expl={expl}", flush=True)
            continue
        rep = d["report"]
        dest = strat / f"{rid}.json"
        _atomic_write_text(dest, json.dumps(rep, indent=2) + "\n")
        (out / "markers" / f"{rid}.done").write_text(
            "ok-oneoff-max-expl-5.0\n", encoding="utf-8"
        )
        accepted.append(
            {
                "job_id": rid,
                "exploitability_bb": expl,
                "iters": rep.get("iterations_run"),
            }
        )
        print(
            f"[step6] ONE-OFF ACCEPT {rid} expl_bb={expl} "
            f"iters={rep.get('iterations_run')} "
            f"(run cap {ONE_OFF}; default stays 1.0)",
            flush=True,
        )
    man = {
        "out_dir": str(out),
        "note": (
            "First-train one-off max_expl_bb=5.0. "
            "TEACHER_MAX_EXPL_BB default unchanged at 1.0. "
            "48-sample MC expl plateaus ~4bb on random rivers and on "
            "quads-on-board; 80k iters still 3.89."
        ),
        "one_off_max_expl_bb": ONE_OFF,
        "default_max_expl_bb": 1.0,
        "completed": [a["job_id"] for a in accepted],
        "accepted": accepted,
        "rejected_at_1_0": True,
    }
    _atomic_write_text(out / "manifest_oneoff5.json", json.dumps(man, indent=2) + "\n")
    print(f"[step6] accepted {len(accepted)}", flush=True)


if __name__ == "__main__":
    main()
