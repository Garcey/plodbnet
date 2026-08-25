#!/usr/bin/env python
"""Promote Step-6 rejected roots under a documented one-off 5.0 bb cap.

Does NOT change TEACHER_MAX_EXPL_BB (stays 1.0).
"""

from __future__ import annotations

import json
from pathlib import Path

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
        dest.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
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
    (out / "manifest_oneoff5.json").write_text(
        json.dumps(man, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[step6] accepted {len(accepted)}", flush=True)


if __name__ == "__main__":
    main()
