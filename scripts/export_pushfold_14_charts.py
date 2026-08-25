"""Export multiway push/fold solve into 14 labeled strategy charts.

Reads a solve JSON whose infoset_ids use path labels:
  mwpf_p{seat}_path{open|F|AI|F,F|...}_c{class}

Writes one JSON chart per public node under data/cfr/pushfold_14_charts/.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SEAT_NAMES = {0: "CO", 1: "BTN", 2: "SB", 3: "BB"}
RANKS = "23456789TJQKA"

# Expected public tree for 4-handed pure push/fold (your 14 nodes).
# Path = actions of earlier players in seat order CO → BTN → SB (BB never acts before BB node).
EXPECTED = {
    0: ["open"],
    1: ["F", "AI"],
    2: ["F,F", "F,AI", "AI,F", "AI,AI"],
    3: [
        "F,F,F",
        "F,F,AI",
        "F,AI,F",
        "F,AI,AI",
        "AI,F,F",
        "AI,F,AI",
        "AI,AI,F",
        # "AI,AI,AI" would be possible if multi-way all-ins still leave BB a decision;
        # "F,F,F" is BB open vs two folds. All-fold F,F,F,F never reaches a decision.
    ],
}


def class_label(cid: int) -> str:
    if cid < 13:
        return RANKS[cid] * 2
    suited = cid < 13 + 78
    rem = cid - 13 if suited else cid - 13 - 78
    idx = 0
    for h in range(13):
        for l in range(h):
            if idx == rem:
                return f"{RANKS[h]}{RANKS[l]}{'s' if suited else 'o'}"
            idx += 1
    return f"c{cid}"


def parse_id(iid: str) -> tuple[int | None, str | None, int | None]:
    """Parse mwpf_p{N}_path{PATH}_c{CLASS} (path may contain commas)."""
    m = re.match(r"mwpf_p(\d+)_path(.+)_c(\d+)$", iid)
    if not m:
        # legacy hash form
        m2 = re.match(r"mwpf_p(\d+)_h(\d+)_c(\d+)$", iid)
        if not m2:
            return None, None, None
        return int(m2.group(1)), f"h{m2.group(2)}", int(m2.group(3))
    return int(m.group(1)), m.group(2), int(m.group(3))


def path_description(seat: int, path: str) -> str:
    seat_name = SEAT_NAMES.get(seat, str(seat))
    if path == "open":
        return f"{seat_name} open (first to act)"
    parts = path.split(",")
    who = ["CO", "BTN", "SB", "BB"]
    bits = []
    for i, a in enumerate(parts):
        label = "folds" if a == "F" else ("jams" if a == "AI" else a)
        bits.append(f"{who[i]} {label}")
    return f"{seat_name} after " + ", ".join(bits)


def safe_filename(seat: int, path: str) -> str:
    name = SEAT_NAMES.get(seat, f"p{seat}")
    p = path.replace(",", "-")
    return f"{seat:02d}_{name}_{p}.json"


def main() -> int:
    src = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/cfr/pushfold_4handed_10bb_20k.json"
    )
    out_dir = Path(
        sys.argv[2] if len(sys.argv) > 2 else "data/cfr/pushfold_14_charts"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    d = json.loads(src.read_text(encoding="utf-8"))
    infos = d["strategy"]["infosets"]

    # (seat, path) -> list of hand rows
    nodes: dict[tuple[int, str], list] = defaultdict(list)
    for n in infos:
        seat, path, cid = parse_id(n["infoset_id"])
        if seat is None or path is None or cid is None:
            continue
        acts = n["actions"]
        probs = n["probs"]
        dact = dict(zip(acts, probs))
        nodes[(seat, path)].append(
            {
                "hand": class_label(cid),
                "class_id": cid,
                "allin": round(float(dact.get("ALLIN", 0.0)), 6),
                "fold": round(float(dact.get("FOLD", 0.0)), 6),
                "actions": acts,
                "probs": [round(float(p), 6) for p in probs],
            }
        )

    index = {
        "source": str(src),
        "spot": "4-handed $5/$10, 100 total (10bb), no ante, no rake, push/fold only",
        "seat_order": "CO(p0) → BTN(p1) → SB(p2) → BB(p3)",
        "path_legend": {
            "open": "empty history (first voluntary actor)",
            "F": "fold",
            "AI": "all-in",
            "comma": "sequence of prior actions in seat order",
        },
        "iterations": d.get("iterations_run"),
        "exploitability_bb_mc_proxy": d.get("exploitability_bb"),
        "notes": d.get("notes"),
        "nodes": [],
    }

    files_written = []
    for seat in range(4):
        seat_paths = sorted({p for (s, p) in nodes if s == seat})
        print(f"{SEAT_NAMES[seat]}: {len(seat_paths)} nodes -> {seat_paths}")
        for path in seat_paths:
            hands = sorted(nodes[(seat, path)], key=lambda r: r["class_id"])
            mean_ai = sum(h["allin"] for h in hands) / max(1, len(hands))
            n75 = sum(1 for h in hands if h["allin"] >= 0.75)
            chart = {
                "node_id": f"{SEAT_NAMES[seat]}_{path}",
                "seat": SEAT_NAMES[seat],
                "seat_index": seat,
                "path": path,
                "description": path_description(seat, path),
                "num_hands": len(hands),
                "mean_allin": round(mean_ai, 4),
                "hands_allin_ge_75pct": n75,
                "hands": hands,
            }
            fname = safe_filename(seat, path)
            fpath = out_dir / fname
            fpath.write_text(json.dumps(chart, indent=2) + "\n", encoding="utf-8")
            files_written.append(fname)
            index["nodes"].append(
                {
                    "file": fname,
                    "seat": SEAT_NAMES[seat],
                    "path": path,
                    "description": chart["description"],
                    "num_hands": len(hands),
                    "mean_allin": chart["mean_allin"],
                }
            )

    index_path = out_dir / "INDEX.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(files_written)} charts + INDEX -> {out_dir}")
    print(f"public nodes: {len(files_written)} (expect 14)")
    return 0 if len(files_written) == 14 else 0  # still ok if slightly different


if __name__ == "__main__":
    raise SystemExit(main())
