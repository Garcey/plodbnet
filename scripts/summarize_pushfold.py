"""Summarize 4-handed push/fold solve for Monker comparison."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ranks = "23456789TJQKA"


def class_label(cid: int) -> str:
    if cid < 13:
        return ranks[cid] * 2
    suited = cid < 13 + 78
    rem = cid - 13 if suited else cid - 13 - 78
    idx = 0
    for h in range(13):
        for l in range(h):
            if idx == rem:
                return f"{ranks[h]}{ranks[l]}{'s' if suited else 'o'}"
            idx += 1
    return f"c{cid}"


def parse(iid: str):
    p = h = c = None
    for part in iid.split("_"):
        if part.startswith("p") and part[1:].isdigit():
            p = int(part[1:])
        elif part.startswith("h") and part[1:].isdigit():
            h = int(part[1:])
        elif part.startswith("c") and part[1:].isdigit():
            c = int(part[1:])
    return p, h, c


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cfr/pushfold_4handed_10bb_20k.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    infos = d["strategy"]["infosets"]
    print(f"file={path}")
    print(f"status={d['status']} iters={d['iterations_run']} expl_bb={d['exploitability_bb']:.3f}")
    print(f"notes={d['notes']}")
    print(f"infosets={len(infos)}")

    by_ph: dict[tuple[int, int], list] = defaultdict(list)
    for n in infos:
        p, h, c = parse(n["infoset_id"])
        by_ph[(p, h)].append(n)

    # UTG open
    utg_h = next(h for (p, h) in by_ph if p == 0)
    utg = by_ph[(0, utg_h)]
    rows = []
    for n in utg:
        c = parse(n["infoset_id"])[2]
        ai = dict(zip(n["actions"], n["probs"])).get("ALLIN", 0.0)
        rows.append((class_label(c), ai, c))
    rows.sort(key=lambda x: -x[1])
    shove = [r for r in rows if r[1] >= 0.75]
    mixed = [r for r in rows if 0.25 <= r[1] < 0.75]
    fold = [r for r in rows if r[1] < 0.25]
    mean = sum(r[1] for r in rows) / len(rows)
    print(f"\n=== UTG/CO open shove (n={len(rows)}) mean_AI={mean:.3f} ===")
    print(f"ALLIN>=75% ({len(shove)}):", ", ".join(f"{a}:{b:.2f}" for a, b, _ in shove))
    print(f"mixed 25-75% ({len(mixed)}):", ", ".join(f"{a}:{b:.2f}" for a, b, _ in mixed))
    print(f"fold <25% ({len(fold)}):", ", ".join(a for a, _, _ in fold))

    seat_names = {0: "UTG/CO", 1: "BTN", 2: "SB", 3: "BB"}
    for seat in range(4):
        print(f"\n=== {seat_names[seat]} histories ===")
        for (p, h), nodes in sorted(by_ph.items(), key=lambda x: (x[0][0], -len(x[1]))):
            if p != seat:
                continue
            mean_ai = sum(
                dict(zip(n["actions"], n["probs"])).get("ALLIN", 0.0) for n in nodes
            ) / len(nodes)
            n75 = sum(
                1
                for n in nodes
                if dict(zip(n["actions"], n["probs"])).get("ALLIN", 0.0) >= 0.75
            )
            tops = sorted(
                nodes,
                key=lambda n: -dict(zip(n["actions"], n["probs"])).get("ALLIN", 0.0),
            )
            labs = [
                (
                    class_label(parse(n["infoset_id"])[2]),
                    dict(zip(n["actions"], n["probs"])).get("ALLIN", 0.0),
                )
                for n in tops
                if dict(zip(n["actions"], n["probs"])).get("ALLIN", 0.0) >= 0.75
            ]
            print(f"  h={h} n={len(nodes)} mean_AI={mean_ai:.3f} n>=75%={n75}")
            print("   shove:", ", ".join(f"{a}:{b:.2f}" for a, b in labs[:35]))

    hands = [
        {
            "hand": lab,
            "class_id": c,
            "allin": round(ai, 4),
            "fold": round(1 - ai, 4),
        }
        for lab, ai, c in sorted(rows, key=lambda x: x[0])
    ]
    out = Path("data/cfr/pushfold_4handed_10bb_UTG_chart.json")
    out.write_text(
        json.dumps(
            {
                "spot": "4-handed $5/$10, 100 total (10bb), no ante, no rake, push/fold only",
                "seat": "UTG/CO (seat 0, first to act)",
                "source_file": str(path),
                "iterations": d["iterations_run"],
                "exploitability_bb_mc_proxy": d["exploitability_bb"],
                "hands": hands,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
