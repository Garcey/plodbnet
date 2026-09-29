"""Export a 4-handed push/fold solve into its 14 labeled strategy charts.

Reads a solve JSON whose infoset_ids use path labels:
  mwpf_p{seat}_path{open|F|AI|F,F|...}_c{class}

Writes one JSON chart per public node (+ INDEX.json) under
data/cfr/pushfold_14_charts/.

Exit codes (TOOL-061 — it used to "succeed" whatever it found):
  0  exactly the 14 expected public nodes were exported
  1  some expected node is missing (too few iterations for every class to reach
     it) or an unexpected one appeared — the charts are written for inspection
  2  the input is not a 4-handed push/fold solve (wrong seats / raise menu)
The INDEX "spot" text is derived from the input's root (seats, stacks, blinds,
ante), not a hard-coded description of one particular solve.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.jsonio import atomic_write_json  # noqa: E402
from plo5bp.gto.preflop_class import preflop_class_label  # noqa: E402

SEAT_NAMES = {0: "CO", 1: "BTN", 2: "SB", 3: "BB"}

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


def expected_nodes() -> set[tuple[int, str]]:
    return {(seat, path) for seat, paths in EXPECTED.items() for path in paths}


def _fmt_bb(x: float) -> str:
    return f"{x:g}bb"


def spot_text(root: dict) -> str:
    """Human description of the solved spot, from the report's root."""
    n = int(root.get("num_seats") or 0)
    bb = float(root.get("bb_chips") or 0) or 1.0
    stacks = [float(x) for x in (root.get("stacks_bb") or [])]
    if not stacks and root.get("effective_stack_bb") is not None:
        stacks = [float(root["effective_stack_bb"])] * max(n, 1)
    uniq = sorted(set(stacks))
    stack_txt = (
        f"{_fmt_bb(uniq[0])} stacks" if len(uniq) == 1
        else "stacks " + "/".join(_fmt_bb(x) for x in stacks)
    )
    sb = float(root.get("sb_chips") or 0) / bb
    ante = float(root.get("ante_chips") or 0) / bb
    ante_txt = f"ante {_fmt_bb(ante)}" if ante > 0 else "no ante"
    menu = "push/fold only" if not root.get("raise_sizes_pm") and root.get("allin_atom", True) \
        else f"raise sizes {root.get('raise_sizes_pm')}"
    return f"{n}-handed, {stack_txt}, blinds {_fmt_bb(sb)}/1bb, {ante_txt}, no rake, {menu}"


def root_problem(root: dict) -> str | None:
    """Why this report is not a 4-handed push/fold solve (None = it is)."""
    if int(root.get("num_seats") or 0) != len(SEAT_NAMES):
        return f"num_seats={root.get('num_seats')} (the 14-chart tree is 4-handed)"
    if root.get("raise_sizes_pm") or not root.get("allin_atom", True):
        return f"raise menu {root.get('raise_sizes_pm')} is not push/fold (FOLD|ALLIN)"
    return None


def class_label(cid: int) -> str:
    """169-class label — the ONE mapping, plo5bp.gto.preflop_class (TOOL-048)."""
    return preflop_class_label(int(cid))


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
    root = d.get("root") if isinstance(d.get("root"), dict) else {}
    problem = root_problem(root)
    if problem is not None:
        print(f"[export14] not a 4-handed push/fold solve: {problem}", file=sys.stderr)
        return 2
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
        "spot": spot_text(root),
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
            atomic_write_json(fpath, chart, indent=2)
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
    got = set(nodes)
    want = expected_nodes()
    missing, extra = sorted(want - got), sorted(got - want)
    index["expected_nodes"] = len(want)
    index["missing_nodes"] = [f"{SEAT_NAMES[s]}:{p}" for s, p in missing]
    index["unexpected_nodes"] = [f"{SEAT_NAMES.get(s, s)}:{p}" for s, p in extra]
    atomic_write_json(index_path, index, indent=2)
    print(f"\nwrote {len(files_written)} charts + INDEX -> {out_dir}")
    print(f"public nodes: {len(files_written)} (expect {len(want)})")
    if missing or extra:
        print(
            f"[export14] MISMATCH: missing {index['missing_nodes']} "
            f"unexpected {index['unexpected_nodes']} — solve longer or check the root",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
