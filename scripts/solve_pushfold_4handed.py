"""4-handed push/fold preflop solve: $5/$10, 100 total each, no ante, no rake.

Seat convention (solver multiway preflop):
  0 = UTG (CO in 4-handed), 1 = BTN, 2 = SB, 3 = BB
Starting stacks include blinds: each seat has 100bb of chips before posting.
After posting: SB has 95 behind, BB has 90 behind (5/10 chips if bb=10k engine units).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, solve, rust_cfr_available
from plo5bp.gto.roots import CLUBGG_NLH_ROOT

def main() -> int:
    print(f"rust_cfr_available={rust_cfr_available()}")
    # Chip scale: 1 bb = 10000 engine chips (ClubGG), so $10 bb â†’ bb_chips=10000
    # $5/$10, no ante. Starting stack $100 = 10 bb total.
    bb = 10_000
    sb = 5_000
    ante = 0
    stack_bb = 10.0  # $100 / $10
    n = 4
    # pot after blinds only = 1.5 bb
    pot_bb = (sb + bb) / float(bb)  # 1.5

    root = RootSpec(
        street=0,
        pot_bb=pot_bb,
        effective_stack_bb=stack_bb,
        board=[],
        num_seats=n,
        bb_chips=bb,
        sb_chips=sb,
        ante_chips=ante,
        raise_sizes_pm=[],          # empty + allin = pure push/fold
        allin_atom=True,
        stacks_bb=[stack_bb] * n,  # starting totals; blinds deducted inside solver
        root_id="pf4_pushfold_10bb_5_10_noante_norake",
    )
    cfg = SolveConfig(
        max_iterations=300000,
        target_exploitability_bb=0.5,
        thread_num=1,
        seed=42,
        algorithm="mccfr_es",
        card_abstraction="none",
    )
    print("solving 4-handed push/fold preflop ...")
    print(json.dumps(root.as_dict(), indent=2))
    report = solve(root, cfg)
    out = Path("data/cfr/pushfold_4handed_10bb_300k.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(out)
    print(f"status={report.status} iters={report.iterations_run} expl={report.exploitability_bb}")
    print("notes:", report.notes)
    # Summarize open-shove freqs by seat for pure open nodes (empty history)
    infosets = report.strategy.get("infosets") or []
    print(f"infosets={len(infosets)}")
    # Group root open decisions: history empty-ish ids containing _h0_ or similar
    by_player = {}
    for node in infosets:
        iid = node["infoset_id"]
        # mwpf_p{player}_h{history}_c{class}
        parts = iid.split("_")
        # find pN
        p = None
        h = None
        c = None
        for part in parts:
            if part.startswith("p") and part[1:].isdigit():
                p = int(part[1:])
            elif part.startswith("h") and part[1:].isdigit():
                h = int(part[1:])
            elif part.startswith("c") and part[1:].isdigit():
                c = int(part[1:])
        if p is None:
            continue
        acts = node["actions"]
        probs = node["probs"]
        by_player.setdefault(p, []).append((h, c, acts, probs))

    # Print open shove % (history==0) for each seat, top shove hands
    from collections import defaultdict
    # Need class labels
    try:
        from plo5bp import _engine
        # build 169 labels via a tiny helper if available
    except Exception:
        pass

    ranks = "23456789TJQKA"
    def class_label(cid: int) -> str:
        if cid < 13:
            r = ranks[cid]
            return f"{r}{r}"
        suited = cid < 13 + 78
        rem = cid - 13 if suited else cid - 13 - 78
        idx = 0
        for h in range(13):
            for l in range(h):
                if idx == rem:
                    s = "s" if suited else "o"
                    return f"{ranks[h]}{ranks[l]}{s}"
                idx += 1
        return f"c{cid}"

    seat_names = {0: "UTG/CO", 1: "BTN", 2: "SB", 3: "BB"}
    summary = {}
    for p, nodes in sorted(by_player.items()):
        open_nodes = [(h, c, a, pr) for (h, c, a, pr) in nodes if h == 0]
        # Also collect any history for BB facing
        shove_weighted = 0.0
        fold_weighted = 0.0
        pure_shoves = []
        for h, c, acts, probs in open_nodes:
            # equal class weight approx for summary
            d = dict(zip(acts, probs))
            ai = d.get("ALLIN", 0.0)
            fo = d.get("FOLD", 0.0)
            shove_weighted += ai
            fold_weighted += fo
            if ai >= 0.5:
                pure_shoves.append((class_label(c), ai))
        pure_shoves.sort(key=lambda x: -x[1])
        summary[seat_names.get(p, str(p))] = {
            "open_infosets": len(open_nodes),
            "mean_allin_among_open": shove_weighted / max(1, len(open_nodes)),
            "top_shoves": pure_shoves[:25],
        }
        print(f"\n=== seat {p} ({seat_names.get(p)}) open (h=0) ===")
        print(f"  open_infosets={len(open_nodes)} mean_ALLIN={shove_weighted/max(1,len(open_nodes)):.3f}")
        print("  top ALLIN>=0.5:", ", ".join(f"{h}:{p:.2f}" for h,p in pure_shoves[:20]))

    summary_path = Path("data/cfr/pushfold_4handed_10bb_300k_summary.json")
    summary_path.write_text(json.dumps({"report_meta": {
        "status": report.status,
        "iterations": report.iterations_run,
        "exploitability_bb": report.exploitability_bb,
        "notes": report.notes,
        "num_infosets": len(infosets),
    }, "by_seat_open": summary}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out} and {summary_path}")
    return 0 if report.status == "ok" else 1

if __name__ == "__main__":
    raise SystemExit(main())


