"""Chart a run's per-update metrics (runs/<stem>.metrics.jsonl) -- no plotting
library needed: writes ONE self-contained HTML file of small SVG line charts
(or prints a table with --table).

    python scripts/plot_metrics.py runs/vSix6.metrics.jsonl               # -> runs/vSix6.metrics.html
    python scripts/plot_metrics.py runs/vSix6.metrics.jsonl --table --last 20
    python scripts/plot_metrics.py runs/a.metrics.jsonl runs/b.metrics.jsonl -o ab.html
    python scripts/plot_metrics.py runs/vSix6.metrics.jsonl --keys ppo.entropy,value_health.all.ev

A key is a dotted path into each record (train.py writes them: see
plo5bp/train/loop.py `metrics.write`); several files are overlaid per chart.
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from plo5bp.train.metrics import read_metrics  # noqa: E402

DEFAULT_KEYS = (
    "ppo.entropy", "ppo.gate_entropy", "ppo.approx_kl", "ppo.kl_k3", "ppo.kl0",
    "ppo.clip_frac", "ppo.value_loss", "ppo.critic_value_loss", "ppo.q_loss",
    "value_health.all.ev", "value_health.all.bias",
    "ppo.grad_norm_actor", "ppo.grad_norm_critic", "ppo.grad_clip_actor",
    "rows", "rows_per_s", "update_s", "lr", "entropy_coef",
)
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def get(record: dict, key: str) -> "float | None":
    cur: object = record
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None
    return float(cur) if math.isfinite(float(cur)) else None


def series(records: list[dict], key: str) -> list[tuple[float, float]]:
    out = []
    for r in records:
        x, y = get(r, "update"), get(r, key)
        if x is not None and y is not None:
            out.append((x, y))
    return out


def svg_chart(title: str, lines: list[tuple[str, list[tuple[float, float]]]],
              w: int = 420, h: int = 200) -> str:
    pts = [p for _, s in lines for p in s]
    if not pts:
        return ""
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y0, y1 = y0 - 1e-9 - abs(y0) * 0.05, y1 + 1e-9 + abs(y1) * 0.05
    pad_l, pad_r, pad_t, pad_b = 56, 10, 22, 22

    def sx(x: float) -> float:
        return pad_l + (x - x0) / (x1 - x0) * (w - pad_l - pad_r)

    def sy(y: float) -> float:
        return h - pad_b - (y - y0) / (y1 - y0) * (h - pad_t - pad_b)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">',
        f'<text x="{pad_l}" y="14" class="t">{html.escape(title)}</text>',
        f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" class="ax"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" class="ax"/>',
        f'<text x="{pad_l - 4}" y="{sy(y1) + 4:.1f}" class="l" text-anchor="end">{y1:.4g}</text>',
        f'<text x="{pad_l - 4}" y="{sy(y0):.1f}" class="l" text-anchor="end">{y0:.4g}</text>',
        f'<text x="{pad_l}" y="{h - 6}" class="l">u{x0:g}</text>',
        f'<text x="{w - pad_r}" y="{h - 6}" class="l" text-anchor="end">u{x1:g}</text>',
    ]
    for i, (_name, s) in enumerate(lines):
        if not s:
            continue
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s)
        parts.append(
            f'<polyline points="{d}" fill="none" stroke="{COLORS[i % len(COLORS)]}" '
            'stroke-width="1.5"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS),
                    help="comma-separated dotted keys to chart")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="HTML file (default: <first file>.html)")
    ap.add_argument("--table", action="store_true", help="print a table instead")
    ap.add_argument("--last", type=int, default=0, help="only the last N updates")
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    runs = []
    for f in args.files:
        recs = read_metrics(f)
        if args.last > 0:
            recs = recs[-args.last:]
        runs.append((f.name.replace(".metrics.jsonl", ""), recs))

    if args.table:
        for name, recs in runs:
            print(f"== {name} ({len(recs)} updates)")
            print("update  " + "  ".join(f"{k.split('.')[-1][:12]:>12s}" for k in keys))
            for r in recs:
                cells = []
                for k in keys:
                    v = get(r, k)
                    cells.append(f"{v:12.5g}" if v is not None else f"{'-':>12s}")
                print(f"{int(get(r, 'update') or 0):6d}  " + "  ".join(cells))
        return 0

    charts = [
        svg_chart(k, [(name, series(recs, k)) for name, recs in runs]) for k in keys
    ]
    legend = " ".join(
        f'<span style="color:{COLORS[i % len(COLORS)]}">&#9632; {html.escape(n)}</span>'
        for i, (n, _) in enumerate(runs)
    )
    out = args.out or args.files[0].with_suffix(".html")
    out.write_text(
        "<!doctype html><meta charset='utf-8'><title>Training metrics</title>"
        "<style>body{font:13px system-ui,sans-serif;margin:16px;background:#fff;color:#111}"
        ".g{display:flex;flex-wrap:wrap;gap:12px}svg{background:#fafafa;border:1px solid #e5e5e5}"
        ".t{font-weight:600;font-size:12px}.l{font-size:10px;fill:#555}.ax{stroke:#bbb}</style>"
        f"<p>{legend}</p><div class='g'>{''.join(c for c in charts if c)}</div>",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
