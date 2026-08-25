# Append memory index line if missing
from pathlib import Path
m = Path("memory/MEMORY.md")
t = m.read_text(encoding="utf-8")
line = "- [vMin1 bare-visibility obs experiment](project_vmin1_minimal_obs_experiment.md) — obs_mode=minimal (796 table-visible dims) + smaller net; overnight A/B vs vSix4; cold-start only\n"
if "vMin1 bare-visibility" not in t:
    # after v6 value gap line if present
    anchor = "v6 own/true value gaps"
    if anchor in t:
        # insert after that line
        lines = t.splitlines(True)
        out = []
        for L in lines:
            out.append(L)
            if anchor in L:
                out.append(line)
        m.write_text("".join(out), encoding="utf-8")
    else:
        m.write_text(t + "\n" + line, encoding="utf-8")
    print("MEMORY.md updated")
else:
    print("MEMORY.md already indexed")
