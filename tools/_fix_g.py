from pathlib import Path
p = Path("scripts/vMin1_guardian.sh")
t = p.read_text(encoding="utf-8")
if "checkpoints/vMin1.pt" not in t.split("train_pid")[2][:500]:
    t = t.replace(
        '  L=$(ls -t checkpoints/vMin1_*.pt 2>/dev/null | head -1)\n  if [ -n "${L:-}" ]; then\n',
        '  L=$(ls -t checkpoints/vMin1_*.pt 2>/dev/null | head -1)\n  if [ -z "${L:-}" ] && [ -f checkpoints/vMin1.pt ]; then\n    L=checkpoints/vMin1.pt\n  fi\n  if [ -n "${L:-}" ]; then\n',
    )
    p.write_text(t, encoding="utf-8", newline="\n")
    print("local guardian fixed")
else:
    print("local already ok or different")
print(p.read_text()[p.read_text().find("if [ -z"):p.read_text().find("if [ -z")+400])
