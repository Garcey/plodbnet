# Python dependencies

`pyproject.toml` says what each part of the project NEEDS (ranges, by extra:
`web`, `public`, `desktop`, `ocr`, `dev`). The files here pin what each MACHINE
actually runs, so an install is reproducible and a server rebuild does not jump
several major versions at once.

| File | Machine | How it is made |
|---|---|---|
| `server.in` | production server — direct dependencies | edited by hand |
| `server.txt` | production server — every package, exact | `bash scripts/deploy_prod.sh freeze` (from the live server) or `… lock` (resolve `server.in` on the server) |
| `pod.in` | training pod — direct dependencies | edited by hand |
| `pod-freeze.txt` | training pod — exact | `pip freeze --exclude-editable` on the pod |
| `desktop-freeze.txt` | the Windows desktop — exact | `pip freeze --exclude-editable` |

The deploy checks the live server's packages against `server.txt` on every deploy:
if they match it reuses the live environment; if `server.txt` changed it builds a
new environment beside the live one (`/opt/wrapgto/venvs/<hash>`), tests the new
code in it, and switches both together — nothing is ever `pip install`ed into the
live environment by hand. `bash scripts/deploy_prod.sh check` reports any drift.
