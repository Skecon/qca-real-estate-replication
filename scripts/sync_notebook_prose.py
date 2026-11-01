"""Push markdown edits from the generator into the executed notebooks.

`build_notebooks.py --run` re-executes everything, which takes about fifteen
minutes because of the pattern-by-pattern MLP. When only the prose has changed
there is nothing to recompute, so this copies the markdown across and leaves
every code cell and its stored output untouched.

    python scripts/sync_notebook_prose.py

It refuses to touch a notebook whose code cells no longer line up with the
generator, because then the outputs on disk no longer belong to the code and a
full re-execution is the only honest option.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_notebooks import NB_DIR, NOTEBOOKS  # noqa: E402


def main() -> int:
    stale = []
    for name, generated in NOTEBOOKS.items():
        path = NB_DIR / name
        if not path.exists():
            print(f"{name}: not written yet, skipping")
            continue

        on_disk = nbf.read(path, as_version=4)

        gen_code = [c.source for c in generated.cells if c.cell_type == "code"]
        disk_code = [c.source for c in on_disk.cells if c.cell_type == "code"]
        if gen_code != disk_code:
            stale.append(name)
            print(f"{name}: CODE HAS CHANGED -- re-execute, prose not synced")
            continue

        changed = 0
        for disk_cell, gen_cell in zip(on_disk.cells, generated.cells):
            if disk_cell.cell_type != gen_cell.cell_type:
                stale.append(name)
                break
            if disk_cell.cell_type != "markdown":
                continue
            if disk_cell.source != gen_cell.source:
                disk_cell.source = gen_cell.source
                changed += 1
        else:
            nbf.write(on_disk, path)
            print(f"{name}: {changed} markdown cell(s) updated, outputs preserved")

    if stale:
        print(
            "\nRe-execute these with `python scripts/build_notebooks.py --run`:\n  "
            + "\n  ".join(sorted(set(stale)))
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
