"""Convert a `# %%`-cell Python script into a Jupyter notebook (no external deps).

Usage: python tools/py2ipynb.py notebooks/cheat_dinov2_temporal.py kaggle/<slug>/<slug>.ipynb
"""
import json
import sys
from pathlib import Path


def convert(src: Path, dst: Path) -> None:
    cells, cur, kind = [], [], "code"

    def flush():
        text = "\n".join(cur).strip("\n")
        if text:
            if kind == "markdown":
                text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            lines = text.splitlines(keepends=True)
            cell = {"cell_type": kind, "metadata": {}, "source": lines}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            cur, kind = [], ("markdown" if "[markdown]" in line else "code")
        else:
            cur.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                        "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{dst}: {len(cells)} cells")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
