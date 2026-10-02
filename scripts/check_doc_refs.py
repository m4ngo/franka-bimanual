#!/usr/bin/env python3
"""Check that `file.py#L<n>` links in the repo's markdown still point at code.

    python scripts/check_doc_refs.py [doc ...]

Line-numbered links rot the moment anything is inserted above them, and a
reference that silently drifts is worse than none -- it sends a reader to the
wrong function while looking authoritative. This resolves every relative link,
and for a `#L<n>` one prints the line it lands on so the claim can be eyeballed.

Heuristic, deliberately: a link whose text names a `def`/`class` is CHECKED
against the target line, because that is the case worth catching automatically.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = ("baselines/LIBERO_SIM.md", "baselines/README.md", "ENTRYPOINTS.md")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def check(doc: Path) -> tuple[int, int]:
    problems = checked = 0
    for m in _LINK.finditer(doc.read_text()):
        text, target = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "#")):
            continue
        rel, _, frag = target.partition("#")
        path = (doc.parent / rel).resolve()
        if not path.is_file():
            print(f"{doc}: missing file -> {target}")
            problems += 1
            continue
        if not frag.startswith("L"):
            continue
        lines = path.read_text().splitlines()
        n = int(frag[1:])
        if not 1 <= n <= len(lines):
            print(f"{doc}: line {n} outside {rel} ({len(lines)} lines)")
            problems += 1
            continue
        checked += 1
        line = lines[n - 1].strip()
        # Only when the link text looks like an identifier: a prose link such as
        # "the settle" carries no claim about what is on that line.
        name = _NAME.fullmatch(text.strip("`"))
        if name and ("def " in line or "class " in line or "=" in line):
            symbol = text.strip("`").split(".")[-1].split("(")[0]
            if symbol not in line:
                print(f"{doc}: [{text}] -> {rel}:{n} is {line[:70]!r}")
                problems += 1
    return problems, checked


def main() -> int:
    docs = [Path(a) for a in sys.argv[1:]] or [_REPO_ROOT / d for d in DEFAULT_DOCS]
    total = links = 0
    for doc in docs:
        if not doc.is_file():
            print(f"no such doc: {doc}")
            total += 1
            continue
        p, c = check(doc)
        total += p
        links += c
    print(f"{links} line-numbered reference(s) checked, {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
