#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check local Markdown file links in baiez's project and agent documents."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def documents() -> list[Path]:
    return sorted(
        [
            *ROOT.glob("*.md"),
            *(ROOT / "e2e").glob("*.md"),
            *(ROOT / ".agents/docs").rglob("*.md"),
            *(ROOT / ".agents/skills").rglob("*.md"),
        ]
    )


def main() -> int:
    missing = []
    checked = 0
    for document in documents():
        for raw in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw.removeprefix("<").removesuffix(">")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            checked += 1
            destination = document.parent / unquote(parsed.path)
            if not destination.exists():
                missing.append((document.relative_to(ROOT), target))
    for source, target in missing:
        print(f"{source}: missing local link {target}", file=sys.stderr)
    if missing:
        return 1
    print(f"Checked {checked} local links in {len(documents())} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
