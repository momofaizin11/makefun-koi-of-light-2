#!/usr/bin/env python3
"""make.fun entry - "Koi of Light".

Copies the artwork sitting next to this file (entry.png) into the output
directory, byte for byte. Standard library only, deterministic, offline.

Output: $MAKEFUN_OUTPUT_DIR/entry.png  (default: /output/entry.png)
"""

import os
import pathlib

ENTRY_NAME = "entry.png"
SOURCE = pathlib.Path(__file__).resolve().parent / ENTRY_NAME
OUTPUT_DIR = pathlib.Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ENTRY_NAME).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
