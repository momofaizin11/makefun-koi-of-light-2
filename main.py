#!/usr/bin/env python3
"""make.fun entry - "Koi of Light".

Copies the artwork sitting next to this file (entry.png) into the output
directory, byte for byte. Standard library only, deterministic, offline, and it
never redraws anything.

Output: $MAKEFUN_OUTPUT_DIR/entry.png  (fallback: /output/entry.png)
Exits non-zero if the copy is missing or not byte-identical to entry.png.
"""

import hashlib
import os
import shutil
import sys

NAME = "entry.png"
FALLBACK_DIR = "/output"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), NAME)
    if not os.path.isfile(src):
        print("entry.png not found next to main.py", file=sys.stderr)
        return 2

    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "").strip() or FALLBACK_DIR
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, NAME)

    shutil.copyfile(src, dst)

    src_size, dst_size = os.path.getsize(src), os.path.getsize(dst)
    if dst_size != src_size:
        print("copy size mismatch: %d != %d" % (dst_size, src_size), file=sys.stderr)
        return 3
    if sha256(dst) != sha256(src):
        print("copy is not byte-identical to entry.png", file=sys.stderr)
        return 4

    print("copied %s -> %s (%d bytes, sha256 %s)" % (src, dst, dst_size, sha256(dst)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
