# Koi of Light

A koi drawn as living light, swimming through dark water: head and bright eye
to the upper right, forked silky tail sweeping down to the lower left.

The image is 1024x1024, 8-bit RGB, and is produced by `tools/render.py`
(numpy + Pillow, no randomness - the same frame every run). Palette: amber-gold
body (around #FDDB9E with #EAC590 glow) against deep blue-black water, with
pale blue caustics and a cream rim that separates the fish from the dark.

Regenerate the picture: `python3 tools/render.py entry.png` (also writes a
300px twin).
Run the entry: `MAKEFUN_OUTPUT_DIR=/somewhere python3 main.py` copies `entry.png`
byte for byte with the standard library only (default output `/output`).
