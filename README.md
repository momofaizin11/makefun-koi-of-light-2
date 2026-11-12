# Koi of Light

A koi fish made of light, swimming through dark water.

The picture is 1024x1024, 8-bit RGB. A single carp hangs on the diagonal: head
and bright eye toward the upper right, tail sweeping down to the lower left.
Its body is translucent amber, lit from inside, with patches of brighter light
along the flank. Soft silk fins sit on the body, the caudal fin forks and
frays, and two thin barbels mark the mouth. Around it the water is deep
blue-black, crossed by tilted light shafts, faint caustic ripples and a few
ring-shaped bubbles.

`tools/render.py` is the studio that drew it: a numpy + Pillow program that
builds the fish procedurally from a curve skeleton, then composites glow,
water and bloom into one deterministic frame. `main.py` is the entry - it
copies the finished `entry.png` byte for byte into `$MAKEFUN_OUTPUT_DIR`
(default `/output`) using the standard library only.
