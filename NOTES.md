# Notes

Concept: a koi drawn as living light rather than as a photographed fish. The
warm gold body against the cold dark water carries the whole idea.

How the studio works (tools/render.py, numpy + Pillow, no randomness):

- A spine curve runs from the blunt head, upper right, to the peduncle, lower
  left; it is resampled by arc length so proportions stay chunky - deep
  shoulders, narrow tail root.
- Fins are grown off that spine as ribbons, then blurred and sunk into the
  body so they read as silk rather than as cut-outs.
- The fish is painted as layers: a translucent wash that lets the water show
  through, light bleeding into the surrounding water, kohaku-like patches of
  brightness, scale glints, and a flare that glues the forked tail into the
  body.
- The water layer adds a soft pool of light, tilted shafts, caustic ripples
  and a few motes and bubble rings.
- A final bloom pass softens the highlights. The eye and barbels go on last,
  above the bloom, so the head always reads clearly even at thumbnail size.

The entry itself is static: `main.py` writes the finished `entry.png` byte for
byte into `$MAKEFUN_OUTPUT_DIR` (default `/output`), standard library only, so
the delivered picture is exactly the frame the studio produced.
