#!/usr/bin/env python3
"""Koi of light - procedural studio renderer (1024x1024 RGB, deterministic).

v5: chunky koi proportions (arc-length driven), broad forked caudal grown from
the peduncle, broad rounded silk fins sunk into the body, glossy eyes with a
crescent glint, underwater light shafts + caustics + few motes, diagonal
composition head-upper-right / tail-lower-left.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W = H = 1024
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/koi.png"

yy, xx = np.mgrid[0:H, 0:W]
XG = xx.astype(np.float32)
YG = yy.astype(np.float32)


def sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def over(scene, a, color):
    a = np.clip(a, 0, 1)[:, :, None]
    c = np.asarray(color, np.float32)[None, None, :] / 255.0
    return scene * (1 - a) + c * a


def catmull(pts, n=400):
    P = np.asarray(pts, float)
    P = np.vstack([P[0], P, P[-1]])
    segs = len(pts) - 1
    out, per = [], max(2, n // segs)
    for i in range(segs):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        t = np.linspace(0.0, 1.0, per, endpoint=(i == segs - 1))[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out).astype(np.float32)


def raster(poly, blur=1.2):
    l = Image.new("L", (W, H), 0)
    ImageDraw.Draw(l).polygon([(float(p[0]), float(p[1])) for p in poly], fill=255)
    if blur:
        l = l.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(l, np.float32) / 255.0


def strokes(segs, blur=1.6):
    l = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(l)
    for pts, w in segs:
        d.line([(float(p[0]), float(p[1])) for p in pts], fill=255, width=int(w),
               joint="curve")
    if blur:
        l = l.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(l, np.float32) / 255.0


def soft(m, r):
    """Blur a float mask (0..1) by r px."""
    if r <= 0:
        return m
    return np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(r)), np.float32) / 255.0


# ---------------------------------------------------------------- spine
# head upper-right -> peduncle lower-left; arc length ~660 px, height ~205 px
SPINE_RAW = [(798, 248), (692, 286), (586, 352), (482, 446), (380, 568), (276, 716)]
SPINE = catmull(SPINE_RAW, 420)
seg = np.diff(SPINE, axis=0)
slen = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
U = (slen / slen[-1]).astype(np.float32)
TAN = np.gradient(SPINE, axis=0)
TAN /= (np.linalg.norm(TAN, axis=1, keepdims=True) + 1e-6)
NOR = np.stack([-TAN[:, 1], TAN[:, 0]], axis=1)          # dorsal flank (R > 0 side)

d2 = np.full((H, W), 1e12, np.float32)
idx = np.zeros((H, W), np.int32)
for i in range(len(SPINE)):
    dx = XG - SPINE[i, 0]
    dy = YG - SPINE[i, 1]
    di = dx * dx + dy * dy
    m = di < d2
    d2 = np.where(m, di, d2)
    idx = np.where(m, i, idx)
del d2
PDX = XG - SPINE[idx, 0]
PDY = YG - SPINE[idx, 1]
CROSS = TAN[idx, 0] * PDY - TAN[idx, 1] * PDX
R = np.hypot(PDX, PDY) * np.sign(CROSS)
US = U[idx]
ra = np.abs(R)
dorsal = R > 0

# chunky koi: blunt head, deep shoulders, clear narrow peduncle
PROFU = [0.00, 0.04, 0.10, 0.18, 0.30, 0.42, 0.55, 0.68, 0.80, 0.90, 0.97, 1.00]
PROFW = [16.0, 46.0, 76.0, 100.0, 115.0, 114.0, 104.0, 88.0, 70.0, 54.0, 42.0, 36.0]
BW = np.interp(US, PROFU, PROFW).astype(np.float32)
BW = BW * np.where(dorsal, 0.88, 1.09)
body = sm((BW - ra) / 2.3)

hc = SPINE[6] + TAN[6] * 26.0
hl = (XG - hc[0]) * TAN[6][0] + (YG - hc[1]) * TAN[6][1]
hw = (XG - hc[0]) * -TAN[6][1] + (YG - hc[1]) * TAN[6][0]
head_cap = sm((1.0 - np.hypot(hl / 58.0, hw / 54.0)) / 0.24)
body = np.clip(np.maximum(body, head_cap), 0, 1)
# blunt jaw / mouth lip so the snout end reads as a fish head, not a dome
jc = SPINE[1] + TAN[1] * 18.0 - NOR[1] * 12.0
jl = (XG - jc[0]) * TAN[1][0] + (YG - jc[1]) * TAN[1][1]
jw = (XG - jc[0]) * -TAN[1][1] + (YG - jc[1]) * TAN[1][0]
lip = sm((1.0 - np.hypot(jl / 30.0, jw / 20.0)) / 0.28)
body = np.clip(np.maximum(body, lip), 0, 1)

# ---------------------------------------------------------------- fins
def ribbon(root, ang, length, halfw, bend=0.30, hook=0.18, sc_amp=0.13, sc_n=5.0,
           phase=0.0, taper=0.45, n=120):
    c, s = math.cos(ang), math.sin(ang)
    lead, trail, ctr = [], [], []
    hwd = hwt = halfw * 0.5
    for i in range(n):
        u = i / (n - 1)
        vc = bend * (u ** 1.6) + hook * math.sin(math.pi * u ** 0.9 + phase) * u
        hwd = halfw * (0.32 + 0.68 * math.sin(math.pi * u ** 0.62))
        hwd *= 1.0 - taper * sm((u - 0.58) / 0.42)
        hwt = hwd * (1.0 + sc_amp * (0.5 + 0.5 * math.cos(sc_n * 2 * math.pi * u + phase)))
        bx = root[0] + c * u * length - s * vc
        by = root[1] + s * u * length + c * vc
        lead.append((bx - s * hwd, by + c * hwd))
        trail.append((bx + s * hwt, by - c * hwt))
        ctr.append((bx, by))
    # rounded silk tip cap so the membrane does not end in a cut edge
    cap = []
    cx, cy = ctr[-1]
    hw_end = max(hwd, hwt)
    for i in range(1, 11):
        th = math.pi * i / 11.0
        cap.append((cx - s * math.cos(th) * hw_end + c * math.sin(th) * hw_end * 0.5,
                    cy + c * math.cos(th) * hw_end + s * math.sin(th) * hw_end * 0.5))
    return lead, trail, ctr, cap


def u_field(root, ang, length):
    c, s = math.cos(ang), math.sin(ang)
    dx, dy = XG - root[0], YG - root[1]
    return np.clip((dx * c + dy * s) / length, 0, 1)


def flank_ang(i, side, sweep_deg):
    """Angle for a fin root on flank `side` (+1 dorsal / -1 ventral), swept
    `sweep_deg` degrees toward the tail."""
    n = NOR[i] * side
    cross = n[0] * TAN[i][1] - n[1] * TAN[i][0]
    return math.atan2(n[1], n[0]) + math.radians(sweep_deg) * (1.0 if cross > 0 else -1.0)


A_tail = math.atan2(TAN[-1][1], TAN[-1][0]) + math.radians(16)   # sweeps left + down
TROOT = SPINE[-1] - TAN[-1] * 22.0
lob_up = ribbon(TROOT, A_tail - math.radians(22), 280, 96, bend=-0.24, hook=0.24,
                sc_amp=0.12, sc_n=4, phase=0.3, taper=0.24)
lob_dn = ribbon(TROOT, A_tail + math.radians(22), 262, 88, bend=0.26, hook=0.20,
                sc_amp=0.12, sc_n=4, phase=1.1, taper=0.26)
lob_mid = ribbon(TROOT, A_tail, 200, 56, bend=0.0, hook=0.18,
                 sc_amp=0.12, sc_n=3, phase=0.7, taper=0.38)
lob_root = ribbon(TROOT, A_tail, 130, 40, bend=0.0, hook=0.10, sc_amp=0.10, sc_n=3,
                  phase=0.0, taper=0.20)
o = SPINE[90] - NOR[90] * 3.0
pec1 = ribbon(o, flank_ang(90, -1, 42), 200, 68, bend=0.30, hook=0.24, sc_amp=0.14,
              sc_n=5, phase=0.2, taper=0.34)
o = SPINE[120] - NOR[120] * 3.0
pec2 = ribbon(o, flank_ang(120, -1, 58), 125, 44, bend=-0.30, hook=0.26, sc_amp=0.14,
              sc_n=4, phase=1.7, taper=0.40)
o = SPINE[196] + NOR[196] * 20.0
dors = ribbon(o, flank_ang(196, 1, 30), 145, 78, bend=0.28, hook=0.18, sc_amp=0.12,
              sc_n=6, phase=0.6, taper=0.34)
o = SPINE[222] - NOR[222] * 14.0
pelv = ribbon(o, flank_ang(222, -1, 34), 125, 46, bend=0.34, hook=0.20, sc_amp=0.13,
              sc_n=4, phase=1.0, taper=0.44)
o = SPINE[294] - NOR[294] * 10.0
anal = ribbon(o, flank_ang(294, -1, 30), 140, 50, bend=0.36, hook=0.20, sc_amp=0.13,
              sc_n=4, phase=2.1, taper=0.44)

ALL_FINS = [("lob_up", lob_up), ("lob_dn", lob_dn), ("lob_mid", lob_mid),
            ("pec1", pec1), ("pec2", pec2), ("dors", dors), ("pelv", pelv),
            ("anal", anal)]

BACK_FINS = [  # drawn behind the body -> roots hidden, reads as attached
    (lob_up, 0.72, (250, 186, 118), (255, 238, 206)),
    (lob_dn, 0.68, (248, 182, 114), (255, 234, 200)),
    (lob_mid, 0.46, (252, 196, 138), (255, 240, 210)),
    (lob_root, 0.62, (252, 200, 142), (255, 242, 214)),
    (dors, 0.52, (238, 172, 106), (255, 232, 196)),
    (anal, 0.50, (238, 172, 106), (255, 230, 194)),
    (pelv, 0.50, (244, 180, 116), (255, 234, 200)),
]
FRONT_FINS = [
    (pec1, 0.58, (255, 210, 158), (255, 244, 216)),
    (pec2, 0.52, (255, 206, 152), (255, 242, 212)),
]

# veil for back-fin rays / membrane where they cross the translucent body
BACK_RAY = 0.30


def draw_fin(scene, glow, fin, mul, col, hot, rm=1.0):
    lead, trail, ctr, cap = fin
    root = ctr[0]
    ang = math.atan2(ctr[-1][1] - ctr[0][1], ctr[-1][0] - ctr[0][0])
    length = math.hypot(ctr[-1][0] - ctr[0][0], ctr[-1][1] - ctr[0][1]) + 1e-3
    alpha = raster(lead + cap + trail[::-1], blur=1.3)
    SIL[:] = np.clip(SIL + alpha * max(mul, 0.85), 0, 1)
    uf = u_field(root, ang, length)
    # crisp near the root, dissolving towards the tip -> silk
    alpha = alpha * (1.0 - sm((uf - 0.45) / 0.55)) + soft(alpha, 5.0) * sm((uf - 0.45) / 0.55)
    ripple = 0.88 + 0.12 * np.cos(5.0 * math.pi * uf + 1.5)
    geom = 0.34 + 0.66 * (1.0 - uf) ** 0.8
    segs = []
    for k in range(-7, 8):
        v = k / 7.0
        p0 = (root[0] + math.cos(ang) * 30 - math.sin(ang) * v * 26,
              root[1] + math.sin(ang) * 30 + math.cos(ang) * v * 26)
        i = int(0.92 * (len(lead) - 1))
        cx = (lead[i][0] + trail[i][0]) / 2
        cy = (lead[i][1] + trail[i][1]) / 2
        p1 = (cx + (lead[i][0] - trail[i][0]) * v * 0.5,
              cy + (lead[i][1] - trail[i][1]) * v * 0.5)
        segs.append(([p0, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2), p1], 3))
    rays = strokes(segs, blur=1.1) * alpha * rm
    a = alpha * geom * mul * ripple
    scene = over(scene, a, col)
    edge = np.clip(alpha - soft(alpha, 2.0), 0, 1)
    scene = over(scene, edge * mul * 1.0 * rm, (255, 250, 234))
    # finer secondary veins
    segs2 = []
    for k in range(-10, 11):
        v = (k + 0.5) / 10.5
        p0 = (root[0] + math.cos(ang) * 20 - math.sin(ang) * v * 16,
              root[1] + math.sin(ang) * 20 + math.cos(ang) * v * 16)
        i = int(0.88 * (len(lead) - 1))
        cx = (lead[i][0] + trail[i][0]) / 2
        cy = (lead[i][1] + trail[i][1]) / 2
        p1 = (cx + (lead[i][0] - trail[i][0]) * v * 0.5,
              cy + (lead[i][1] - trail[i][1]) * v * 0.5)
        segs2.append(([p0, p1], 1))
    rays2 = strokes(segs2, blur=1.2) * alpha * rm
    mixed = np.clip(a * 0.50 + rays * 0.75 + edge * 1.0, 0, 1)
    scene = over(scene, mixed * 0.55 * mul * rm, hot)
    scene = over(scene, rays2 * 0.34 * mul * geom, (255, 250, 230))
    # contact shadow so the membrane separates from the body it crosses
    sep = np.clip(soft(alpha, 5.0) - alpha, 0, 1) * body
    scene = over(scene, sep * 0.30, (32, 46, 74))
    glow += mixed * 0.5 + edge * 0.35
    return scene, glow


# ---------------------------------------------------------------- water
top = sm((420.0 - YG) / 660.0)
wat = np.zeros((H, W, 3), np.float32)
wat[..., 0] = 0.085 + 0.05 * top
wat[..., 1] = 0.20 + 0.16 * top
wat[..., 2] = 0.38 + 0.28 * top
wat *= 0.42
# caustic ripples (soft wavy bands)
cau = (np.sin(XG * 0.0043 + YG * 0.0014) * np.sin(XG * 0.0011 - YG * 0.0031)
       + 0.65 * np.sin((XG + YG) * 0.0021) + 0.4 * np.sin(XG * 0.0009 - YG * 0.0052))
cau = np.clip(cau * 0.42 + 0.5, 0, 1) ** 2.6
wat += cau[:, :, None] * top[:, :, None] * np.array([0.13, 0.25, 0.40], np.float32)
# soft light pool top-left (balance for the bright fish top-right)
pool = np.exp(-(((XG - 230.0) / 330.0) ** 2 + ((YG - 190.0) / 300.0) ** 2))
wat += soft(pool, 60)[:, :, None] * np.array([0.05, 0.10, 0.17], np.float32)
# underwater light shafts (tilted, soft)
sh = np.sin((XG + YG * 0.55) * 0.0052) * 0.5 + 0.5
sh = np.clip(sh, 0, 1) ** 3.0
sh = np.asarray(Image.fromarray((sh * 255).astype(np.uint8))
                .filter(ImageFilter.GaussianBlur(26)), np.float32) / 255.0
wat += sh[:, :, None] * (top ** 1.3)[:, :, None] * np.array([0.045, 0.095, 0.17], np.float32)
vig = np.clip(1.10 - np.hypot(XG - W * 0.46, YG - H * 0.48) / (W * 0.84), 0, 1) ** 0.75
wat *= vig[:, :, None]

scene = wat.copy()
glow = np.zeros((H, W), np.float32)
SIL = np.zeros((H, W), np.float32)
for fin, mul, col, hot in BACK_FINS:
    scene, glow = draw_fin(scene, glow, fin, mul, col, hot, BACK_RAY)

# ---------------------------------------------------------------- body
p = np.clip(R / np.maximum(BW, 1e-3), -1.5, 1.5)
SIL[:] = np.clip(SIL + body, 0, 1)
# light bleeding into the water around the fish
scene = over(scene, soft(body, 26) * 0.55, (255, 176, 92))
scene = over(scene, soft(body, 9) * 0.45, (255, 196, 120))
# translucent base wash so the water reads through the body
scene = over(scene, body * 0.72, (198, 116, 56))
shade = np.clip(1.0 - np.abs(p - 0.06) / 1.00, 0, 1) ** 0.78
# soft, warped scale sheen: wide and uneven so the flanks never read as ribs
warp = 1.7 * np.sin(US * 6.3 + 0.7) + 0.9 * np.sin(US * 11.0 - p * 1.3)
scales = 0.5 + 0.5 * np.cos(US * 2 * math.pi * 13.0 + p * 2.1 + warp)
scales = soft(scales, 2.5)
env = 0.55 + 0.45 * np.sin(US * 2 * math.pi * 1.6 + p * 1.2) ** 2
scene = over(scene, body * shade * (0.47 + 0.05 * (scales - 0.5) * 2.0 * env), (255, 186, 108))
# scale glints
scene = over(scene, body * (np.clip(scales, 0, 1) ** 6) * 0.06 * env, (255, 244, 216))
core = body * np.clip(1.0 - np.abs(p + 0.02) / 0.36, 0, 1) ** 1.15
scene = over(scene, core * 0.62, (255, 234, 190))
# koi pattern: patches of light along the body (kohaku-like)
pat = np.zeros((H, W), np.float32)
for u0, p0, ru, rp, amp in [(0.17, 0.10, 0.11, 0.60, 1.0), (0.34, -0.18, 0.08, 0.50, 0.9),
                            (0.52, 0.22, 0.09, 0.62, 1.0), (0.70, -0.12, 0.07, 0.48, 0.8),
                            (0.86, 0.28, 0.06, 0.40, 0.7)]:
    d = np.sqrt(((US - u0) / ru) ** 2 + ((p - p0) / rp) ** 2)
    pat = np.clip(pat + amp * sm((1.0 - d) / 0.42), 0, 1)
pat = soft(pat, 6)
scene = over(scene, pat * body * 0.42, (255, 138, 52))
back = body * sm((p - 0.68) / 0.32)
scene = over(scene, back * 0.34, (96, 48, 26))
rim = body * sm((ra - BW * 0.80) / (BW * 0.20 + 1e-3))
scene = over(scene, rim * 0.72, (255, 240, 206))
belly = body * sm((-p - 0.48) / 0.46)
scene = over(scene, belly * 0.24, (255, 216, 156))
gill = soft(np.exp(-(((US - 0.155) * 300.0) ** 2)) * body, 3.0)
scene = over(scene, gill * 0.26, (140, 74, 34))
glow += body * (0.45 * shade + 0.60 * core + 0.70 * rim + 0.35 * pat)

# tail flare: glues the caudal membrane into the peduncle
ct, st = math.cos(A_tail), math.sin(A_tail)
fx = (XG - TROOT[0]) * ct + (YG - TROOT[1]) * st
fy = (XG - TROOT[0]) * -st + (YG - TROOT[1]) * ct
fl = soft(np.exp(-(((fx - 60.0) / 118.0) ** 2 + (fy / 72.0) ** 2)), 10.0)
scene = over(scene, fl * 0.40, (255, 196, 128))
glow += fl * 0.55

# ---------------------------------------------------------------- eye, mouth, barbels
A_head = math.atan2(TAN[22][1], TAN[22][0])


def spot(cx, cy, rx, ry, sk, ang=0.0):
    dx, dy = XG - cx, YG - cy
    c, s = math.cos(ang), math.sin(ang)
    return sm((1.0 - np.hypot((dx * c + dy * s) / rx, (-dx * s + dy * c) / ry)) / sk)


eye_c = SPINE[24] + NOR[24] * 27.0 - TAN[24] * 8.0
socket = spot(eye_c[0], eye_c[1], 40.0, 32.0, 0.62, A_head)
lid = spot(eye_c[0], eye_c[1], 28.0, 23.0, 0.50, A_head)
iris = spot(eye_c[0], eye_c[1], 21.0, 17.0, 0.44, A_head)
pupil = spot(eye_c[0] + 2.0, eye_c[1] + 1.2, 9.0, 11.5, 0.72, A_head)
glint = spot(eye_c[0] - 8.0, eye_c[1] - 8.0, 5.6, 3.6, 1.05, A_head - 0.4)
glint2 = spot(eye_c[0] + 8.4, eye_c[1] + 5.2, 3.0, 2.0, 1.05, A_head - 0.4)
iris_ring = np.clip(iris - soft(iris, 3.0), 0, 1)
far_eye = spot(*(SPINE[22] - NOR[22] * 22.0), 12.0, 10.0, 0.70, A_head)

m0 = SPINE[4] + TAN[4] * 34.0 - NOR[4] * 14.0
scene = over(scene, spot(m0[0], m0[1], 18.0, 6.0, 0.55, A_head) * 0.55, (92, 42, 18))
bs = []
for k, (dv, ln, side) in enumerate([(-8.0, 150.0, -1), (9.0, 120.0, 1)]):
    pts = []
    for i in range(18):
        t = i / 17.0
        px = m0[0] - math.cos(A_head) * ln * t + math.cos(A_head + 1.4) * 30 * t ** 2 * side
        py = m0[1] - math.sin(A_head) * ln * t + math.sin(A_head + 1.4) * 30 * t ** 2 * side + dv * t
        pts.append((px, py))
    bs.append((pts, 3))
barb = strokes(bs, blur=1.4)
SIL[:] = np.clip(SIL + barb * 0.9, 0, 1)
scene = over(scene, barb * 0.55, (255, 226, 176))
glow += barb * 0.5

for fin, mul, col, hot in FRONT_FINS:
    scene, glow = draw_fin(scene, glow, fin, mul, col, hot)

# ---------------------------------------------------------------- trails, bubbles
trails = np.zeros((H, W), np.float32)
paths = [
    [(246, 736), (168, 820), (92, 900), (18, 976)],
    [(340, 640), (196, 706), (60, 726), (-30, 700)],
    [(600, 250), (498, 142), (390, 42), (300, -40)],
    [(772, 388), (900, 456), (1016, 542)],
    [(658, 158), (700, 50), (694, -40)],
]
for path in paths:
    pth = catmull(path, 150)
    for i in range(len(pth)):
        t = i / len(pth)
        r = 2.6 + 2.6 * (1 - t)
        d = np.hypot(XG - pth[i, 0], YG - pth[i, 1])
        trails += np.exp(-(d / r) ** 2) * (0.11 * (1 - t) ** 1.4 + 0.006)
trails = np.asarray(Image.fromarray((np.clip(trails, 0, 1) * 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(3)), np.float32) / 255.0
scene = over(scene, np.clip(trails * 1.5, 0, 1) * 0.72, (140, 208, 255))

rng = np.random.default_rng(5)
bubbles = np.zeros((H, W), np.float32)
for _ in range(9):
    bx, by = rng.uniform(150, 760), rng.uniform(140, 840)
    br = rng.uniform(4, 11)
    d = np.hypot(XG - bx, YG - by)
    bubbles += np.exp(-(((d - br) / (br * 0.28)) ** 2)) * rng.uniform(0.16, 0.38)
scene = over(scene, np.clip(bubbles, 0, 1), (186, 228, 255))

motes = np.zeros((H, W), np.float32)
for _ in range(14):
    mx, my = rng.uniform(0, W), rng.uniform(0, 460)
    mr = rng.uniform(2.0, 6.0)
    motes += np.exp(-(np.hypot(XG - mx, YG - my) / mr) ** 2) * rng.uniform(0.10, 0.30)
motes = np.asarray(Image.fromarray((np.clip(motes, 0, 1) * 255).astype(np.uint8))
                   .filter(ImageFilter.GaussianBlur(1.6)), np.float32) / 255.0
scene = over(scene, np.clip(motes, 0, 1) * 0.35, (200, 232, 255))

# ---------------------------------------------------------------- bloom
gm = np.clip(glow + trails * 0.5 + sh * 0.25, 0, 1)
gimg = Image.fromarray((gm * 255).astype(np.uint8))
bl1 = np.asarray(gimg.filter(ImageFilter.GaussianBlur(13)), np.float32) / 255.0
bl2 = np.asarray(gimg.filter(ImageFilter.GaussianBlur(72)), np.float32) / 255.0
halo = (0.50 * bl1 + 0.45 * bl2)[:, :, None] * np.array([1.00, 0.74, 0.46], np.float32)
scene = 1.0 - (1.0 - np.clip(scene, 0, 1)) * (1.0 - np.clip(halo, 0, 1))

# eye last, on top of the bloom so it always reads
scene = over(scene, soft(far_eye, 2.0) * 0.28, (255, 246, 226))
scene = over(scene, soft(socket, 5.0) * 0.50, (96, 50, 24))
scene = over(scene, soft(lid, 2.4) * 0.85, (250, 196, 120))
scene = over(scene, soft(iris, 2.0) * 0.92, (214, 128, 36))
scene = over(scene, soft(pupil, 2.4) * 0.94, (14, 8, 5))
scene = over(scene, iris_ring * 0.45, (255, 222, 160))
scene = over(scene, glint, (255, 255, 255))
scene = over(scene, soft(glint2, 1.2) * 0.85, (255, 250, 236))

if os.environ.get("KOI_SILHOUETTE"):
    s = np.clip(SIL, 0, 1)
    out = np.stack([s, s, s], -1) * 0.92 + 0.06
    simg = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    simg.save(OUT, "PNG", optimize=True)
    simg.resize((420, 420), Image.Resampling.LANCZOS).save(
        OUT.rsplit(".", 1)[0] + "_s.png", "PNG", optimize=True)
    ys, xs = np.nonzero(s > 0.5)
    print("spine head %s tail %s len %.0f" % (SPINE[0].round(0), SPINE[-1].round(0),
                                              slen[-1]))
    for nm, (lead, trail, ctr, cap) in ALL_FINS:
        r = np.asarray(ctr[0], float)
        t = np.asarray(ctr[-1], float)
        m = len(ctr) // 2
        w = float(np.hypot(lead[m][0] - trail[m][0], lead[m][1] - trail[m][1]))
        print("  %-7s root (%.0f,%.0f) tip (%.0f,%.0f) w %.0f" % (nm, r[0], r[1], t[0], t[1], w))
    if len(xs):
        print("sil -> %s bbox x[%d,%d] y[%d,%d] fill %.2f bodyL %.0f maxH %.0f pedH %.0f"
              % (OUT, xs.min(), xs.max(), ys.min(), ys.max(), np.mean(s > 0.5),
                 slen[-1], 2 * max(PROFW), 2 * PROFW[-1]))
    raise SystemExit(0)

final = (np.clip(scene, 0, 1) * 255.0).round().astype(np.uint8)
img = Image.fromarray(final).convert("RGB")
img.save(OUT, "PNG", optimize=True, compress_level=9)
img.resize((300, 300), Image.Resampling.LANCZOS).save(
    OUT.rsplit(".", 1)[0] + "_300.png", "PNG", optimize=True)
print("%s %s %s %.0f KB" % (OUT, img.size, img.mode, os.stat(OUT).st_size / 1024))
