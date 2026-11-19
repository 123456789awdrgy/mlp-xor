# -*- coding: utf-8 -*-
"""把训练 JSON 画成四张图（Pillow，无 matplotlib）。"""
import json
from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


data = json.load(open("mlp_data.json", encoding="utf-8"))
hist = data["history"]
snaps = data["snapshots"]
track = data["weights_track"]
samples = data["meta"]["samples"]

INK = (30, 40, 55)
MUTED = (110, 120, 135)
GRID = (228, 232, 240)
ACC = (55, 95, 165)
RED = (198, 62, 62)


def dashed(d, pts, fill, width=3, dash=12, gap=8):
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist == 0:
            continue
        t = 0.0
        while t < dist:
            t2 = min(t + dash, dist)
            d.line([(a[0] + dx * t / dist, a[1] + dy * t / dist),
                    (a[0] + dx * t2 / dist, a[1] + dy * t2 / dist)], fill=fill, width=width)
            t = t2 + gap


# ---------- 图1：loss 曲线 ----------
W1, H1 = 1100, 620
im = Image.new("RGB", (W1, H1), (255, 255, 255))
d = ImageDraw.Draw(im)
d.text((40, 26), "训练损失随 epoch 变化", font=f(26, True), fill=INK)
L, T, R, B = 110, 100, 980, 540
d.rectangle([L, T, R, B], outline=(200, 206, 216), width=2)
losses = [h["loss"] for h in hist]
n = len(losses)
ymax = max(losses)
for i in range(1, 10):
    y = T + (B - T) * i / 10
    d.line([(L, y), (R, y)], fill=GRID, width=1)
    d.text((L - 56, y - 8), f"{ymax * (10 - i) / 10:.2f}", font=f(14), fill=MUTED)
pts = []
for i, v in enumerate(losses):
    x = L + (R - L) * i / (n - 1)
    y = B - (B - T) * v / ymax
    pts.append((x, y))
d.line(pts, fill=ACC, width=3)
for ep, dx in ((50, 10), (200, 10), (1000, 10), (4000, -120)):
    v = hist[ep]["loss"]
    x = L + (R - L) * ep / (n - 1)
    y = B - (B - T) * v / ymax
    d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=RED)
    d.text((x + dx, y - 28), f"e{ep}: {v:.3f}", font=f(13), fill=RED)
for ep in (0, 1000, 2000, 3000, 4000):
    x = L + (R - L) * ep / (n - 1)
    d.text((x - 12, B + 10), str(ep), font=f(14), fill=MUTED)
d.text((L, B + 36), "epoch", font=f(15), fill=MUTED)
im.save("fig_loss.png")


# ---------- 图2：决策边界演化 ----------
def render_panel(d, snap, left, top, size):
    grid = snap["grid"]
    n = len(grid)
    cell = size / n
    for i in range(n):
        for j in range(n):
            p = grid[i][j]
            if p < 0.5:
                t = p / 0.5
                col = (int(60 + 195 * t), int(100 + 155 * t), int(170 + 85 * t))
            else:
                t = (p - 0.5) / 0.5
                col = (255, int(255 - 185 * t), int(255 - 185 * t))
            d.rectangle([left + j * cell, top + (n - 1 - i) * cell,
                         left + (j + 1) * cell, top + (n - i) * cell], fill=col)
    for s in samples:
        px = left + (s["x1"] + 0.5) / 2.0 * size
        py = top + (1 - (s["x2"] + 0.5) / 2.0) * size
        color = (198, 45, 45) if s["target"] == 1 else (35, 60, 125)
        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=color, outline=(255, 255, 255), width=2)
        d.text((px + 14, py - 9), f"{s['x1']},{s['x2']}→{s['target']}", font=f(13), fill=(25, 30, 40))
    d.rectangle([left, top, left + size, top + size], outline=(180, 186, 196), width=2)
    label = f"epoch {snap['epoch']}   loss {snap['loss']:.3f}"
    tw = d.textlength(label, font=f(18, True))
    d.rectangle([left + 8, top + 8, left + 24 + tw, top + 38], fill=(255, 255, 255))
    d.text((left + 16, top + 12), label, font=f(18, True), fill=INK)


PANEL = 470
im = Image.new("RGB", (2 * PANEL + 180, 2 * PANEL + 190), (255, 255, 255))
d = ImageDraw.Draw(im)
d.text((40, 26), "决策边界随训练变化（蓝=判 0，红=判 1）", font=f(26, True), fill=INK)
for (left, top), snap in zip([(60, 90), (610, 90), (60, 620), (610, 620)], snaps[:4]):
    render_panel(d, snap, left, top, PANEL)
im.save("fig_boundary.png")


# ---------- 图3：参数轨迹 ----------
im = Image.new("RGB", (1100, 620), (255, 255, 255))
d = ImageDraw.Draw(im)
d.text((40, 26), "参数随训练的变化（虚线为对称的那个，两两重合）", font=f(24, True), fill=INK)
L, T, R, B = 110, 100, 780, 540
d.rectangle([L, T, R, B], outline=(200, 206, 216), width=2)
epochs = [t["epoch"] for t in track]
allv = [v for t in track for v in t["W"]] + [v for t in track for v in t["v"]]
vmin, vmax = min(allv), max(allv)
pad = 0.6
for i in range(1, 10):
    y = T + (B - T) * i / 10
    d.line([(L, y), (R, y)], fill=GRID, width=1)
    d.text((L - 56, y - 8), f"{vmax - (vmax - vmin + 2 * pad) * i / 10:.1f}", font=f(13), fill=MUTED)


def ypos(v):
    return B - (B - T) * ((v - (vmin - pad)) / (vmax - vmin + 2 * pad))


series = [
    ("W11", 0, (198, 62, 62), False), ("W22", 3, (198, 62, 62), True),
    ("W12", 1, (216, 148, 40), False), ("W21", 2, (216, 148, 40), True),
    ("v1", 0, (120, 85, 165), False), ("v2", 1, (120, 85, 165), True),
]
for name, idx, color, dash in series:
    pts = []
    for t in track:
        val = t["W"][idx] if name.startswith("W") else t["v"][idx]
        pts.append((L + (R - L) * t["epoch"] / epochs[-1], ypos(val)))
    if dash:
        dashed(d, pts, color)
    else:
        d.line(pts, fill=color, width=3)

legend = [("W11 与 W22（重合）", (198, 62, 62)), ("W12 与 W21（重合）", (216, 148, 40)),
          ("v1 与 v2（重合）", (120, 85, 165))]
for i, (name, color) in enumerate(legend):
    y0 = 130 + i * 34
    d.line([(820, y0 + 9), (860, y0 + 9)], fill=color, width=4)
    d.text((868, y0), name, font=f(15), fill=INK)
d.text((820, 260), "初始化时把两个隐藏", font=f(14), fill=MUTED)
d.text((820, 282), "单元设成了对称的，", font=f(14), fill=MUTED)
d.text((820, 304), "梯度也对称，所以", font=f(14), fill=MUTED)
d.text((820, 326), "两条线始终重叠。", font=f(14), fill=MUTED)
d.text((L, B + 36), "epoch", font=f(15), fill=MUTED)
im.save("fig_weights.png")


# ---------- 图4：隐藏层输出 ----------
im = Image.new("RGB", (1100, 620), (255, 255, 255))
d = ImageDraw.Draw(im)
d.text((40, 26), "四个样本在隐藏层的激活值（训练结束）", font=f(26, True), fill=INK)
L, T, R, B = 110, 100, 1000, 520
d.line([(L, B), (R, B)], fill=(190, 196, 206), width=2)
hidden = data["final"]["hidden"]
probs = data["final"]["probs"]
group_w = (R - L) / len(samples)
bar_w = 56
for i, (s, h, p) in enumerate(zip(samples, hidden, probs)):
    gx = L + group_w * i + group_w / 2
    for j, val in enumerate(h):
        x = gx + (j - 0.5) * (bar_w + 14) - bar_w / 2
        hgt = (B - T - 70) * val
        color = (55, 95, 165) if j == 0 else (198, 90, 90)
        d.rectangle([x, B - hgt, x + bar_w, B], fill=color)
        d.text((x + 6, B - hgt - 24), f"{val:.2f}", font=f(14), fill=INK)
    d.text((gx - 44, B + 12), f"({s['x1']}, {s['x2']})", font=f(16, True), fill=INK)
    d.text((gx - 44, B + 38), f"标签 {s['target']} · 预测 {data['final']['preds'][i]} · p={p:.3f}",
           font=f(13), fill=MUTED)
for i, (name, color) in enumerate([("h1", (55, 95, 165)), ("h2", (198, 90, 90))]):
    x0, y0 = 830 + i * 100, 120
    d.rectangle([x0, y0, x0 + 18, y0 + 18], fill=color)
    d.text((x0 + 26, y0 - 2), name, font=f(16), fill=INK)
im.save("fig_hidden.png")

print("figures done")
