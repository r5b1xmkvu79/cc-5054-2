"""Generate the synthetic archive footage: hard cuts, two similar dissolves,
one single-frame flash. Writes out/frames.rgb (rgb24) and out/manifest.json."""
import os

import numpy as np

from common import FPS, H, N, OUT, W, save_json

# Shot layout (frame indices)
S1 = (0, 40)      # "city": blue gradient, red circle, static buildings
S2 = (40, 80)     # "field": green gradient, scrolling bars, sun
DA = (80, 88)     # dissolve A: field -> night, 8 frames (fast)
S3 = (88, 140)    # "night": purple gradient, moving square, stars
FLASH = 120       # single-frame white flash inside shot 3
DB = (140, 156)   # dissolve B: night -> lagoon, 16 frames (gentle)
S4 = (156, 200)   # "lagoon": teal gradient, ellipse, scan line
S5 = (200, 240)   # "lava": orange gradient, bouncing ball

rng = np.random.default_rng(7)
STARS = [(int(rng.integers(0, W)), int(rng.integers(0, H // 2))) for _ in range(40)]

ys = np.arange(H, dtype=np.float32)[:, None]
xs = np.arange(W, dtype=np.float32)[None, :]


def gradient(top, bottom):
    t = (ys / (H - 1)).astype(np.float32)
    return (np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
            )[:, None, :].repeat(W, axis=1)


def disc(img, cx, cy, r, color):
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    img[m] = color


def rect(img, x0, y0, x1, y1, color):
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = color


def shot1(t):
    img = gradient((12, 18, 52), (36, 66, 150))
    for i, bw in enumerate((36, 28, 44)):
        rect(img, 30 + i * 95, 110 - 14 * i, 30 + i * 95 + bw, H, (8, 10, 26))
    disc(img, 24 + (t * 3.1) % (W + 60) - 30, 62, 17, (220, 40, 40))
    return img


def shot2(t):
    img = gradient((24, 92, 34), (128, 205, 96))
    disc(img, 262, 38, 20, (250, 220, 60))
    for i in range(2):
        x = int((t * 5 + i * 150) % (W + 40)) - 20
        rect(img, x, 0, x + 14, H, (240, 240, 240))
    return img


def shot3(t):
    img = gradient((44, 12, 74), (16, 8, 40))
    for sx, sy in STARS:
        img[sy, sx] = (230, 230, 240)
    s = 26
    x = int(40 + (t * 2.2) % (W - 2 * s - 60))
    y = int(30 + (t * 1.3) % (H - 2 * s - 40))
    rect(img, x, y, x + s, y + s, (40, 200, 220))
    return img


def shot4(t):
    img = gradient((8, 48, 62), (34, 96, 104))
    y = int((t * 7) % H)
    rect(img, 0, y, W, min(H, y + 3), (120, 150, 152))
    disc(img, 60 + (t * 2.6) % (W - 120), 120, 24, (120, 46, 110))
    return img


def shot5(t):
    img = gradient((130, 42, 12), (225, 125, 32))
    cx = 40 + (t * 4.0) % (W - 80)
    cy = 40 + abs(((t * 5.0) % 160) - 80)
    disc(img, int(cx), int(cy), 15, (40, 20, 60))
    return img


def frame_at(n):
    if S1[0] <= n < S1[1]:
        return shot1(n)
    if S2[0] <= n < S2[1]:
        return shot2(n)
    if DA[0] <= n < DA[1]:
        a = (n - DA[0] + 1) / (DA[1] - DA[0] + 1)
        return shot2(n) * (1 - a) + shot3(n) * a
    if S3[0] <= n < S3[1]:
        if n == FLASH:
            return np.full((H, W, 3), 255, np.float32)
        return shot3(n)
    if DB[0] <= n < DB[1]:
        a = (n - DB[0] + 1) / (DB[1] - DB[0] + 1)
        return shot3(n) * (1 - a) + shot4(n) * a
    if S4[0] <= n < S4[1]:
        return shot4(n)
    return shot5(n)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = np.empty((N, H, W, 3), np.uint8)
    for n in range(N):
        raw[n] = np.clip(frame_at(n), 0, 255).astype(np.uint8)
    path = os.path.join(OUT, "frames.rgb")
    raw.tofile(path)

    save_json({
        "w": W, "h": H, "fps": FPS, "n": N,
        "shots": [
            {"name": "city", "start": S1[0], "end": S1[1]},
            {"name": "field", "start": S2[0], "end": S2[1]},
            {"name": "night", "start": S3[0], "end": S3[1]},
            {"name": "lagoon", "start": S4[0], "end": S4[1]},
            {"name": "lava", "start": S5[0], "end": S5[1]},
        ],
        "transitions": [
            {"kind": "cut", "at": S2[0]},
            {"kind": "dissolve", "name": "A", "start": DA[0], "end": DA[1]},
            {"kind": "flash", "at": FLASH},
            {"kind": "dissolve", "name": "B", "start": DB[0], "end": DB[1]},
            {"kind": "cut", "at": S5[0]},
        ],
    }, os.path.join(OUT, "manifest.json"))
    print(f"frames.rgb: {N} frames {W}x{H} -> {path}")


if __name__ == "__main__":
    main()
