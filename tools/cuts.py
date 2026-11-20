"""Detect candidate cut points from the raw footage and plan the prediction
structure (keyframe placement) for two arrangements:
  dense  - short positioning wait: keyframes at every candidate, tight max GOP
  sparse - controlled keyframe bitrate: cuts/flashes/sharp transitions only
Statistic: rgb-histogram intersection distance vs a local baseline, so steady
in-shot motion is not mistaken for a transition.
Writes out/plan.json."""
import os

import numpy as np

from common import H, N, OUT, W, save_json

BINS = 8  # per channel


def histograms(raw):
    f = raw.reshape(N, H * W, 3).astype(np.int16)
    idx = (f[..., 0] >> 5) * BINS * BINS + (f[..., 1] >> 5) * BINS + (f[..., 2] >> 5)
    h = np.zeros((N, BINS ** 3), np.float32)
    for t in range(N):
        h[t] = np.bincount(idx[t], minlength=BINS ** 3)
    return h / (H * W)


def hist_dist(hist):
    d = np.zeros(N, np.float32)
    inter = np.minimum(hist[1:], hist[:-1]).sum(axis=1)
    d[1:] = 1.0 - inter
    return d


def local_baseline(d, win=20):
    base = np.zeros(N, np.float32)
    for t in range(N):
        lo = d[max(1, t - win):t]
        hi = d[t + 1:t + 1 + win]
        ctx = np.concatenate([lo, hi])
        base[t] = np.median(ctx) if len(ctx) else 0.0
    return base


def detect(hist):
    d = hist_dist(hist)
    base = local_baseline(d)
    score = d / np.maximum(base, 0.004)

    # elevated frames, grouped into runs; runs separated by <=3 frames merge
    mask = (d > 0.08) & (score > 8.0)
    runs = []
    t = 1
    while t < N:
        if mask[t]:
            s = t
            while t < N and mask[t]:
                t += 1
            if runs and s - runs[-1][1] <= 3:
                runs[-1] = (runs[-1][0], t)
            else:
                runs.append((s, t))  # d-indices [s, t): frames s-1..t-1 changing
        else:
            t += 1

    cands = []
    for s, e in runs:
        n = e - s
        if n == 1:
            cands.append({"frame": s, "kind": "cut", "conf": 1.0,
                          "hdist": round(float(d[s]), 3)})
        elif n == 2:
            back = 1.0 - np.minimum(hist[s - 1], hist[e]).sum()
            if back < 0.08:
                cands.append({"frame": s, "kind": "flash", "conf": 1.0,
                              "hdist": round(float(d[s]), 3)})
            else:
                cands.append({"frame": s, "kind": "cut", "conf": 1.0,
                              "hdist": round(float(d[s]), 3)})
        else:
            strength = float(d[s:e].mean())
            kind = "dissolve_sharp" if strength >= 0.35 else "dissolve_gradual"
            cands.append({"frame": s, "kind": kind, "end": e,
                          "conf": round(float(score[s:e].mean()), 2),
                          "hdist": round(strength, 3)})

    cands.sort(key=lambda c: c["frame"])
    stats = {"median_hdist": round(float(np.median(d[1:])), 4)}
    return cands, stats


def plan(cands):
    instant = [c for c in cands
               if c["kind"] in ("cut", "flash", "dissolve_sharp")]
    dense_k = sorted({0} | {c["frame"] for c in cands})
    sparse_k = sorted({0} | {c["frame"] for c in instant})
    return {
        "dense": {"forced": dense_k, "g": 48,
                  "note": "short positioning wait: keyframe at every candidate"},
        "sparse": {"forced": sparse_k, "g": 240,
                   "note": "controlled keyframe bitrate: cuts/flashes/sharp "
                           "transitions only"},
    }


def main():
    raw = np.fromfile(os.path.join(OUT, "frames.rgb"), dtype=np.uint8)
    hist = histograms(raw)
    cands, stats = detect(hist)
    plans = plan(cands)
    save_json({"stats": stats, "candidates": cands, "variants": plans},
              os.path.join(OUT, "plan.json"))
    print(f"stats: {stats}")
    print("candidates:")
    for c in cands:
        line = f"  frame {c['frame']:3d}  {c['kind']:15s} conf={c['conf']} hdist={c['hdist']}"
        if "end" in c:
            line += f" end={c['end']}"
        print(line)
    for name, v in plans.items():
        print(f"plan {name}: forced={v['forced']} g={v['g']}")


if __name__ == "__main__":
    main()
