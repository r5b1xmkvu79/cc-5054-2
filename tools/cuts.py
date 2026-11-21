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
WIN = 20
THRESH_D = 0.08
THRESH_SCORE = 8.0
SCORE_FLOOR = 0.004
MERGE_GAP = 3


def frame_counts(frame):
    """Integer rgb histogram of one HxWx3 frame."""
    f = frame.reshape(H * W, 3).astype(np.int16)
    idx = (f[:, 0] >> 5) * BINS * BINS + (f[:, 1] >> 5) * BINS + (f[:, 2] >> 5)
    return np.bincount(idx, minlength=BINS ** 3).astype(np.int64)


def hist_counts(raw):
    """Integer per-frame rgb histogram (n, BINS**3). Exact fingerprint source."""
    frames = raw.reshape(N, H, W, 3)
    return np.stack([frame_counts(frames[t]) for t in range(N)])


def normalize_counts(counts):
    return (counts.astype(np.float64) / float(H * W)).astype(np.float32)


def histograms(raw):
    return normalize_counts(hist_counts(raw))


def dist_at(hist, t):
    """Histogram-intersection distance across the pair (t-1, t)."""
    if t <= 0:
        return 0.0
    return float(1.0 - np.minimum(hist[t - 1], hist[t]).sum())


def hist_dist(hist):
    d = np.zeros(N, np.float32)
    inter = np.minimum(hist[1:], hist[:-1]).sum(axis=1)
    d[1:] = 1.0 - inter
    return d


def baseline_at(d, t, win=WIN):
    lo = d[max(1, t - win):t]
    hi = d[t + 1:t + 1 + win]
    ctx = np.concatenate([lo, hi])
    return float(np.median(ctx)) if len(ctx) else 0.0


def local_baseline(d, win=WIN):
    base = np.zeros(N, np.float32)
    for t in range(N):
        base[t] = baseline_at(d, t, win)
    return base


def threshold(d, base):
    score = d / np.maximum(base, SCORE_FLOOR)
    mask = (d > THRESH_D) & (score > THRESH_SCORE)
    return score, mask


def build_runs(mask, lo=1, hi=N):
    """Elevated d-indices grouped into runs; runs <=MERGE_GAP apart merge."""
    runs = []
    t = lo
    while t < hi:
        if mask[t]:
            s = t
            while t < hi and mask[t]:
                t += 1
            if runs and s - runs[-1][1] <= MERGE_GAP:
                runs[-1] = (runs[-1][0], t)
            else:
                runs.append((s, t))  # d-indices [s, t): frames s-1..t-1 changing
        else:
            t += 1
    return runs


def classify_runs(runs, d, hist, score):
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
    return cands


def detect(hist):
    d = hist_dist(hist)
    base = local_baseline(d)
    score, mask = threshold(d, base)
    runs = build_runs(mask)
    cands = classify_runs(runs, d, hist, score)
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


def analyze(raw):
    hist = histograms(raw)
    return detect(hist)


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
