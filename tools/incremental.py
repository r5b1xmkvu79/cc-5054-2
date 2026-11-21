"""Incremental cut analysis with a local-replacement optimization.

Dependency analysis
-------------------
Detection reads only per-frame 8-bin-per-channel rgb histograms; every later
value is a deterministic function of that histogram sequence and fixed
parameters:

  d[t]     = 1 - sum(min(hist[t-1], hist[t]))         pair (t-1, t)
  base[t]  = median(d[t-WIN .. t-1] + d[t+1 .. t+WIN])
  score[t] = d[t] / max(base[t], SCORE_FLOOR)
  mask[t]  = d[t] > THRESH_D and score[t] > THRESH_SCORE
  runs     = maximal mask runs, merged across gaps <= MERGE_GAP
  cands    = classify(runs) using hist[s-1], hist[e]

An edit therefore propagates exactly as far as it moves that chain:

  * d changes only on pairs touching a frame whose histogram changed;
  * base changes only within WIN of a changed d;
  * the mask changes only where d or base changed;
  * runs/candidates are cheap derivations over the global mask.

Cache-hit policy
----------------
`Analysis` retains the frames it analysed and their histograms. On update:

  level 1  vectorised byte comparison against the retained frames. Frames
           whose bytes are unchanged keep their histogram verbatim - this is
           only a fast path, since bytes are not what detection reads.
  level 2  frames whose bytes changed get a fresh histogram. If it equals the
           old histogram the frame is content-equivalent (detection cannot
           tell the difference) and its statistics are still reused. Only a
           genuinely changed histogram dirties the dependency chain.

So a rename (same bytes, different label) is a level-1 hit and reuses
everything; a reorder moves frames to new positions, misses level 1, and - if
the histograms genuinely differ - must recompute. Reuse is never keyed on file
identity or on the eventual output.

Stopping condition
------------------
A dirty set of pair indices is expanded by WIN (the baseline window) to form
recompute windows. This expansion is a fixed point: every base a window
recomputes reads d only inside the window, and the mask outside the windows is
asserted unchanged. If the dirty region covers most of the stream, or the
outside-mask assertion fails, we stop trying to be local and do one full pass.
`analyze_with_reuse` additionally decodes both paths and asserts bit equality,
so the optimization is verified rather than assumed.
"""
import numpy as np

import cuts
from common import H, N, W

PARAMS = {
    "bins": cuts.BINS, "win": cuts.WIN,
    "thresh_d": cuts.THRESH_D, "thresh_score": cuts.THRESH_SCORE,
    "score_floor": cuts.SCORE_FLOOR, "merge_gap": cuts.MERGE_GAP,
    "w": W, "h": H, "n": N,
}

# base at t reads d over WIN each side. A changed histogram moves d on its own
# pair and the next pair, so it reaches WIN frames beyond that pair: the hard
# propagation distance of one changed histogram is WIN + 1 frames.
BASE_REACH = cuts.WIN
PROPAGATION_BOUND = BASE_REACH + 1
# when the dirty analysis region exceeds this share of the stream, one full
# pass is cheaper and simpler to justify than many stitched windows.
FULL_FALLBACK_SHARE = 0.65


class Analysis:
    def __init__(self, frames, counts, hist, d, base, mask, score, runs, cands,
                 stats, windows, reasons, recomputed):
        self.frames = frames          # retained HxWx3 uint8 frames (N, H, W, 3)
        self.counts = counts          # (N, 512) int histograms
        self.hist = hist              # normalized histograms
        self.d = d
        self.base = base
        self.mask = mask
        self.score = score
        self.runs = runs
        self.cands = cands
        self.stats = stats
        self.windows = windows        # list of [lo, hi] recomputed analysis spans
        self.reasons = reasons
        self.recomputed = recomputed  # set of frame ids whose d/base were redone

    def signature(self):
        """Ordered content+parameter fingerprint of what detection reads."""
        return signature(self.counts)


def signature(counts):
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(sorted(PARAMS.items())).encode())
    h.update(np.ascontiguousarray(counts, dtype=np.int64).tobytes())
    return h.hexdigest()


def _derive(counts):
    hist = cuts.normalize_counts(counts)
    d = cuts.hist_dist(hist)
    base = cuts.local_baseline(d)
    score, mask = cuts.threshold(d, base)
    runs = cuts.build_runs(mask)
    cands = cuts.classify_runs(runs, d, hist, score)
    stats = {"median_hdist": round(float(np.median(d[1:])), 4)}
    return hist, d, base, score, mask, runs, cands, stats


def full_analysis(frames):
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    counts = np.stack([cuts.frame_counts(f) for f in frames])
    hist, d, base, score, mask, runs, cands, stats = _derive(counts)
    return Analysis(frames, counts, hist, d, base, mask, score, runs, cands,
                    stats, windows=[[0, N - 1]],
                    reasons=[{"range": [0, N - 1], "why": "full analysis"}],
                    recomputed=set(range(N)))


def _pair_indices(changed):
    """d-indices whose pair (t-1, t) touches a frame with a changed histogram."""
    pairs = set()
    for f in changed:
        if 1 <= f < N:
            pairs.add(f)          # d[f]   uses hist[f-1], hist[f]
        if 1 <= f + 1 < N:
            pairs.add(f + 1)      # d[f+1] uses hist[f], hist[f+1]
    return sorted(pairs)


def _windows(d_changed):
    """Bounding spans of the frames whose base must be recomputed."""
    if not d_changed:
        return []
    spans = [[max(0, u - BASE_REACH), min(N - 1, u + BASE_REACH)]
             for u in d_changed]
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _full_fallback(frames, why, windows=None):
    full = full_analysis(frames)
    reasons = [{"range": [0, N - 1], "why": why}]
    if windows:
        reasons.append({"range": [windows[0][0], windows[-1][1]],
                        "why": "dirty windows that triggered the fallback"})
    meta = {"mode": "full", "windows": [[0, N - 1]], "analysis_frames": N,
            "recomputed": N, "histograms": N, "reused": 0, "savings": 0.0,
            "reasons": reasons, "fallback": why}
    return full, meta


def analyze_incremental(old, new_frames):
    """Recompute only what a local edit can affect. Returns (Analysis, meta).

    meta records: mode, windows, analysis_frames (d/base redone), histograms
    (per-frame histograms computed), reused, savings, and human reasons.
    """
    new_frames = np.ascontiguousarray(new_frames, dtype=np.uint8)

    # level 1: vectorised byte diff against the retained frames. Only frames
    # that differ need a histogram; identical bytes prove identical histogram.
    byte_diff = np.any(new_frames != old.frames, axis=(1, 2, 3))
    counts = old.counts.copy()
    histograms = 0
    changed = []
    content_equivalent = []
    byte_identical = [int(t) for t in np.nonzero(~byte_diff)[0]]
    for t in np.nonzero(byte_diff)[0]:
        c = cuts.frame_counts(new_frames[t])
        histograms += 1
        if np.array_equal(c, old.counts[t]):
            content_equivalent.append(int(t))   # same detector input, new bytes
        else:
            counts[t] = c
            changed.append(int(t))

    if not changed:
        # Every frame's histogram - the only detector input - is unchanged.
        if histograms == 0:
            mode = "reuse-all"
            why = ("all frames byte-identical to the previous edition "
                   "(level-1 hit); histograms, baseline context and "
                   "parameters unchanged")
        else:
            mode = "reuse-content"
            why = (f"{len(content_equivalent)} frame(s) changed bytes but kept "
                   "an identical rgb histogram (level-2 content-equivalent); "
                   "detection reads only the histogram, so every statistic is "
                   "reused")
        reused = Analysis(new_frames, old.counts, old.hist, old.d, old.base,
                          old.mask, old.score, old.runs, old.cands, old.stats,
                          windows=[[0, N - 1]],
                          reasons=[{"range": [0, N - 1], "why": why}],
                          recomputed=set())
        meta = {"mode": mode, "windows": [], "analysis_frames": 0,
                "histograms": histograms, "reused": N, "savings": 1.0,
                "reasons": [{"range": [0, N - 1], "why": why}],
                "byte_identical_frames": len(byte_identical),
                "content_equivalent_frames": content_equivalent}
        return reused, meta

    d_changed = _pair_indices(changed)
    windows = _windows(d_changed)
    analysis_frames = sum(hi - lo + 1 for lo, hi in windows)
    if not windows or analysis_frames > FULL_FALLBACK_SHARE * N:
        return _full_fallback(
            new_frames,
            "dirty analysis region covers most of the stream; one full pass is "
            "cheaper to justify than many stitched windows", windows)

    hist = old.hist.copy()
    hist[changed] = cuts.normalize_counts(counts[changed])
    d_new = old.d.copy()
    for t in d_changed:
        d_new[t] = cuts.dist_at(hist, t)

    # nothing outside the touched pairs may move
    other = np.ones(N, bool)
    other[0] = False
    other[d_changed] = False
    if not np.array_equal(d_new[other], old.d[other]):
        return _full_fallback(new_frames,
                              "dependency closure escaped the recompute window",
                              windows)

    base_new = old.base.copy()
    for lo, hi in windows:
        for t in range(lo, hi + 1):
            base_new[t] = cuts.baseline_at(d_new, t)

    score = d_new / np.maximum(base_new, cuts.SCORE_FLOOR)
    mask = (d_new > cuts.THRESH_D) & (score > cuts.THRESH_SCORE)

    outside = np.ones(N, bool)
    outside[0] = False
    for lo, hi in windows:
        outside[lo:hi + 1] = False
    if not np.array_equal(mask[outside], old.mask[outside]):
        return _full_fallback(
            new_frames,
            "mask changed outside the window; influence not provably bounded",
            windows)

    runs = cuts.build_runs(mask)
    cands = cuts.classify_runs(runs, d_new, hist, score)
    stats = {"median_hdist": round(float(np.median(d_new[1:])), 4)}

    reasons = [{"range": [min(changed), max(changed)],
                "why": f"{len(changed)} frame(s) with a changed rgb histogram"}]
    for lo, hi in windows:
        reasons.append({"range": [lo, hi],
                        "why": f"d/base reach of the edit (+/-{BASE_REACH}-frame "
                               "baseline window)"})
    analysis = Analysis(new_frames, counts, hist, d_new, base_new, mask, score,
                        runs, cands, stats, windows=windows, reasons=reasons,
                        recomputed={t for lo, hi in windows
                                    for t in range(lo, hi + 1)})
    meta = {"mode": "incremental", "windows": windows,
            "analysis_frames": analysis_frames,
            "recomputed": analysis_frames, "histograms": histograms,
            "reused": N - analysis_frames,
            "savings": round((N - analysis_frames) / N, 4),
            "reasons": reasons, "changed_frames": changed,
            "byte_identical_frames": len(byte_identical),
            "content_equivalent_frames": content_equivalent}
    return analysis, meta


def analyze_with_reuse(old, new_frames):
    """Incremental analysis + bit-exact proof against a full recompute."""
    got, meta = analyze_incremental(old, new_frames)
    ref = full_analysis(new_frames)
    assert got.runs == ref.runs, ("runs differ", got.runs, ref.runs)
    assert got.cands == ref.cands, ("candidates differ", got.cands, ref.cands)
    assert got.stats == ref.stats, ("stats differ", got.stats, ref.stats)
    assert np.array_equal(got.counts, ref.counts), "histograms differ"
    assert np.array_equal(got.d, ref.d), "d differs from full recompute"
    assert np.array_equal(got.base, ref.base), "base differs from full recompute"
    assert np.array_equal(got.score, ref.score), "score differs from full recompute"
    assert np.array_equal(got.mask, ref.mask), "mask differs from full recompute"
    assert got.signature() == ref.signature()
    return got, meta, ref
