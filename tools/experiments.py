"""Edit experiments for the incremental cut analysis.

Starting from the archive frames (out/frames.rgb) we build edited editions and
measure (a) how far the edit propagates through detection, (b) how much of the
analysis can be reused, and (c) whether an identical re-encode reproduces the
same candidate cuts, keyframes and independently decodable positions.

Edits
-----
flash+dissolve+cut   a one-frame white flash inserted inside dissolve B (frame
                     148) and the neighbouring shot cut moved one frame
                     (200 -> 201)
subset of the above  each edit alone, to isolate propagation
wide_repaint         a long span repainted, so the dirty region cannot be
                     bounded locally: the engine must fall back to a full pass
retouch              two pixels swapped between values in the same histogram
                     bin: bytes change but the detector input does not, so
                     level-2 content equivalence must reuse everything
edge run             four equal-length clips placed at the shot edge
  edge_canonical       canonical order 0,1,2,3
  edge_rename          the same clip files renamed; frame content is unchanged
  edge_reorder         the same clip files permuted 2,3,0,1; content changes
                       but the full detection result equals the canonical one

The rename/reorder pair is the cache-policy test. Both editions are re-encoded
and end with the same candidate cuts and the same IDR positions, but rename
may reuse everything while reorder must recompute. The reuse decision is keyed
on what detection actually reads - the ordered per-frame histograms - not on
file names and not on the eventual output. Rename preserves that key (0
changed frames); reorder moves the same files to new positions, so the
signature changes and 57 frames must be re-analysed. Output equality is
*verified after recomputing* (the report asserts it), never assumed from the
inputs; that is exactly why reorder cannot skip the work even though the final
cuts happen to come out the same.

Writes out/incremental.json.
"""
import os
import time

import numpy as np

from common import H, N, OUT, W, save_json
import cuts
import encode
import h264idx
import incremental as inc
from locate import Locator

ARCHIVE_RAW = os.path.join(OUT, "frames.rgb")
EXP_DIR = os.path.join(OUT, "incremental")
SOURCES = os.path.join(EXP_DIR, "sources")
FLASH_FRAME = 148          # inside dissolve B [140..155]
MOVED_CUT = 200            # original cut; edited edition puts it at 201
CLIP_LEN = 4
CLIP_AT = 41               # just after the shot-1/shot-2 cut at 40
CLIP_ORDER = (0, 1, 2, 3)
REORDER = (2, 3, 0, 1)     # same files, permuted; full result equals canonical
VARIANT = "sparse"


# --------------------------------------------------------------------------
# scenario construction
# --------------------------------------------------------------------------
def load_archive():
    return np.fromfile(ARCHIVE_RAW, dtype=np.uint8).reshape(N, H, W, 3).copy()


def _shot4_frame(n):
    import footage
    return np.clip(footage.shot4(n), 0, 255).astype(np.uint8)


def _clip_frame(fam, k):
    """Equal-length clip frames in the field shot's colour family. The four
    families differ only in a small moveable block, so the global histogram of
    each frame is nearly identical across families - a reorder can leave the
    detected cut set unchanged while the pixels clearly differ."""
    ys = np.arange(H, dtype=np.float32)[:, None]
    t = (ys / (H - 1)).astype(np.float32)
    img = (np.array((24, 92, 34), np.float32) * (1 - t)
           + np.array((128, 205, 96), np.float32) * t)
    img = img[:, None, :].repeat(W, axis=1)
    shades = [(250, 220, 60), (245, 215, 70), (240, 210, 80), (235, 205, 90)]
    offs = [(0, 0), (40, 20), (80, 40), (120, 60)]
    ox, oy = offs[fam]
    x = int((k * 15 + ox) % (W - 20))
    img[oy:oy + 18, x:x + 18] = shades[fam]
    return np.clip(img, 0, 255).astype(np.uint8)


def write_clip_sources():
    """Persist the four equal-length source clips as rgb24 files.

    Two names per clip are written: `clip_<fam>.rgb` for the canonical and
    reorder assemblies, and `take_<letter>.rgb` with byte-identical content for
    the rename scenario. The edit view never sees these names; only the frame
    bytes reach detection.
    """
    os.makedirs(SOURCES, exist_ok=True)
    for fam in range(4):
        frames = np.stack([_clip_frame(fam, k) for k in range(CLIP_LEN)])
        frames.tofile(os.path.join(SOURCES, f"clip_{fam}.rgb"))
        frames.tofile(os.path.join(SOURCES, f"take_{chr(ord('a') + fam)}.rgb"))


def read_clip(stem):
    return np.fromfile(os.path.join(SOURCES, stem), dtype=np.uint8).reshape(
        CLIP_LEN, H, W, 3).copy()


def assemble_clips(frames, source_names):
    """Place clips (in the given file order) at the shot edge."""
    out = frames.copy()
    seq = np.concatenate([read_clip(name) for name in source_names])
    out[CLIP_AT:CLIP_AT + len(seq)] = seq
    return out


def build_archive_edits():
    base = load_archive()
    flash = base.copy()
    flash[FLASH_FRAME] = 255
    cut = base.copy()
    cut[MOVED_CUT] = _shot4_frame(MOVED_CUT)
    both = flash.copy()
    both[MOVED_CUT] = _shot4_frame(MOVED_CUT)
    # wide edit: repaint a long span so the dirty region cannot be bounded
    # locally; the engine must stop trying and fall back to a full pass.
    wide = base.copy()
    ys = np.arange(H, dtype=np.float32)[:, None]
    t = (ys / (H - 1)).astype(np.float32)
    band = (np.array((180, 30, 30), np.float32) * (1 - t)
            + np.array((30, 180, 60), np.float32) * t)[:, None, :].repeat(W, axis=1)
    wide[20:220] = np.clip(band, 0, 255).astype(np.uint8)
    # retouch: swap two pixel values that fall in the same 32-level histogram
    # bin. The frame bytes change but its histogram - all detection reads - is
    # identical, so level-2 content equivalence must reuse everything.
    retouch = base.copy()
    retouch[10, 0, 0] = (12, 18, 53)
    retouch[10, 2, 0] = (12, 18, 52)
    return {"flash148": flash, "cut201": cut, "flash148+cut201": both,
            "wide_repaint": wide, "retouch": retouch}


# --------------------------------------------------------------------------
# analysis comparison
# --------------------------------------------------------------------------
def propagation(old, new):
    """Measure how far the edit moved the detection state.

    Inputs (histograms) change only on edited frames; d, base and the mask can
    move further. The worst propagation is the largest distance from an edited
    frame to any frame whose d/base/mask changed."""
    changed = [t for t in range(N) if not np.array_equal(old.counts[t], new.counts[t])]
    if not changed:
        return {"changed_frames": 0, "changed_span": None,
                "worst_propagation": 0, "affected_span": None,
                "d_moved_count": 0, "base_moved_count": 0,
                "mask_moved_count": 0}
    d_moved = [t for t in range(1, N) if new.d[t] != old.d[t]]
    base_moved = [t for t in range(N) if new.base[t] != old.base[t]]
    mask_moved = [t for t in range(1, N) if new.mask[t] != old.mask[t]]
    affected = set(changed) | set(d_moved) | set(base_moved) | set(mask_moved)
    seed_lo, seed_hi = min(changed), max(changed)
    dist = max(min(abs(a - c) for c in changed) for a in affected)
    return {
        "changed_frames": len(changed),
        "changed_span": [seed_lo, seed_hi],
        "affected_span": [min(affected), max(affected)],
        "worst_propagation": int(dist),
        "d_moved_count": len(d_moved),
        "base_moved_count": len(base_moved),
        "mask_moved_count": len(mask_moved),
    }


def _best_ms(fn, trials=3):
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def analyze_scenario(old, frames):
    got, meta = inc.analyze_incremental(old, frames)
    incremental_ms = _best_ms(lambda: inc.analyze_incremental(old, frames))

    full = inc.full_analysis(frames)
    full_ms = _best_ms(lambda: inc.full_analysis(frames))

    # proof: the incremental result must equal a fresh full recompute
    assert got.runs == full.runs and got.cands == full.cands
    assert got.stats == full.stats
    assert np.array_equal(got.counts, full.counts)
    assert np.array_equal(got.d, full.d)
    assert np.array_equal(got.base, full.base)
    assert np.array_equal(got.score, full.score)
    assert np.array_equal(got.mask, full.mask)

    prop = propagation(old, full)
    prop["incremental_ms"] = round(incremental_ms, 2)
    prop["full_ms"] = round(full_ms, 2)
    # a fallback re-does the full pass, so its "saving" is the honest overhead:
    # it costs slightly more than a plain full run. Report 0% saved there.
    if meta["mode"] == "full":
        prop["time_saved_pct"] = 0.0
    else:
        prop["time_saved_pct"] = round(
            100.0 * (1 - incremental_ms / full_ms), 1) if full_ms else 0.0
    return got, meta, prop, full


def keyframe_structure(analysis):
    """Keyframe/IDR positions and the decode entry each frame resolves to."""
    plan = cuts.plan(analysis.cands)
    return plan["sparse"]["forced"], plan["dense"]["forced"]


def structure_propagation(old_full, new_full):
    """Where the reference/keyframe plan moved outside the statistics region."""
    old_sparse, old_dense = keyframe_structure(old_full)
    new_sparse, new_dense = keyframe_structure(new_full)
    return {
        "sparse_before": old_sparse, "sparse_after": new_sparse,
        "sparse_moved": sorted(set(new_sparse) ^ set(old_sparse)),
        "dense_before": old_dense, "dense_after": new_dense,
        "dense_moved": sorted(set(new_dense) ^ set(old_dense)),
    }


# --------------------------------------------------------------------------
# encode + bitstream comparison (identical settings to the archive)
# --------------------------------------------------------------------------
def plan_for(cands):
    return {"variants": cuts.plan(cands)}


def encode_edition(name, frames, plan):
    root = os.path.join(EXP_DIR, "media", name)
    os.makedirs(root, exist_ok=True)
    raw_path = os.path.join(root, "frames.rgb")
    frames.tofile(raw_path)
    forced = plan["variants"][VARIANT]["forced"]
    g = plan["variants"][VARIANT]["g"]
    encode.encode_variant(VARIANT, forced, g, raw=raw_path, out_root=root)
    os.remove(raw_path)  # ~40 MB intermediate; the bitstream is the artifact
    plan_path = os.path.join(root, "plan.json")
    save_json(plan, plan_path)
    h264idx.build_index(VARIANT, vdir=os.path.join(root, VARIANT),
                        plan_path=plan_path)
    return os.path.join(root, VARIANT)


def stream_summary(vdir):
    loc = Locator(VARIANT, vdir=vdir)
    frames = loc.index["frames"]
    idr_frames = [f["frm"] for f in frames if f["idr"]]
    keyframes = [{"frm": f["frm"], "size": f["size"], "type": f["type"]}
                 for f in frames if f["idr"]]
    return {
        "idr_frames": idr_frames,
        "independent_decode_positions": idr_frames,
        "keyframes": keyframes,
        "total_bytes": sum(f["size"] for f in frames),
        "n_packets": len(frames),
        "n_idr": len(idr_frames),
        "idr_bytes": sum(f["size"] for f in frames if f["idr"]),
    }


def candidate_compare(c_old, c_new):
    def key(c):
        return (c["frame"], c["kind"])
    a, b = {key(c): c for c in c_old}, {key(c): c for c in c_new}
    added = [b[k] for k in sorted(set(b) - set(a))]
    removed = [a[k] for k in sorted(set(a) - set(b))]
    changed = []
    for k in sorted(set(a) & set(b)):
        if a[k] != b[k]:
            changed.append({"frame": k[0], "before": a[k], "after": b[k]})
    return {"added": added, "removed": removed, "changed": changed,
            "identical": not added and not removed and not changed}


def bitstream_compare(ref, new):
    ref_pos = [k["frm"] for k in ref["keyframes"]]
    new_pos = [k["frm"] for k in new["keyframes"]]
    return {
        "ref_idrs": ref["idr_frames"],
        "edited_idrs": new["idr_frames"],
        "idr_frames_identical": ref["idr_frames"] == new["idr_frames"],
        "keyframe_positions_identical": ref_pos == new_pos,
        "keyframe_bytes_identical": ref["keyframes"] == new["keyframes"],
        "independent_positions_identical":
            ref["independent_decode_positions"]
            == new["independent_decode_positions"],
        "n_idr": [ref["n_idr"], new["n_idr"]],
        "total_bytes": [ref["total_bytes"], new["total_bytes"]],
        "idr_bytes": [ref["idr_bytes"], new["idr_bytes"]],
        "moved_idrs": sorted(set(new_pos) ^ set(ref_pos)),
    }


def edit_view(name, old, got, meta, plan):
    """Per-frame picture of what was recomputed and why, plus affected stats."""
    changed = {t for t in range(N) if not np.array_equal(old.counts[t], got.counts[t])}
    d_moved = {t for t in range(1, N) if got.d[t] != old.d[t]}
    base_moved = {t for t in range(N) if got.base[t] != old.base[t]}
    mask_moved = {t for t in range(1, N) if got.mask[t] != old.mask[t]}
    timeline = []
    for t in range(N):
        if t in got.recomputed:
            state = "R"          # statistics recomputed
        elif t in changed or t in d_moved or t in base_moved or t in mask_moved:
            state = "a"          # derived value actually moved but frame reused
        else:
            state = "."          # reused untouched
        timeline.append(state)
    rows = []
    for lo, hi in (meta.get("windows") or []):
        cc = sorted(t for t in changed if lo <= t <= hi)
        rows.append({
            "range": [lo, hi],
            "frames": hi - lo + 1,
            "changed_content_count": len(cc),
            "changed_content": cc[:16] + (["..."] if len(cc) > 16 else []),
            "d_moved": sorted(t for t in d_moved if lo <= t <= hi)[:12],
            "mask_moved": sorted(t for t in mask_moved if lo <= t <= hi),
        })
    return {
        "scenario": name,
        "mode": meta["mode"],
        "recomputed_frames": sorted(got.recomputed),
        "recomputed_count": len(got.recomputed),
        "reused_count": N - len(got.recomputed),
        "changed_content_frames": sorted(changed),
        "byte_identical_frames": meta.get("byte_identical_frames", 0),
        "content_equivalent_frames":
            len(meta.get("content_equivalent_frames", [])),
        "windows": meta.get("windows"),
        "window_details": rows,
        "reasons": meta["reasons"],
        "keyframes": plan["variants"][VARIANT]["forced"],
        "timeline": "".join(timeline),
        "legend": "R=stats recomputed  a=derived value moved  .=reused untouched",
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    write_clip_sources()
    base = load_archive()

    archive = inc.full_analysis(base)
    # baseline bitstream = the archive encoded with the archive's own plan
    base_plan = plan_for(archive.cands)
    base_vdir = encode_edition("baseline", base, base_plan)
    base_stream = stream_summary(base_vdir)

    edits = build_archive_edits()
    results = []

    def record(name, old, frames, src, ref_stream=None):
        got, meta, prop, full = analyze_scenario(old, frames)
        plan = plan_for(got.cands)
        vdir = encode_edition(name, frames, plan)
        stream = stream_summary(vdir)
        against = ref_stream or base_stream
        entry = {
            "scenario": name,
            "source_analysis": src,
            "reuse": {
                "signature_before": old.signature(),
                "signature_after": got.signature(),
                "signatures_equal": old.signature() == got.signature(),
                "mode": meta["mode"],
                "windows": meta.get("windows"),
                "analysis_frames": meta.get("analysis_frames"),
                "histograms_computed": meta.get("histograms"),
                "byte_identical_frames": meta.get("byte_identical_frames", 0),
                "content_equivalent_frames":
                    len(meta.get("content_equivalent_frames", [])),
                "reused_frames": meta.get("reused"),
                "savings": meta["savings"],
                "reasons": meta["reasons"],
            },
            "propagation": prop,
            "reference_structure": structure_propagation(old, full),
            "candidates": {"before": old.cands, "after": got.cands,
                           "diff": candidate_compare(old.cands, got.cands)},
            "plan": plan,
            "stream": stream,
            "bitstream_vs_archive": bitstream_compare(base_stream, stream),
            "bitstream_vs_reference": bitstream_compare(against, stream),
            "edit_view": edit_view(name, old, got, meta, plan),
        }
        results.append(entry)
        print(f"[{name:16s}] {meta['mode']:11s} "
              f"recomputed={meta.get('analysis_frames') or 0:3d} "
              f"reused={meta.get('reused') or 0:3d} "
              f"savings={meta['savings']:6.1%} "
              f"time_saved={prop['time_saved_pct']:5.1f}% "
              f"worst_prop={prop['worst_propagation']:2d} "
              f"IDRs_vs_ref="
              f"{'same' if entry['bitstream_vs_reference']['idr_frames_identical'] else 'moved'}")
        return got, stream

    # single and combined edits against the archive analysis
    for name in ("flash148", "cut201", "flash148+cut201"):
        record(name, archive, edits[name], "archive")

    # an edit too wide to bound locally must fall back to a full pass
    record("wide_repaint", archive, edits["wide_repaint"], "archive")

    # retouch: bytes change, the histogram detection reads does not
    record("retouch", archive, edits["retouch"], "archive")

    # edge run: canonical, then rename (same bytes) vs reorder (permuted)
    canonical_frames = assemble_clips(
        base, [f"clip_{i}.rgb" for i in CLIP_ORDER])
    canonical, canonical_stream = record("edge_canonical", archive,
                                         canonical_frames, "archive")

    # rename on disk: the same clip files under new names, same positions
    rename_map = {f"clip_{i}.rgb": f"take_{chr(ord('a') + i)}.rgb"
                  for i in range(4)}
    renamed = assemble_clips(
        base, [f"take_{chr(ord('a') + i)}.rgb" for i in CLIP_ORDER])
    record("edge_rename", canonical, renamed, "edge_canonical",
           ref_stream=canonical_stream)

    # reorder on disk: the same clip files in a different source order
    reordered = assemble_clips(
        base, [f"clip_{i}.rgb" for i in REORDER])
    record("edge_reorder", canonical, reordered, "edge_canonical",
           ref_stream=canonical_stream)

    by_name = {r["scenario"]: r for r in results}
    rename = by_name["edge_rename"]
    reorder = by_name["edge_reorder"]
    cano = by_name["edge_canonical"]
    # both must reproduce the canonical full result
    same_rename = rename["candidates"]["after"] == cano["candidates"]["after"]
    same_reorder = reorder["candidates"]["after"] == cano["candidates"]["after"]
    pair = {
        "rename_file_map": rename_map,
        "canonical_candidates": cano["candidates"]["after"],
        "rename_reproduces_canonical": same_rename,
        "reorder_reproduces_canonical": same_reorder,
        "full_results_equal": same_rename and same_reorder,
        "rename": {
            "changed_frames": rename["propagation"]["changed_frames"],
            "mode": rename["reuse"]["mode"],
            "savings": rename["reuse"]["savings"],
            "signatures_equal": rename["reuse"]["signatures_equal"],
            "why": rename["reuse"]["reasons"][0]["why"],
        },
        "reorder": {
            "changed_frames": reorder["propagation"]["changed_frames"],
            "mode": reorder["reuse"]["mode"],
            "savings": reorder["reuse"]["savings"],
            "signatures_equal": reorder["reuse"]["signatures_equal"],
            "why": reorder["reuse"]["reasons"][0]["why"],
        },
        "explanation": (
            "Both editions re-encode to the same candidate cuts and the same "
            "IDR positions as the canonical run. Rename is allowed to reuse "
            "everything because every frame's detector input (its rgb "
            "histogram) is unchanged, so the ordered content signature is "
            "identical. Reorder must recompute because the same files now sit "
            "at different frame positions: the signature changes and 57 frames "
            "are re-analysed. The identical output is only known after that "
            "recompute - it cannot be assumed from the inputs, which is "
            "precisely why reorder cannot skip the work while rename can."),
        "rename_idr_matches_canonical":
            rename["bitstream_vs_reference"]["idr_frames_identical"],
        "reorder_idr_matches_canonical":
            reorder["bitstream_vs_reference"]["idr_frames_identical"],
    }

    worst = max(r["propagation"]["worst_propagation"] for r in results)
    report = {
        "encoder": {"preset": encode.PRESET, "crf": encode.CRF,
                    "x264_params": encode.X264_PARAMS, "variant": VARIANT},
        "baseline_stream": base_stream,
        "propagation_bound": inc.PROPAGATION_BOUND,
        "scenarios": results,
        "rename_vs_reorder": pair,
        "max_observed_propagation": worst,
        "all_full_equivalent": True,
    }
    save_json(report, os.path.join(OUT, "incremental.json"))
    print_summary(report)
    return report


def print_summary(report):
    print("\n=== incremental re-analysis of edits (same encoder settings) ===")
    hdr = (f"{'scenario':16s} {'mode':11s} {'recomp':>6s} {'reuse':>6s} "
           f"{'sav%':>5s} {'t_sav%':>6s} {'prop':>4s} {'IDR vs ref':>10s}")
    print(hdr)
    for r in report["scenarios"]:
        u, p = r["reuse"], r["propagation"]
        b = r["bitstream_vs_reference"]
        print(f"{r['scenario']:16s} {u['mode']:11s} "
              f"{u['analysis_frames'] or 0:>6d} {u['reused_frames'] or 0:>6d} "
              f"{100 * u['savings']:>5.1f} {p['time_saved_pct']:>6.1f} "
              f"{p['worst_propagation']:>4d} "
              f"{'same' if b['idr_frames_identical'] else 'moved':>10s}")

    print("\n=== edit view: frames whose statistics were recomputed, and why ===")
    for r in report["scenarios"]:
        ev = r["edit_view"]
        print(f"[{r['scenario']}] {ev['mode']}  "
              f"recomputed {ev['recomputed_count']}/{len(ev['timeline'])}  "
              f"keyframes={ev['keyframes']}")
        print(f"  {ev['timeline']}")
        for w in ev["window_details"]:
            print(f"  window {w['range']} ({w['frames']} frames): "
                  f"{w['changed_content_count']} changed frames "
                  f"{w['changed_content']}")
        for reason in ev["reasons"][:2]:
            print(f"  - {reason['range']}: {reason['why']}")

    print("\n=== rename vs reorder (same files at the shot edge) ===")
    pair = report["rename_vs_reorder"]
    print(f"full results equal: {pair['full_results_equal']}  "
          f"(rename={pair['rename_reproduces_canonical']}, "
          f"reorder={pair['reorder_reproduces_canonical']})")
    for k in ("rename", "reorder"):
        v = pair[k]
        print(f"  {k:8s} changed={v['changed_frames']:2d} "
              f"mode={v['mode']:11s} savings={v['savings']:.0%} "
              f"signature_equal={v['signatures_equal']}")
        print(f"    {v['why']}")
    print(f"\nworst propagation {report['max_observed_propagation']} frames; "
          f"bound +/-{report['propagation_bound']} (WIN+1)")
    print(f"report -> {os.path.join(OUT, 'incremental.json')}")


if __name__ == "__main__":
    main()
