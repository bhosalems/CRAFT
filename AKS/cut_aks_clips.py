"""Cut AKS-selected frames out of source videos and re-encode each (query, video)
pair as a short mp4 at <out_root>/q<query_id>/<video_id>.mp4, in temporal order.

The cut clip contains exactly the frames AKS selected, encoded at --output_fps
(default 1.0). Downstream MAGMAR-MWV reads each clip with its standard
video_path API, so set MAGMAR-MWV's runtime fps >= --output_fps and max_frames
>= the largest selected_frames count to ingest every AKS frame.

Optionally pass --queries to filter to a subset of query_ids (e.g. dev split).
The (query, video) pairs cut do NOT depend on the topic mapping used at
evaluation time; the mapping is read at MAGMAR-MWV invocation time.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

from decord import VideoReader, cpu


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selected_frames', required=True, help='selected_frames.json from frame_select.py')
    p.add_argument('--meta', required=True, help='meta.json from feature_extract_folder.py')
    p.add_argument('--source_video_root', required=True, help='dir containing original <video_id>.mp4 files')
    p.add_argument('--out_root', required=True, help='output dir; clips written to <out_root>/q<query_id>/<video_id>.mp4')
    p.add_argument('--queries', default=None, help='optional JSONL to filter by query_id (only its query_ids will be cut)')
    p.add_argument('--output_fps', type=float, default=1.0, help='temporal rate of the curated clip (default 1.0)')
    p.add_argument('--crf', type=int, default=20, help='libx264 CRF quality (lower = better, default 20)')
    p.add_argument('--overwrite', action='store_true', help='re-cut even if output exists')
    p.add_argument('--ffmpeg', default='ffmpeg', help='path to ffmpeg binary')
    return p.parse_args()


def load_query_filter(path):
    if not path:
        return None
    keep = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                keep.add(str(json.loads(line)['query_id']))
    return keep


def encode_clip(frames, out_path, fps, crf, ffmpeg_bin):
    """frames: list of (H, W, 3) uint8 numpy arrays. Pipes raw RGB to ffmpeg."""
    if not frames:
        raise ValueError('no frames to encode')
    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        # libx264 with yuv420p requires even dims; pad by 1 if odd.
        new_h = h + (h % 2)
        new_w = w + (w % 2)
        # pad with edge pixels so dims become even
        import numpy as np
        padded = []
        for f in frames:
            pf = np.zeros((new_h, new_w, 3), dtype=f.dtype)
            pf[:h, :w] = f
            padded.append(pf)
        frames = padded
        h, w = new_h, new_w

    cmd = [
        ffmpeg_bin, '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}', '-r', f'{fps}',
        '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf),
        '-movflags', '+faststart',
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            assert f.shape == (h, w, 3) and f.dtype.name == 'uint8'
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'ffmpeg failed (rc={rc}) for {out_path}:\n{err}')


def main():
    args = parse_args()

    # Resolve ffmpeg up front so missing binary fails fast with a clear message
    # rather than per-clip spam.
    ffmpeg_bin = args.ffmpeg
    if not os.path.isabs(ffmpeg_bin):
        resolved = shutil.which(ffmpeg_bin)
        if resolved is None:
            sys.exit(
                f"error: '{ffmpeg_bin}' not found on PATH. "
                f"Install via: conda install -n AKS -c conda-forge ffmpeg, "
                f"or pass --ffmpeg /absolute/path/to/ffmpeg"
            )
        ffmpeg_bin = resolved
    elif not os.path.exists(ffmpeg_bin):
        sys.exit(f"error: --ffmpeg path does not exist: {ffmpeg_bin}")
    args.ffmpeg = ffmpeg_bin
    print(f'[info] using ffmpeg: {ffmpeg_bin}')

    with open(args.selected_frames) as f:
        selected = json.load(f)
    with open(args.meta) as f:
        meta = json.load(f)
    if len(selected) != len(meta):
        sys.exit(f'length mismatch: selected_frames has {len(selected)} entries, meta has {len(meta)}')

    keep_qids = load_query_filter(args.queries)
    if keep_qids is not None:
        print(f'[info] filtering to {len(keep_qids)} query_ids from {args.queries}')

    counts = {'total': 0, 'cut': 0, 'skipped_filter': 0, 'skipped_exists': 0,
              'skipped_no_source': 0, 'skipped_no_frames': 0, 'failed': 0}
    by_query = {}

    for idx, (m, frames) in enumerate(zip(meta, selected)):
        counts['total'] += 1
        qid = str(m['query_id'])
        vid = m['video_id']

        if keep_qids is not None and qid not in keep_qids:
            counts['skipped_filter'] += 1
            continue
        if not frames:
            print(f'[warn] empty selected_frames for q{qid}/{vid}; skipping')
            counts['skipped_no_frames'] += 1
            continue

        src = os.path.join(args.source_video_root, f'{vid}.mp4')
        if not os.path.exists(src):
            print(f'[warn] missing source {src}; skipping')
            counts['skipped_no_source'] += 1
            continue

        out_dir = os.path.join(args.out_root, f'q{qid}')
        out_path = os.path.join(out_dir, f'{vid}.mp4')
        if os.path.exists(out_path) and not args.overwrite:
            counts['skipped_exists'] += 1
            by_query.setdefault(qid, 0)
            by_query[qid] += 1
            continue
        os.makedirs(out_dir, exist_ok=True)

        try:
            vr = VideoReader(src, ctx=cpu(0), num_threads=1)
            n_total = len(vr)
            valid = [f for f in frames if 0 <= f < n_total]
            if len(valid) != len(frames):
                print(f'[warn] q{qid}/{vid}: {len(frames)-len(valid)} indices out of range; trimming')
            if not valid:
                counts['skipped_no_frames'] += 1
                continue
            arrs = [vr[i].asnumpy() for i in valid]
            encode_clip(arrs, out_path, args.output_fps, args.crf, args.ffmpeg)
            counts['cut'] += 1
            by_query.setdefault(qid, 0)
            by_query[qid] += 1
            if counts['cut'] % 10 == 0 or idx == len(meta) - 1:
                print(f'[info] {counts["cut"]} cut so far; latest: q{qid}/{vid} ({len(valid)} frames)', flush=True)
        except Exception as e:
            print(f'[err] q{qid}/{vid} failed: {e}')
            counts['failed'] += 1

    print()
    print(f'[done] total entries:        {counts["total"]}')
    print(f'       cut now:              {counts["cut"]}')
    print(f'       already existed:      {counts["skipped_exists"]}')
    print(f'       skipped (not in qid filter): {counts["skipped_filter"]}')
    print(f'       skipped (no source):  {counts["skipped_no_source"]}')
    print(f'       skipped (empty sel):  {counts["skipped_no_frames"]}')
    print(f'       failed:               {counts["failed"]}')
    print()
    print(f'[summary] clips per query (cut + already existed):')
    for qid in sorted(by_query, key=lambda x: int(x)):
        print(f'  q{qid}: {by_query[qid]}')


if __name__ == '__main__':
    main()
