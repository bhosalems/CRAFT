#!/usr/bin/env python3
"""Chunk long MAGMaR videos into <=N-second MP4s (in-place).

Why:
- Some backends (torchvision/ffmpeg) can become unstable or extremely slow when
  decoding long videos under load (EAGAIN / resource temporarily unavailable).
- This utility splits any long video into smaller MP4 files so the pipeline can
  process every segment without skipping.

What it does:
- Reads one or more topic->video_id mappings (JSON)
- Probes each referenced <video_id>.mp4 under --video-root
- If duration > --max-seconds, creates chunk files in the SAME directory:
    <video_id>__chunk000.mp4, <video_id>__chunk001.mp4, ...
- Produces:
  - updated mapping JSON(s) with chunk IDs substituted
  - a chunk map JSON: chunk_id -> {original_id, start_s, end_s}

Notes:
- Original MP4s are never modified or deleted.
- Chunks are created with stream copy when possible; if that fails, we re-encode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    start_s: float
    end_s: float


def _which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def probe_duration_seconds(video_path: Path) -> float:
    ffprobe = _which("ffprobe")
    if ffprobe:
        # Use format duration; robust across codecs.
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        proc = _run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")

        s = (proc.stdout or "").strip()
        try:
            return float(s)
        except ValueError as exc:
            raise RuntimeError(
                f"Could not parse ffprobe duration output for {video_path}: {s!r}"
            ) from exc

    # Fallback: PyAV (no external binary required).
    try:
        import av  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ffprobe not found in PATH and PyAV is not importable. "
            "Install ffmpeg/ffprobe or ensure Python package 'av' is installed."
        ) from exc

    container = av.open(str(video_path))
    try:
        if container.duration is not None:
            return float(container.duration) / 1_000_000.0

        # Stream-based fallback.
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is not None and video_stream.duration is not None and video_stream.time_base is not None:
            return float(video_stream.duration * video_stream.time_base)

        raise RuntimeError(f"Could not determine duration for {video_path} (no duration metadata)")
    finally:
        container.close()


def _pyav_split_reencode(
    in_path: Path,
    video_root: Path,
    chunks: List[ChunkSpec],
    *,
    force: bool,
) -> None:
    try:
        import av  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyAV is required for chunking when ffmpeg is not available; could not import 'av'."
        ) from exc

    if not chunks:
        return

    out_paths: List[Path] = [video_root / f"{spec.chunk_id}.mp4" for spec in chunks]

    # If everything already exists and we're not forcing, do nothing.
    if not force:
        all_exist = True
        for p in out_paths:
            if not (p.exists() and p.stat().st_size > 0):
                all_exist = False
                break
        if all_exist:
            return

    in_container = av.open(str(in_path))
    try:
        in_video_stream = next((s for s in in_container.streams if s.type == "video"), None)
        if in_video_stream is None:
            raise RuntimeError(f"No video stream found in {in_path}")

        from fractions import Fraction

        fps_rate = None
        if getattr(in_video_stream, "average_rate", None):
            fps_rate = in_video_stream.average_rate
        elif getattr(in_video_stream, "base_rate", None):
            fps_rate = in_video_stream.base_rate
        if not fps_rate:
            fps_rate = Fraction(30, 1)

        # Seek close to the first chunk start to reduce work.
        start_seek_s = max(0.0, chunks[0].start_s)
        try:
            in_container.seek(int(start_seek_s * 1_000_000), any_frame=False, backward=True)
        except Exception:
            # Best-effort; decoding from start still works.
            pass

        current_idx = 0
        current_spec = chunks[current_idx]

        out_container = None
        out_stream = None

        def open_writer(spec: ChunkSpec) -> Tuple[Optional["av.container.output.OutputContainer"], Optional["av.video.stream.VideoStream"]]:
            out_path = video_root / f"{spec.chunk_id}.mp4"
            if out_path.exists() and out_path.stat().st_size > 0 and not force:
                return None, None

            out_path.parent.mkdir(parents=True, exist_ok=True)
            oc = av.open(str(out_path), mode="w")
            vs = oc.add_stream("libx264", rate=fps_rate)
            vs.width = in_video_stream.codec_context.width
            vs.height = in_video_stream.codec_context.height
            vs.pix_fmt = "yuv420p"
            vs.options = {"crf": "28", "preset": "ultrafast"}
            return oc, vs

        def close_writer(oc, vs) -> None:
            if oc is None or vs is None:
                return
            try:
                for packet in vs.encode():
                    if packet is None or packet.pts is None or packet.dts is None:
                        continue
                    try:
                        oc.mux(packet)
                    except Exception as exc:
                        print(f"[warn] mux failed during flush ({oc.name}): {exc}")
                        continue
            finally:
                try:
                    oc.close()
                except Exception as exc:
                    print(f"[warn] failed closing output container ({oc.name}): {exc}")

        out_container, out_stream = open_writer(current_spec)

        for frame in in_container.decode(video=0):
            t = getattr(frame, "time", None)
            if t is None:
                if frame.pts is None or frame.time_base is None:
                    continue
                t = float(frame.pts * frame.time_base)

            # Advance chunk index if needed.
            while current_idx < len(chunks) and t >= chunks[current_idx].end_s:
                close_writer(out_container, out_stream)
                out_container, out_stream = None, None
                current_idx += 1
                if current_idx >= len(chunks):
                    break
                current_spec = chunks[current_idx]
                out_container, out_stream = open_writer(current_spec)

            if current_idx >= len(chunks):
                break

            if t < current_spec.start_s:
                continue

            if out_container is None or out_stream is None:
                continue

            # Ensure compatible pixel format.
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")

            # Reset PTS so each chunk starts at t=0 instead of the
            # original absolute timestamp (e.g. chunk001 at 120s).
            if frame.pts is not None and frame.time_base is not None:
                offset_pts = int(current_spec.start_s / frame.time_base)
                frame.pts = frame.pts - offset_pts
                frame.dts = None  # let encoder recompute

            for packet in out_stream.encode(frame):
                if packet is None or packet.pts is None or packet.dts is None:
                    continue
                try:
                    out_container.mux(packet)
                except Exception as exc:
                    print(f"[warn] mux failed ({out_container.name}): {exc}")
                    continue

        close_writer(out_container, out_stream)
    finally:
        in_container.close()


def build_chunks(video_id: str, duration_s: float, max_seconds: float) -> List[ChunkSpec]:
    if duration_s <= max_seconds:
        return []

    n = int(math.ceil(duration_s / max_seconds))
    chunks: List[ChunkSpec] = []
    for i in range(n):
        start = i * max_seconds
        end = min(duration_s, (i + 1) * max_seconds)
        chunk_id = f"{video_id}__chunk{i:03d}"
        chunks.append(ChunkSpec(chunk_id=chunk_id, start_s=start, end_s=end))
    return chunks


def _ffmpeg_split_copy(in_path: Path, out_path: Path, start_s: float, end_s: float) -> bool:
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg on this node or ensure it is available."
        )

    dur = max(0.0, end_s - start_s)
    # Stream copy first (fast). -ss before -i is faster but may be less accurate.
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(in_path),
        "-t",
        f"{dur:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-reset_timestamps",
        "1",
        str(out_path),
    ]
    proc = _run(cmd)
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def _ffmpeg_split_reencode(in_path: Path, out_path: Path, start_s: float, end_s: float) -> None:
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg on this node or ensure it is available."
        )

    dur = max(0.0, end_s - start_s)
    # Re-encode for reliability.
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(in_path),
        "-t",
        f"{dur:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-reset_timestamps",
        "1",
        str(out_path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode split failed for {in_path} -> {out_path}: {proc.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {out_path}")


def split_video_into_chunks(
    video_root: Path,
    video_id: str,
    max_seconds: float,
    *,
    dry_run: bool,
    force: bool,
) -> Tuple[List[ChunkSpec], float]:
    in_path = video_root / f"{video_id}.mp4"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing video file: {in_path}")

    duration_s = probe_duration_seconds(in_path)
    chunks = build_chunks(video_id, duration_s, max_seconds)

    if not chunks:
        return [], duration_s

    # If ffmpeg isn't available, fall back to PyAV re-encode splitting.
    if _which("ffmpeg") is None:
        if not dry_run:
            _pyav_split_reencode(in_path, video_root, chunks, force=force)
        return chunks, duration_s

    for spec in chunks:
        out_path = video_root / f"{spec.chunk_id}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0 and not force:
            continue

        if dry_run:
            continue

        # Try fast path; fallback to re-encode.
        ok = _ffmpeg_split_copy(in_path, out_path, spec.start_s, spec.end_s)
        if not ok:
            # Remove partial output if any.
            try:
                if out_path.exists():
                    out_path.unlink()
            except OSError:
                pass
            _ffmpeg_split_reencode(in_path, out_path, spec.start_s, spec.end_s)

    return chunks, duration_s


def load_mapping(path: Path) -> Dict[str, List[str]]:
    with path.open("r") as f:
        data = json.load(f)
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
    return out


def write_mapping(path: Path, mapping: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")


def write_chunk_map(path: Path, chunk_map: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_map": chunk_map,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def expand_mapping_with_chunks(
    mapping: Dict[str, List[str]],
    chunk_specs_by_video: Dict[str, List[ChunkSpec]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for topic, vids in mapping.items():
        new_vids: List[str] = []
        for vid in vids:
            specs = chunk_specs_by_video.get(vid) or []
            if specs:
                new_vids.extend([s.chunk_id for s in specs])
            else:
                new_vids.append(vid)
        out[topic] = new_vids
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk long videos and generate *_v2 topic mappings")
    ap.add_argument("--video-root", required=True, help="Directory containing <video_id>.mp4")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Max duration per chunk in seconds (default: 120)",
    )
    ap.add_argument(
        "--mapping-in",
        action="append",
        default=[],
        help="Input topic->videos mapping JSON (can be passed multiple times)",
    )
    ap.add_argument(
        "--mapping-out",
        action="append",
        default=[],
        help="Output mapping JSON path(s), one per --mapping-in (same count)",
    )
    ap.add_argument(
        "--chunk-map-out",
        required=True,
        help="Where to write chunk_id -> original_id mapping JSON",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done; do not create chunks or write mappings",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recreate chunk files even if they already exist",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a referenced <video_id>.mp4 is missing under --video-root",
    )

    args = ap.parse_args()

    video_root = Path(args.video_root)
    if not video_root.exists():
        raise FileNotFoundError(f"video_root does not exist: {video_root}")

    if len(args.mapping_in) != len(args.mapping_out):
        raise ValueError("--mapping-in and --mapping-out must have the same count")

    mappings_in = [Path(p) for p in args.mapping_in]
    mappings_out = [Path(p) for p in args.mapping_out]

    # Collect all unique video IDs.
    all_video_ids: List[str] = []
    loaded_mappings: List[Dict[str, List[str]]] = []
    for p in mappings_in:
        m = load_mapping(p)
        loaded_mappings.append(m)
        for vids in m.values():
            all_video_ids.extend(vids)

    unique_ids = sorted(set(all_video_ids))

    chunk_specs_by_video: Dict[str, List[ChunkSpec]] = {}
    chunk_map: Dict[str, dict] = {}

    long_count = 0
    missing_count = 0
    for vid in tqdm(unique_ids, desc="Chunking videos", unit="video"):
        in_path = video_root / f"{vid}.mp4"
        if not in_path.exists():
            missing_count += 1
            if args.strict:
                raise FileNotFoundError(f"Missing video file: {in_path}")
            tqdm.write(f"[warn] missing video file, skipping chunking: {in_path}")
            continue

        chunks, duration_s = split_video_into_chunks(
            video_root,
            vid,
            args.max_seconds,
            dry_run=bool(args.dry_run),
            force=bool(args.force),
        )
        if chunks:
            long_count += 1
            chunk_specs_by_video[vid] = chunks
            for spec in chunks:
                chunk_map[spec.chunk_id] = {
                    "original_id": vid,
                    "start_s": round(spec.start_s, 3),
                    "end_s": round(spec.end_s, 3),
                }

    if args.dry_run:
        print(f"[dry-run] unique videos referenced: {len(unique_ids)}")
        print(f"[dry-run] missing mp4s under --video-root: {missing_count}")
        print(f"[dry-run] videos needing chunking (> {args.max_seconds}s): {long_count}")
        return

    # Write chunk map.
    write_chunk_map(Path(args.chunk_map_out), chunk_map)

    # Write updated mappings.
    for m_in, m_out, m in zip(mappings_in, mappings_out, loaded_mappings):
        updated = expand_mapping_with_chunks(m, chunk_specs_by_video)
        write_mapping(m_out, updated)

    print(f"[ok] chunk map -> {args.chunk_map_out} (chunks: {len(chunk_map)})")
    for out_path in args.mapping_out:
        print(f"[ok] mapping v2 -> {out_path}")


if __name__ == "__main__":
    main()
