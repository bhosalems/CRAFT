#!/usr/bin/env python3
"""Offline per-video ASR — designed to run OUTSIDE the main pipeline.

For each video referenced by the topic mapping, produce a JSON transcript
on disk. The main pipeline (extract_query_claims.py) reads these caches
via ``--asr-dir`` and pastes the transcript into the VLM prompt. The
script never coexists with vLLM in the same process, and re-runs are
idempotent (existing transcripts are skipped unless ``--force``).

Two backends, gated by ``--mode``:
  - ``qwen``  : Qwen3-ASR-1.7B only. Covers 30 languages incl.
                English/Chinese/Cantonese/Thai. Marks low-resource clips
                with ``needs_fallback: true`` for a later omni pass.
  - ``omni``  : facebook/omniASR-LLM-7B only. Processes any video where
                the cache is missing or where Qwen marked ``needs_fallback``.
                Tries ``mya_Mymr`` then ``nep_Deva`` and keeps the longer
                transcript.
  - ``auto``  : Both (Qwen first, then omni if needed). Convenient for an
                env that has both packages installed.

Why split? ``omnilingual-asr`` requires fairseq2/fairseq2n, which has
strict torch-version pinning that may conflict with vLLM. The two-mode
design lets you generate Qwen transcripts from your main env, then run
``--mode omni`` from a separate fairseq2-compatible env against the same
cache directory.

Usage:
    # In the main env (Qwen only):
    python extract_asr.py \
        --mode qwen \
        --video-root /a2il/data/mbhosale/MAGMaR2026_test \
        --mapping data/topic_video_mapping_dev_v2.json \
        --out-dir /a2il/data/mbhosale/MAGMaR2026_test/asr

    # In an isolated omniASR env, against the same cache:
    python extract_asr.py \
        --mode omni \
        --video-root /a2il/data/mbhosale/MAGMaR2026_test \
        --mapping data/topic_video_mapping_dev_v2.json \
        --out-dir /a2il/data/mbhosale/MAGMaR2026_test/asr
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

_logger = logging.getLogger("extract_asr")


# Qwen3-ASR's officially supported languages. The runtime API returns
# full English names (e.g. "English", "Thai") rather than ISO codes, so
# we match against names; we also accept ISO codes defensively in case
# a future package version changes the format.
QWEN3_ASR_LANGS = {
    # Full names (what Qwen3ASRModel.transcribe() actually returns):
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Hungarian", "Macedonian", "Romanian",
    # ISO codes (defensive — accepted but not currently emitted):
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
    "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
    "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
}

# Minimum transcript length (after strip) below which we treat Qwen3-ASR
# as having failed and fall back to omniASR.
MIN_TEXT_LEN = 5

# Qwen3-detected language name → omniASR BCP-47+Script code.
#
# Policy: the omni pass uses the language Qwen3 already detected on the clip
# — one language per audio. If Qwen3 returned a language we don't have a
# mapping for, we skip the clip rather than guess; an empty/wrong transcript
# is worse than none. Add entries here as new languages appear in the data.
# omniASR itself supports 1,600+ languages, so the only constraint is having
# the right BCP-47+Script code on hand.
QWEN3_TO_OMNI_LANG: Dict[str, str] = {
    # Qwen3-supported languages — they reach the omni pass when Qwen3's
    # transcript was too short / loopy / missing.
    "English":    "eng_Latn",
    "Chinese":    "zho_Hans",
    "Cantonese":  "yue_Hant",
    "Arabic":     "arb_Arab",
    "German":     "deu_Latn",
    "French":     "fra_Latn",
    "Spanish":    "spa_Latn",
    "Portuguese": "por_Latn",
    "Indonesian": "ind_Latn",
    "Italian":    "ita_Latn",
    "Korean":     "kor_Hang",
    "Russian":    "rus_Cyrl",
    "Thai":       "tha_Thai",
    "Vietnamese": "vie_Latn",
    "Japanese":   "jpn_Jpan",
    "Turkish":    "tur_Latn",
    "Hindi":      "hin_Deva",
    "Malay":      "zsm_Latn",
    "Dutch":      "nld_Latn",
    "Swedish":    "swe_Latn",
    "Danish":     "dan_Latn",
    "Finnish":    "fin_Latn",
    "Polish":     "pol_Latn",
    "Czech":      "ces_Latn",
    "Filipino":   "fil_Latn",
    "Persian":    "pes_Arab",
    "Greek":      "ell_Grek",
    "Hungarian":  "hun_Latn",
    "Macedonian": "mkd_Cyrl",
    "Romanian":   "ron_Latn",
    # Low-resource languages that Qwen3 detects but cannot transcribe well —
    # these are the primary reason the omni pass exists at all.
    "Burmese":    "mya_Mymr",
    "Nepali":     "nep_Deva",
    "Tamil":      "tam_Taml",
    "Khmer":      "khm_Khmr",
    "Lao":        "lao_Laoo",
    "Tibetan":    "bod_Tibt",
    "Mongolian":  "khk_Cyrl",
    "Sinhala":    "sin_Sinh",
    "Bengali":    "ben_Beng",
    "Punjabi":    "pan_Guru",
    "Gujarati":   "guj_Gujr",
    "Marathi":    "mar_Deva",
    "Telugu":     "tel_Telu",
    "Kannada":    "kan_Knda",
    "Malayalam":  "mal_Mlym",
    "Urdu":       "urd_Arab",
    "Pashto":     "pbt_Arab",
    "Hebrew":     "heb_Hebr",
    "Swahili":    "swh_Latn",
    "Yoruba":     "yor_Latn",
    "Hausa":      "hau_Latn",
    "Amharic":    "amh_Ethi",
    "Norwegian":  "nob_Latn",
    "Bulgarian":  "bul_Cyrl",
    "Serbian":    "srp_Cyrl",
    "Croatian":   "hrv_Latn",
    "Slovak":     "slk_Latn",
    "Slovenian":  "slv_Latn",
    "Estonian":   "est_Latn",
    "Latvian":    "lvs_Latn",
    "Lithuanian": "lit_Latn",
    "Ukrainian":  "ukr_Cyrl",
    "Catalan":    "cat_Latn",
    "Basque":     "eus_Latn",
    "Galician":   "glg_Latn",
}

# Same policy for Whisper, but using ISO-639-1 codes. We do NOT use
# Whisper's auto-detect — same reason as above (silently wrong language is
# worse than no transcript).
QWEN3_TO_WHISPER_LANG: Dict[str, str] = {
    "English": "en", "Chinese": "zh", "Cantonese": "yue", "Arabic": "ar",
    "German": "de", "French": "fr", "Spanish": "es", "Portuguese": "pt",
    "Indonesian": "id", "Italian": "it", "Korean": "ko", "Russian": "ru",
    "Thai": "th", "Vietnamese": "vi", "Japanese": "ja", "Turkish": "tr",
    "Hindi": "hi", "Malay": "ms", "Dutch": "nl", "Swedish": "sv",
    "Danish": "da", "Finnish": "fi", "Polish": "pl", "Czech": "cs",
    "Filipino": "tl", "Persian": "fa", "Greek": "el", "Hungarian": "hu",
    "Macedonian": "mk", "Romanian": "ro",
    "Burmese": "my", "Nepali": "ne", "Tamil": "ta", "Khmer": "km",
    "Lao": "lo", "Mongolian": "mn", "Sinhala": "si", "Bengali": "bn",
    "Punjabi": "pa", "Gujarati": "gu", "Marathi": "mr", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Urdu": "ur", "Pashto": "ps",
    "Hebrew": "he", "Swahili": "sw", "Yoruba": "yo", "Hausa": "ha",
    "Amharic": "am", "Norwegian": "no", "Bulgarian": "bg", "Serbian": "sr",
    "Croatian": "hr", "Slovak": "sk", "Slovenian": "sl", "Estonian": "et",
    "Latvian": "lv", "Lithuanian": "lt", "Ukrainian": "uk", "Catalan": "ca",
    "Basque": "eu", "Galician": "gl",
}


def _root_cause(exc: BaseException) -> BaseException:
    """Walk ``__cause__`` / ``__context__`` to the deepest underlying exception.

    fairseq2 (and other pipeline libs) wraps failures with generic messages
    like "The map function has failed while processing the path 'data'…" —
    the actual decoder / OOM / shape-mismatch error sits in ``__cause__``.
    """
    seen: set = {id(exc)}
    cur = exc
    while True:
        nxt = cur.__cause__ or cur.__context__
        if nxt is None or id(nxt) in seen:
            return cur
        seen.add(id(nxt))
        cur = nxt


def _is_whisper_loop(text: str, *, min_tokens: int = 20, ttr_floor: float = 0.18,
                    max_consecutive_run: int = 8, max_ngram_share: float = 0.4) -> bool:
    """Heuristic detector for Whisper loop / repetition hallucinations.

    Whisper (especially on low-resource languages like Burmese, Nepali, Malay)
    sometimes gets stuck and emits the same token or short phrase hundreds of
    times. We flag the transcript as broken when ANY of:

    - **TTR (type-token ratio)** falls below ``ttr_floor`` for transcripts of
      ``min_tokens`` or more — i.e. the vocabulary is suspiciously narrow.
    - **Longest consecutive run** of an identical token is ``max_consecutive_run``
      or more (e.g. "the the the the the the the the").
    - **Most-frequent 3-gram** accounts for ``max_ngram_share`` or more of all
      3-grams (a phrase loop like "I don't know I don't know I don't know").

    Conservative thresholds: real human speech has TTR > 0.3, runs < 4, and
    no single 3-gram dominates. False positives on legitimate repetition
    (e.g. song chorus) are acceptable since the VLM still has the visuals.
    """
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < min_tokens:
        return False

    # 1) Type-token ratio
    distinct = len(set(tokens))
    ttr = distinct / len(tokens)
    if ttr < ttr_floor:
        return True

    # 2) Longest consecutive run
    longest_run = current = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current += 1
            if current > longest_run:
                longest_run = current
        else:
            current = 1
    if longest_run >= max_consecutive_run:
        return True

    # 3) Dominant 3-gram share
    if len(tokens) >= 3:
        from collections import Counter
        trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
        if trigrams:
            counts = Counter(trigrams)
            top_count = counts.most_common(1)[0][1]
            if top_count / len(trigrams) >= max_ngram_share:
                return True

    return False


# ---------------------------------------------------------------------------
# Audio extraction (ffmpeg) + audio-stream probe
# ---------------------------------------------------------------------------

def _has_audio_stream(video_path: Path) -> bool:
    """Return True iff the video file contains at least one audio stream."""
    try:
        import av  # type: ignore
    except Exception:
        # Probe via ffprobe as a fallback.
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return True  # assume yes; let downstream fail loudly
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "default=nw=1:nk=1", str(video_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return "audio" in (proc.stdout or "")

    try:
        container = av.open(str(video_path))
        try:
            return any(s.type == "audio" for s in container.streams)
        finally:
            container.close()
    except Exception:
        return False


def _extract_audio_to_wav(video_path: Path, out_wav: Path, *, sr: int = 16000) -> None:
    """Extract a mono PCM WAV at the given sample rate.

    Prefers a system ffmpeg binary when available (fast subprocess pipe).
    Falls back to PyAV + soundfile when ffmpeg is not in PATH — PyAV
    bundles its own libav, so this works without any system install.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-i", str(video_path),
            "-ac", "1", "-ar", str(sr), "-vn",
            "-loglevel", "error",
            str(out_wav),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 0:
            return
        # fall through to PyAV on ffmpeg failure
        _logger.warning(
            "ffmpeg failed for %s (rc=%s), retrying with PyAV: %s",
            video_path, proc.returncode, proc.stderr.decode(errors="ignore").strip(),
        )

    # PyAV path — decodes via bundled libav, no system ffmpeg required.
    import av  # type: ignore
    import numpy as np
    import soundfile as sf  # type: ignore

    container = av.open(str(video_path))
    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError(f"no audio stream in {video_path}")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sr)

        chunks: list = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()  # shape (1, n) for mono s16
                if arr.ndim == 2:
                    arr = arr.reshape(-1)
                chunks.append(arr.astype(np.int16, copy=False))
        # Flush the resampler.
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            if arr.ndim == 2:
                arr = arr.reshape(-1)
            chunks.append(arr.astype(np.int16, copy=False))

        if not chunks:
            raise RuntimeError(f"PyAV decoded zero audio samples from {video_path}")
        audio = np.concatenate(chunks)
        sf.write(str(out_wav), audio, sr, subtype="PCM_16")
    finally:
        container.close()


# ---------------------------------------------------------------------------
# ASR backends
# ---------------------------------------------------------------------------

class Qwen3ASRBackend:
    """Thin wrapper around Qwen/Qwen3-ASR-1.7B."""

    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-1.7B", device: str = "cuda:0",
                 download_dir: Optional[str] = None):
        import torch  # noqa: WPS433
        from qwen_asr import Qwen3ASRModel  # type: ignore

        kwargs = dict(dtype=torch.bfloat16, device_map=device)
        if download_dir:
            kwargs["cache_dir"] = download_dir
        _logger.info("Loading Qwen3-ASR (%s) on %s", model_id, device)
        self.model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)

    def transcribe(self, audio_path: Path) -> Dict[str, object]:
        results = self.model.transcribe(audio=str(audio_path), language=None)
        first = results[0] if results else None
        if first is None:
            return {"text": "", "language": None}
        return {
            "text": getattr(first, "text", "") or "",
            "language": getattr(first, "language", None),
        }


class OmniASRBackend:
    """Thin wrapper around facebook/omniASR-LLM-7B via the omnilingual_asr pkg."""

    # omniASR rejects anything strictly above 40 s
    # (omnilingual_asr/models/inference/pipeline.py: MAX_ALLOWED_AUDIO_SEC=40).
    # Use a slightly smaller window so floating-point duration math can't trip
    # the cap.
    _MAX_CHUNK_SEC = 38.0

    def __init__(self, model_card: str = "omniASR_LLM_7B"):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # type: ignore

        _logger.info("Loading omniASR (%s)", model_card)
        self.pipeline = ASRInferencePipeline(model_card=model_card)

    @staticmethod
    def _extract_text(result) -> str:
        if not result:
            return ""
        first = result[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(first.get("text", "") or first.get("transcription", "") or "")
        return str(first)

    def transcribe(self, audio_path: Path, target_lang: str) -> str:
        import soundfile as sf  # type: ignore

        info = sf.info(str(audio_path))
        duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0

        if duration <= self._MAX_CHUNK_SEC:
            result = self.pipeline.transcribe([str(audio_path)], lang=[target_lang], batch_size=1)
            return self._extract_text(result)

        # Long audio: split into ≤38 s windows on disk and transcribe each.
        # Keeping chunks on disk (vs. passing arrays) matches the pipeline's
        # file-based input contract and avoids surprises in fairseq2 data graph.
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        win = int(self._MAX_CHUNK_SEC * sr)
        n_chunks = (len(audio) + win - 1) // win
        _logger.info(
            "omniASR(%s): audio %.1fs > %.0fs cap; splitting into %d chunks",
            target_lang, duration, self._MAX_CHUNK_SEC, n_chunks,
        )

        pieces: List[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_chunks):
                chunk = audio[i * win : (i + 1) * win]
                if len(chunk) == 0:
                    continue
                chunk_path = Path(tmpdir) / f"chunk_{i:04d}.wav"
                sf.write(str(chunk_path), chunk, sr, subtype="PCM_16")
                try:
                    result = self.pipeline.transcribe(
                        [str(chunk_path)], lang=[target_lang], batch_size=1,
                    )
                except Exception as exc:
                    root = _root_cause(exc)
                    _logger.warning(
                        "omniASR chunk %d/%d failed (%s): %s: %s",
                        i + 1, n_chunks, target_lang,
                        type(root).__name__, root,
                    )
                    continue
                piece = self._extract_text(result).strip()
                if piece:
                    pieces.append(piece)
        return " ".join(pieces)


class WhisperBackend:
    """openai/whisper-large-v3 via transformers — works in the main env (no
    fairseq2 dependency). The caller is responsible for passing a valid
    Whisper ISO-639-1 code as ``target_lang`` (see ``QWEN3_TO_WHISPER_LANG``).
    Whisper auto-detect is intentionally NOT used: an unknown language means
    we skip the clip rather than risk a wrong-language transcript.
    """

    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "cuda:0"):
        import torch  # noqa: WPS433
        from transformers import pipeline  # type: ignore

        _logger.info("Loading Whisper (%s) on %s", model_id, device)
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch.float16,
            device=device,
            chunk_length_s=30,  # Whisper handles 30s windows; longer audio is auto-chunked.
            return_timestamps=False,
        )

    def _load_audio(self, audio_path: Path):
        import soundfile as sf  # type: ignore
        import numpy as np
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    def transcribe(self, audio_path: Path, target_lang: str) -> str:
        audio, sr = self._load_audio(audio_path)
        gen_kwargs = {"task": "transcribe", "language": target_lang}
        result = self.asr(
            {"array": audio, "sampling_rate": sr},
            generate_kwargs=gen_kwargs,
        )
        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        return ""

    def translate(self, audio_path: Path, source_lang: Optional[str] = None) -> str:
        """Whisper task=translate → forces English output regardless of input.
        ``source_lang`` is either a Whisper ISO code (preferred — pass the
        value resolved from ``QWEN3_TO_WHISPER_LANG``) or a Qwen3 language
        name (resolved here as a convenience). When unmapped/None, Whisper
        auto-detects — acceptable for translate since the output is forced
        to English regardless.
        """
        audio, sr = self._load_audio(audio_path)
        gen_kwargs = {"task": "translate"}
        if source_lang:
            mapped = QWEN3_TO_WHISPER_LANG.get(source_lang, source_lang)
            gen_kwargs["language"] = mapped
        result = self.asr(
            {"array": audio, "sampling_rate": sr},
            generate_kwargs=gen_kwargs,
        )
        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        return ""


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------

def _resolve_video_path(video_root: Path, video_id: str) -> Optional[Path]:
    candidate = video_root / f"{video_id}.mp4"
    return candidate if candidate.exists() else None


def _cache_path(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}.json"


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _qwen3_pass(
    backend: Qwen3ASRBackend,
    video_paths: Dict[str, Path],
    out_dir: Path,
    *,
    force: bool,
    verbose: bool,
) -> List[str]:
    """Run Qwen3-ASR over every video. Returns video_ids that need fallback."""
    needs_fallback: List[str] = []

    for video_id, video_path in video_paths.items():
        cache = _cache_path(out_dir, video_id)
        if cache.exists() and not force:
            try:
                with cache.open() as f:
                    existing = json.load(f)
            except Exception:
                existing = None
            if existing and existing.get("text", "").strip():
                if verbose:
                    print(f"  [cache-hit] {video_id} ({existing.get('asr_model')})")
                continue
            if existing and existing.get("needs_fallback"):
                needs_fallback.append(video_id)
                continue

        if not _has_audio_stream(video_path):
            _write_cache(cache, {
                "video_id": video_id,
                "no_audio": True,
                "asr_model": None,
                "language": None,
                "text": "",
                "segments": [],
            })
            if verbose:
                print(f"  [no-audio] {video_id}")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            wav = Path(tmpdir) / f"{video_id}.wav"
            try:
                _extract_audio_to_wav(video_path, wav)
            except Exception as exc:
                root = _root_cause(exc)
                _logger.warning(
                    "Audio extract failed for %s: %s: %s (outer: %s: %s)",
                    video_id, type(root).__name__, root, type(exc).__name__, exc,
                    exc_info=True,
                )
                _write_cache(cache, {
                    "video_id": video_id, "asr_model": None, "language": None,
                    "text": "", "segments": [],
                    "error": f"{type(root).__name__}: {root}",
                })
                continue

            t0 = time.time()
            try:
                result = backend.transcribe(wav)
            except Exception as exc:
                root = _root_cause(exc)
                _logger.warning(
                    "Qwen3-ASR failed for %s: %s: %s (outer: %s: %s)",
                    video_id, type(root).__name__, root, type(exc).__name__, exc,
                    exc_info=True,
                )
                _write_cache(cache, {
                    "video_id": video_id, "asr_model": "Qwen/Qwen3-ASR-1.7B",
                    "language": None, "text": "", "segments": [],
                    "error": f"{type(root).__name__}: {root}",
                    "needs_fallback": True,
                })
                needs_fallback.append(video_id)
                continue

        text = (result.get("text") or "").strip()
        lang = result.get("language")
        elapsed = time.time() - t0
        is_loop = _is_whisper_loop(text)
        # If Qwen produced a loop, treat the transcript as missing and
        # route the video to the fallback backend (Whisper) which may
        # do better. We keep the model+language metadata for traceability.
        if is_loop:
            payload = {
                "video_id": video_id,
                "asr_model": "Qwen/Qwen3-ASR-1.7B",
                "language": lang,
                "text": "",
                "segments": [],
                "elapsed_seconds": round(elapsed, 2),
                "asr_loop_detected": True,
                "needs_fallback": True,
            }
            _write_cache(cache, payload)
            if verbose:
                print(f"  [qwen3] {video_id}  lang={lang}  chars={len(text)} LOOP -> fallback  ({elapsed:.1f}s)")
            needs_fallback.append(video_id)
            continue

        route_to_fallback = (
            (not text or len(text) < MIN_TEXT_LEN)
            or (lang is not None and lang not in QWEN3_ASR_LANGS)
        )

        payload = {
            "video_id": video_id,
            "asr_model": "Qwen/Qwen3-ASR-1.7B",
            "language": lang,
            "text": text,
            "segments": [],
            "elapsed_seconds": round(elapsed, 2),
            "needs_fallback": bool(route_to_fallback),
        }
        _write_cache(cache, payload)
        if verbose:
            tag = "->fallback" if route_to_fallback else "ok"
            print(f"  [qwen3] {video_id}  lang={lang}  chars={len(text)}  {tag}  ({elapsed:.1f}s)")
        if route_to_fallback:
            needs_fallback.append(video_id)

    return needs_fallback


def _omni_pass(
    backend,
    video_ids: List[str],
    video_paths: Dict[str, Path],
    out_dir: Path,
    *,
    lang_map: Dict[str, str],
    verbose: bool,
    backend_name: Optional[str] = None,
) -> None:
    """For each video, run the fallback backend exactly once using the
    language already detected by Qwen3 in the prior pass. Updates the
    cache entry in place. Works with either OmniASRBackend or
    WhisperBackend — both expose ``transcribe(audio_path, target_lang)``.

    Policy: at most one language per clip. If Qwen3 didn't detect a
    language, or detected one we don't have a mapping for in ``lang_map``,
    the clip is skipped (text stays empty, ``needs_fallback`` stays true).
    Better no ASR than wrong-language ASR.
    """
    if backend_name is None:
        backend_name = type(backend).__name__

    unmapped_langs: Dict[str, int] = {}
    n_skipped_no_lang = 0
    n_skipped_unmapped = 0
    n_transcribed = 0
    n_loop_suppressed = 0
    n_failed = 0

    for video_id in video_ids:
        video_path = video_paths.get(video_id)
        if video_path is None:
            continue
        cache = _cache_path(out_dir, video_id)
        try:
            with cache.open() as f:
                existing = json.load(f)
        except Exception:
            existing = {"video_id": video_id}

        qwen_lang = (existing.get("language") or "").strip()
        if not qwen_lang:
            n_skipped_no_lang += 1
            if verbose:
                print(f"  [skip:no-lang] {video_id} — Qwen3 detected no language; not guessing")
            continue
        target_lang = lang_map.get(qwen_lang)
        if target_lang is None:
            n_skipped_unmapped += 1
            unmapped_langs[qwen_lang] = unmapped_langs.get(qwen_lang, 0) + 1
            if verbose:
                print(f"  [skip:unmapped] {video_id} — Qwen3 lang={qwen_lang!r} not in {backend_name} lang map")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            wav = Path(tmpdir) / f"{video_id}.wav"
            try:
                _extract_audio_to_wav(video_path, wav)
            except Exception as exc:
                root = _root_cause(exc)
                _logger.warning(
                    "Audio extract failed (omni) for %s: %s: %s (outer: %s: %s)",
                    video_id, type(root).__name__, root, type(exc).__name__, exc,
                    exc_info=True,
                )
                n_failed += 1
                continue

            t0 = time.time()
            try:
                text = (backend.transcribe(wav, target_lang=target_lang) or "").strip()
            except Exception as exc:
                root = _root_cause(exc)
                _logger.warning(
                    "%s(%s) failed for %s: %s: %s (outer: %s: %s)",
                    backend_name, target_lang, video_id,
                    type(root).__name__, root,
                    type(exc).__name__, exc,
                    exc_info=True,
                )
                n_failed += 1
                continue
            elapsed = time.time() - t0

        is_loop = _is_whisper_loop(text)
        if verbose:
            short = backend_name.split('/')[-1][:16]
            flag = " LOOP" if is_loop else ""
            print(
                f"  [{short}:{target_lang}] {video_id}  qwen_lang={qwen_lang}  "
                f"chars={len(text)}{flag}  ({elapsed:.1f}s)"
            )

        if text and not is_loop:
            existing.update({
                "asr_model": backend_name,
                "language": target_lang,
                "qwen3_language": qwen_lang,
                "text": text,
                "segments": [],
                "needs_fallback": False,
            })
            _write_cache(cache, existing)
            n_transcribed += 1
        elif text and is_loop:
            # Single-shot looped — record the failure but don't write garbage.
            existing.update({
                "asr_model": backend_name,
                "language": target_lang,
                "qwen3_language": qwen_lang,
                "text": "",
                "segments": [],
                "asr_loop_detected": True,
                "needs_fallback": True,
            })
            _write_cache(cache, existing)
            n_loop_suppressed += 1
            if verbose:
                print(f"  [{backend_name}] {video_id} loop detected; transcript suppressed")
        else:
            # Empty transcript: leave cache as-is (needs_fallback stays True).
            n_failed += 1

    print(
        f"[{backend_name}] summary: transcribed={n_transcribed} "
        f"loop_suppressed={n_loop_suppressed} failed={n_failed} "
        f"skipped_no_lang={n_skipped_no_lang} skipped_unmapped={n_skipped_unmapped}"
    )
    if unmapped_langs:
        top = sorted(unmapped_langs.items(), key=lambda kv: -kv[1])
        print(f"[{backend_name}] unmapped Qwen3 languages (grow lang_map to handle): {top}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_topic_mapping(path: Path) -> Dict[str, List[str]]:
    with path.open() as f:
        data = json.load(f)
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
    return out


def _videos_needing_translation(video_ids: List[str], out_dir: Path) -> List[str]:
    """Return videos that have a non-English transcript but no English
    translation yet. Skips videos with no transcript or no audio.
    """
    out: List[str] = []
    for vid in video_ids:
        cache = _cache_path(out_dir, vid)
        if not cache.exists():
            continue
        try:
            with cache.open() as f:
                rec = json.load(f)
        except Exception:
            continue
        if rec.get("no_audio"):
            continue
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        lang = (rec.get("language") or "").strip()
        # Treat empty/None/"unknown" as worth translating (defensive).
        if lang.lower() in {"english", "en"}:
            continue
        if (rec.get("text_en") or "").strip():
            continue  # already translated
        out.append(vid)
    return out


def _translate_pass(
    backend: "WhisperBackend",
    video_ids: List[str],
    video_paths: Dict[str, Path],
    out_dir: Path,
    *,
    verbose: bool,
) -> None:
    """For each cache entry needing translation, run Whisper task=translate
    and write the result into a ``text_en`` field on the cache JSON.
    Leaves the original ``text`` unchanged.
    """
    for video_id in video_ids:
        video_path = video_paths.get(video_id)
        if video_path is None:
            continue
        cache = _cache_path(out_dir, video_id)
        try:
            with cache.open() as f:
                existing = json.load(f)
        except Exception:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            wav = Path(tmpdir) / f"{video_id}.wav"
            try:
                _extract_audio_to_wav(video_path, wav)
            except Exception as exc:
                root = _root_cause(exc)
                _logger.warning(
                    "Audio extract failed (translate) for %s: %s: %s (outer: %s: %s)",
                    video_id, type(root).__name__, root, type(exc).__name__, exc,
                    exc_info=True,
                )
                continue

            t0 = time.time()
            try:
                text_en = (backend.translate(wav, source_lang=existing.get("language")) or "").strip()
            except Exception as exc:
                root = _root_cause(exc)
                _logger.warning(
                    "Whisper translate failed for %s: %s: %s (outer: %s: %s)",
                    video_id, type(root).__name__, root, type(exc).__name__, exc,
                    exc_info=True,
                )
                continue
            elapsed = time.time() - t0

        if text_en:
            existing["text_en"] = text_en
            existing["translation_model"] = "openai/whisper-large-v3 (translate)"
            _write_cache(cache, existing)
            if verbose:
                print(f"  [translate] {video_id}  src_lang={existing.get('language')}  en_chars={len(text_en)}  ({elapsed:.1f}s)")
        else:
            if verbose:
                print(f"  [translate] {video_id}  empty translation, skipped")


def _videos_needing_omni(video_ids: List[str], out_dir: Path) -> List[str]:
    """Return videos with no cache OR a cache flagged ``needs_fallback``."""
    out: List[str] = []
    for vid in video_ids:
        cache = _cache_path(out_dir, vid)
        if not cache.exists():
            out.append(vid)
            continue
        try:
            with cache.open() as f:
                rec = json.load(f)
        except Exception:
            out.append(vid)
            continue
        if rec.get("no_audio"):
            continue  # nothing to do
        if rec.get("needs_fallback") or not (rec.get("text") or "").strip():
            out.append(vid)
    return out


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    ap = argparse.ArgumentParser(description="Per-video ASR pre-pass (run separately from the main pipeline)")
    ap.add_argument("--mode", choices=("qwen", "omni", "whisper", "translate", "clean-loops", "auto"), default="auto",
                    help=("Which backend(s) to run. 'qwen' = Qwen3-ASR only (current env). "
                          "'omni' = omniASR-LLM-7B only (separate fairseq2 env). "
                          "'whisper' = Whisper large-v3 only via transformers — works in "
                          "the main env, supports Burmese/Nepali, no fairseq2 needed. "
                          "'translate' = post-processing pass that runs Whisper task=translate "
                          "on every cache entry whose detected language is non-English, "
                          "writing the English translation into a new ``text_en`` field. "
                          "'clean-loops' = no-GPU pass that re-checks every cache entry "
                          "with the loop detector and clears the text on hallucinated "
                          "transcripts (run after fixing the detector or upgrading it). "
                          "'auto' = Qwen first then omni for routed videos."))
    ap.add_argument("--whisper-model", default="openai/whisper-large-v3",
                    help="HF model id for the Whisper backend (only used in --mode whisper).")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--mapping", required=True,
                    help="Topic->videos mapping JSON (post-chunking, e.g. topic_video_mapping_dev_v2.json)")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write per-video transcript JSON. Recommend a path co-located with the videos so it can be reused across branches and envs.")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--omni-model-card", default="omniASR_LLM_7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--download-dir", default="")
    ap.add_argument("--force", action="store_true",
                    help="Recompute transcripts even if a cache JSON already exists.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = _load_topic_mapping(Path(args.mapping))
    unique_video_ids = sorted({v for vs in mapping.values() for v in vs})
    video_paths: Dict[str, Path] = {}
    missing: List[str] = []
    for vid in unique_video_ids:
        p = _resolve_video_path(video_root, vid)
        if p is None:
            missing.append(vid)
        else:
            video_paths[vid] = p

    print(f"[asr] mode={args.mode}")
    print(f"[asr] mapping: {args.mapping}")
    print(f"[asr] unique videos referenced: {len(unique_video_ids)} (missing on disk: {len(missing)})")
    print(f"[asr] cache dir: {out_dir}")

    needs_fallback: List[str] = []

    # Pass 0: cache scrubber. Re-applies _is_whisper_loop to every existing
    # cache entry and clears the text on hallucinated transcripts. No GPU
    # needed; useful after upgrading the detector, or for cleaning a cache
    # produced by older versions of this script.
    if args.mode == "clean-loops":
        scrubbed = inspected = 0
        for vid in unique_video_ids:
            cache = _cache_path(out_dir, vid)
            if not cache.exists():
                continue
            try:
                with cache.open() as f:
                    rec = json.load(f)
            except Exception:
                continue
            inspected += 1
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            if _is_whisper_loop(text):
                rec["text"] = ""
                rec["asr_loop_detected"] = True
                rec["needs_fallback"] = True
                # Drop any stale translation tied to a now-empty transcript.
                rec.pop("text_en", None)
                rec.pop("translation_model", None)
                _write_cache(cache, rec)
                scrubbed += 1
                if args.verbose:
                    print(f"  [clean-loops] {vid} text was a loop; cleared (was {len(text)} chars)")
        print(f"[asr] clean-loops: inspected {inspected} cache files, cleared {scrubbed} loop transcripts")
        return

    # Pass 1: Qwen3-ASR (skipped in --mode omni)
    if args.mode in ("qwen", "auto"):
        qwen = Qwen3ASRBackend(
            model_id=args.qwen_model,
            device=args.device,
            download_dir=args.download_dir or None,
        )
        needs_fallback = _qwen3_pass(
            qwen, video_paths, out_dir, force=args.force, verbose=args.verbose,
        )
        del qwen
        try:
            import torch  # noqa: WPS433
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Pass 2: low-resource fallback (omniASR or Whisper, depending on --mode).
    if args.mode in ("omni", "whisper", "auto"):
        if args.mode in ("omni", "whisper"):
            # Standalone fallback run — figure out which videos still need ASR.
            target_ids = _videos_needing_omni(list(video_paths.keys()), out_dir)
        else:
            target_ids = needs_fallback

        if target_ids:
            if args.mode == "whisper":
                print(f"[asr] {len(target_ids)} videos to run through Whisper")
                backend = WhisperBackend(
                    model_id=args.whisper_model,
                    device=args.device,
                )
                backend_name = args.whisper_model
                lang_map = QWEN3_TO_WHISPER_LANG
            else:
                print(f"[asr] {len(target_ids)} videos to run through omniASR")
                backend = OmniASRBackend(model_card=args.omni_model_card)
                backend_name = "facebook/omniASR-LLM-7B"
                lang_map = QWEN3_TO_OMNI_LANG
            _omni_pass(
                backend, target_ids, video_paths, out_dir,
                lang_map=lang_map, verbose=args.verbose,
                backend_name=backend_name,
            )
            del backend
            try:
                import torch  # noqa: WPS433
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            print("[asr] no videos need fallback ASR — nothing to do")

    # Pass 3: translation pass — runs when --mode translate. Iterates the
    # cache, finds entries with non-English transcripts and no ``text_en``,
    # and runs Whisper task=translate on the audio to produce the English
    # translation. Stores it as a separate field; original ``text`` is kept.
    if args.mode == "translate":
        target_ids = _videos_needing_translation(list(video_paths.keys()), out_dir)
        if target_ids:
            print(f"[asr] {len(target_ids)} videos to translate to English")
            backend = WhisperBackend(
                model_id=args.whisper_model,
                device=args.device,
            )
            _translate_pass(
                backend, target_ids, video_paths, out_dir,
                verbose=args.verbose,
            )
            del backend
            try:
                import torch  # noqa: WPS433
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            print("[asr] no non-English transcripts pending translation — nothing to do")

    # Summary
    have_text = 0
    pending = 0
    no_audio = 0
    for vid in unique_video_ids:
        cache = _cache_path(out_dir, vid)
        if not cache.exists():
            pending += 1
            continue
        try:
            with cache.open() as f:
                rec = json.load(f)
        except Exception:
            continue
        if rec.get("no_audio"):
            no_audio += 1
        elif (rec.get("text") or "").strip():
            have_text += 1
        elif rec.get("needs_fallback"):
            pending += 1
    print(f"[asr] done: {have_text}/{len(unique_video_ids)} have transcripts; "
          f"{no_audio} no-audio; {pending} pending (need omni or never run)")


if __name__ == "__main__":
    sys.exit(main() or 0)
