# ASR Pre-Pass Setup

This pipeline reads per-video transcripts from a JSON cache directory
(`ASR_DIR`). The transcripts are produced **outside** the main pipeline
because the two backends — Qwen3-ASR (current env, fits with vLLM) and
Meta omniASR (requires fairseq2/fairseq2n with strict torch pinning) —
cannot live in the same Python environment.

## Layout

```
$ASR_DIR/                       # default: $VIDEO_ROOT/asr
  <video_id>.json               # one file per video
  extract_asr_qwen.log
  extract_asr_omni.log
```

Each transcript JSON has:
```jsonc
{
  "video_id": "oq4kD3XjqUk",
  "asr_model": "Qwen/Qwen3-ASR-1.7B",        // or "facebook/omniASR-LLM-7B"
  "language": "en",                           // detected
  "text": "...",
  "needs_fallback": false,                    // true => still needs omni
  "no_audio": false                           // true => skip, no audio stream
}
```

## Step 1 — Qwen3-ASR in the main venv (covers 30 languages)

This is what the project's `.venv` is for. Already installed:

```bash
cd /home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV
.venv/bin/python -m pip install qwen-asr librosa
```

Run end-to-end (idempotent — re-runs skip already-cached videos):

```bash
.venv/bin/python extract_asr.py \
    --mode qwen \
    --video-root /a2il/data/mbhosale/MAGMaR2026_test \
    --mapping   data/topic_video_mapping_dev_v2.json \
    --out-dir   /a2il/data/mbhosale/MAGMaR2026_test/asr \
    --device cuda:0 \
    --verbose
```

Videos whose detected language is outside Qwen3-ASR's 30-language set
(notably **Burmese** and **Nepali** — Q3/Q4) get an empty transcript and
`"needs_fallback": true`. Step 2 fills these in.

## Step 2 — omniASR in an isolated venv (covers Burmese / Nepali / 1600+)

omniASR depends on `fairseq2` + `fairseq2n`, which only have wheels for
torch ≤ 2.9.1. The main `.venv` runs torch 2.10.0+cu128, so we need a
separate venv. The two envs only communicate via the on-disk cache.

### Option A — conda (recommended)

```bash
# 1. Create the omni-only conda env (Python 3.12 — fairseq2 supports 3.10–3.12)
conda create -n asr_omni python=3.12 -y
conda activate asr_omni

# 2. Install torch 2.9.1 + cu128 via pip (must match fairseq2's wheel index)
pip install --upgrade pip
pip install torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install fairseq2 from Meta's wheel index (must match torch+cuda exactly).
#    NOTE: use --index-url (not --extra-index-url) so pip pulls fairseq2 from
#    Meta's index. With --extra-index-url pip will resolve fairseq2 from PyPI
#    where the default variant is built for an older torch and you'll get
#    "fairseq2 requires PyTorch 2.8.0" at import time. Pin the version that
#    matches the wheel index — current pt2.9.1/cu128 has 0.8.1.
pip install fairseq2==0.8.1 \
    --index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.9.1/cu128 \
    --extra-index-url https://pypi.org/simple

# 4. Install omnilingual-asr + audio I/O (libsndfile from conda-forge avoids
#    needing root for `apt install libsndfile1`)
conda install -c conda-forge libsndfile -y
pip install omnilingual-asr librosa soundfile

# 5. Sanity-check torch and torchaudio are both cu128 builds (fairseq2 will
#    refuse to import otherwise).
python -c "
import torch, torchaudio
print('torch', torch.__version__)
print('torchaudio', torchaudio.__version__)
assert torch.__version__.startswith('2.9.1+cu128'), 'torch is not 2.9.1+cu128'
assert torchaudio.__version__.startswith('2.9.1+cu128'), 'torchaudio is not 2.9.1+cu128'
print('cuda available:', torch.cuda.is_available())
"

# 6. Verify omniASR loads.
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('omni OK')"
```

To leave the env later: `conda deactivate`. To remove it entirely:
`conda remove -n asr_omni --all`.

### If you saw `libcudart.so.13` or `requires PyTorch 2.8.0`

Both errors come from pip silently picking the wrong torch / fairseq2 build.
Fix without recreating the env:

```bash
# Force torch + torchaudio to the cu128 build (matches what fairseq2 expects):
pip install --force-reinstall --no-deps \
    torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Force fairseq2 to the pt2.9.1 build:
pip uninstall -y fairseq2 fairseq2n
pip install fairseq2==0.8.1 \
    --index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.9.1/cu128 \
    --extra-index-url https://pypi.org/simple
```

### Option B — venv

```bash
# 1. Create the omni-only venv (Python 3.12 — fairseq2 supports 3.10–3.12)
cd /home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV
python3.12 -m venv .venv_asr_omni
source .venv_asr_omni/bin/activate

# 2. Install torch 2.9.1 + cu128 (must match fairseq2's wheel index)
pip install --upgrade pip
pip install torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install fairseq2 from Meta's wheel index (must match torch+cuda exactly)
pip install fairseq2 \
    --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.9.1/cu128

# 4. Install omnilingual-asr + audio I/O
pip install omnilingual-asr librosa soundfile

# 5. Verify
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('omni OK')"
```

If you used venv (Option B) and step 5 fails with a `libsndfile` error:

```bash
sudo apt install -y libsndfile1  # or whatever your package manager uses
```

Conda (Option A) gets `libsndfile` from conda-forge in the same step, so
this is only a venv concern.

Then run omniASR over the same cache directory. It only touches videos
that need it (missing cache OR `needs_fallback: true`):

```bash
# conda:
conda activate asr_omni
python /home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV/extract_asr.py \
    --mode omni \
    --video-root /a2il/data/mbhosale/MAGMaR2026_test \
    --mapping   /home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV/data/topic_video_mapping_dev_v2.json \
    --out-dir   /a2il/data/mbhosale/MAGMaR2026_test/asr \
    --verbose

# venv:
.venv_asr_omni/bin/python extract_asr.py \
    --mode omni \
    --video-root /a2il/data/mbhosale/MAGMaR2026_test \
    --mapping   data/topic_video_mapping_dev_v2.json \
    --out-dir   /a2il/data/mbhosale/MAGMaR2026_test/asr \
    --verbose
```

omni tries `mya_Mymr` (Burmese) first, then `nep_Deva` (Nepali), and
keeps the longer transcript. Add other target codes via
`--fallback-langs mya_Mymr,nep_Deva,...`.

## Step 3 — Run the main pipeline as usual

The pipeline reads `ASR_DIR` and pastes transcripts into the VLM prompt.
No code changes needed — `run_query.sh` looks at `ASR_DIR` (default
`$VIDEO_ROOT/asr`):

```bash
deactivate  # back to main env
TEAM_ID=cite_chasers RUN_ID=magmar_query_v12 TASK=oracle \
    MAX_CRITIC_ROUNDS=4 STEP15_CHUNK_SIZE=10 \
    ASR_DIR=/a2il/data/mbhosale/MAGMaR2026_test/asr \
    bash run_query.sh outputs_query_branchv12
```

To disable ASR for an A/B run: `ASR_DIR= bash run_query.sh ...`.

## Notes

- **Cache is content-addressable per `video_id`.** Re-running ASR is a
  no-op for already-transcribed videos. Pass `--force` to recompute.
- **Logs:** the script writes `extract_asr_qwen.log` /
  `extract_asr_omni.log` next to the cache files.
- **Disk:** Qwen3-ASR-1.7B ≈ 4 GB, omniASR-LLM-7B ≈ 14 GB. Make sure
  your HF cache (`~/.cache/huggingface`) has room.
- **No-audio chunks** are detected and recorded as
  `{"no_audio": true}`. The main pipeline skips ASR for those and falls
  back to visual-only extraction automatically.
