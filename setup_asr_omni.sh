#!/usr/bin/env bash
# One-shot setup for the omniASR-only conda env.
#
# What's pinned and why:
#   - omnilingual-asr 0.2.0 requires fairseq2[arrow] in [0.5.2, 0.6.0].
#   - fairseq2 0.6 has wheels for torch 2.8.0 + cu128 on Meta's CDN.
#   - conda's pkgs/main channel ships pytorch 2.8.0 with gpu_cuda128 build
#     and matching torchaudio 2.8.0 cuda128. Conda 2.9.1 is only cu130 there
#     (CUDA 13), which is what triggers libcudart.so.13 errors. So we stick
#     with 2.8.0 here for omniASR even though the main pipeline uses 2.10.
#   - Everything that has a conda binary comes from conda (so binaries match);
#     fairseq2 (no conda channel) and omnilingual-asr (PyPI-only) come from
#     pip last, with --no-deps to keep pip from upgrading torch.
#
# Usage:
#   bash setup_asr_omni.sh                # uses defaults below
#   ENV_NAME=my_asr bash setup_asr_omni.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-asr_omni}"
PY_VER="${PY_VER:-3.12}"
TORCH_VER="${TORCH_VER:-2.8.0}"
CUDA_TAG="${CUDA_TAG:-cu128}"
FAIRSEQ2_VER="${FAIRSEQ2_VER:-0.6}"
OMNI_VER="${OMNI_VER:-0.2.0}"

FAIRSEQ2_INDEX="https://fair.pkg.atmeta.com/fairseq2/whl/pt${TORCH_VER}/${CUDA_TAG}"

echo "[setup_asr_omni] env=$ENV_NAME py=$PY_VER torch=$TORCH_VER+$CUDA_TAG fairseq2=$FAIRSEQ2_VER omni=$OMNI_VER"

if ! command -v conda >/dev/null 2>&1; then
	echo "[error] conda not in PATH — install miniconda/anaconda first." >&2
	exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Fresh env (delete any stale one — partial installs cause cryptic errors)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
	echo "[setup_asr_omni] removing existing env '$ENV_NAME' for a clean slate"
	conda env remove -n "$ENV_NAME" -y
fi
echo "[setup_asr_omni] creating env '$ENV_NAME' with python=$PY_VER"
conda create -n "$ENV_NAME" "python=$PY_VER" -y
conda activate "$ENV_NAME"

# 2. Install pytorch + torchaudio with the cu128 build from conda main.
#    Pinning the build string is what guarantees we get cu128 (not cu130).
echo "[setup_asr_omni] installing pytorch + torchaudio (${TORCH_VER}, ${CUDA_TAG}) from conda"
conda install -y \
	"pytorch=${TORCH_VER}=gpu_cuda128*" \
	"torchaudio=${TORCH_VER}=cuda128*"

# 3. Native audio I/O from conda-forge (avoids the need for apt install libsndfile1)
echo "[setup_asr_omni] installing libsndfile + librosa + soundfile from conda-forge"
conda install -c conda-forge -y libsndfile librosa soundfile pyarrow

# 4. fairseq2 from Meta's pt2.8.0/cu128 wheel index. --no-deps so pip can't
#    upgrade torch to a different cu build.
echo "[setup_asr_omni] installing fairseq2==$FAIRSEQ2_VER (pt${TORCH_VER}/${CUDA_TAG})"
pip install --upgrade pip
pip install --no-deps "fairseq2==${FAIRSEQ2_VER}" \
	--index-url "$FAIRSEQ2_INDEX"
# fairseq2 has a few small pure-Python deps; install them from PyPI without
# touching torch.
pip install --no-deps importlib_resources packaging pyyaml typing_extensions tqdm rich huggingface_hub safetensors filelock

# 5. omnilingual-asr from PyPI. --no-deps prevents pip from upgrading torch /
#    fairseq2 to whatever it thinks is "latest". We installed all real deps
#    above already.
echo "[setup_asr_omni] installing omnilingual-asr==$OMNI_VER"
pip install --no-deps "omnilingual-asr==${OMNI_VER}"

# 6. Verify versions and imports.
echo "[setup_asr_omni] verifying installation"
python - <<PY
import torch, torchaudio, fairseq2, omnilingual_asr
print("torch     ", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda      ", torch.version.cuda, "available:", torch.cuda.is_available())
print("fairseq2  ", fairseq2.__version__)
print("omni      ", getattr(omnilingual_asr, "__version__", "n/a"))
assert "+cu128" in torch.__version__ or torch.version.cuda == "12.8", \
    f"torch is not cu128: {torch.__version__}"
PY
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('omni OK')"

echo
echo "[setup_asr_omni] DONE. To use:"
echo "    conda activate $ENV_NAME"
echo "    python /home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV/extract_asr.py \\"
echo "        --mode omni \\"
echo "        --video-root /a2il/data/mbhosale/MAGMaR2026_test \\"
echo "        --mapping   /home/csgrad/mbhosale/phd/SCALE/MAGMAR-MWV/data/topic_video_mapping_dev_v2.json \\"
echo "        --out-dir   /a2il/data/mbhosale/MAGMaR2026_test/asr \\"
echo "        --verbose"
