#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_REPO="$REPO_ROOT/external/HumanGS"
COMMIT=b97d2f24beeda76da8fafe9c17ba7434a23def97
if [[ ! -e "$METHOD_REPO" ]]; then
  git clone https://github.com/Devdoot57/HumanGS.git "$METHOD_REPO"
  git -C "$METHOD_REPO" checkout --detach "$COMMIT"
fi
[[ "$(git -C "$METHOD_REPO" rev-parse HEAD)" == "$COMMIT" ]] || { echo 'HumanGS checkout differs from the tested revision' >&2; exit 1; }
bash "$REPO_ROOT/scripts/setup/link_smplx_model.sh" "$METHOD_REPO/body_models/smplx/SMPLX_NEUTRAL.npz"
if ! conda run -n humangs python --version >/dev/null 2>&1; then
  conda create -y -n humangs python=3.11 pip
fi
conda install -y -n humangs -c nvidia -c conda-forge cuda-toolkit=13.0 gcc_linux-64=13 gxx_linux-64=13 ninja
ENV_PREFIX="$(conda run -n humangs python -c 'import sys; print(sys.prefix)')"
export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDNN_LIB_CONFIG=FULL
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
conda run --no-capture-output -n humangs python -m pip install -r "$METHOD_REPO/requirements.txt" pyyaml imageio h5py scipy torchmetrics torch-fidelity fvcore iopath ninja setuptools
# Torch's pinned cuDNN 9.20 wheel omits an optional engine and can load an
# incompatible system library. Use a complete ABI-compatible cuDNN 9.24 set.
# https://github.com/pytorch/pytorch/issues/188892
conda run --no-capture-output -n humangs python -m pip install --no-deps nvidia-cudnn-cu13==9.24.0.43
if ! conda run -n humangs python -c 'import json, importlib.metadata, pytorch3d._C; assert json.loads(importlib.metadata.distribution("pytorch3d").read_text("direct_url.json"))["vcs_info"]["commit_id"] == "75ebeeaea0908c5527e7b1e305fbc7681382db47"' >/dev/null 2>&1; then
  conda run --no-capture-output -n humangs env FORCE_CUDA=1 NVCC_FLAGS=-static-global-template-stub=false python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' --no-build-isolation
fi
if [[ ! -d "$METHOD_REPO/diff-gaussian-rasterization" ]]; then
  git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$METHOD_REPO/diff-gaussian-rasterization"
fi
git -C "$METHOD_REPO/diff-gaussian-rasterization" checkout --detach 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
git -C "$METHOD_REPO/diff-gaussian-rasterization" submodule update --init --recursive
# CUDA 13 requires an explicit declaration for uint32_t in this upstream header.
if ! grep -q '#include <cstdint>' "$METHOD_REPO/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"; then
  sed -i '1i #include <cstdint>' "$METHOD_REPO/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
fi
conda run --no-capture-output -n humangs python -m pip install "$METHOD_REPO/diff-gaussian-rasterization" --no-build-isolation
conda run --no-capture-output -n humangs python "$REPO_ROOT/scripts/diagnostics/check_humangs_env.py"
echo 'HumanGS environment ready.'
