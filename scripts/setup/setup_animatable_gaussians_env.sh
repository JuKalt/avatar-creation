#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_REPO="$REPO_ROOT/external/AnimatableGaussians"
METHOD_REPO_URL="${ANIMATABLE_GAUSSIANS_REPO_URL:-https://github.com/lizhe00/AnimatableGaussians.git}"
METHOD_REPO_COMMIT="${ANIMATABLE_GAUSSIANS_REPO_COMMIT:-2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114}"
ENV_NAME="animatable-gaussians"

command -v conda >/dev/null 2>&1 || {
  echo "conda was not found on PATH. Install Miniconda or Miniforge first." >&2
  exit 1
}
if [[ ! -e "$METHOD_REPO" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git was not found on PATH. It is required to clone Animatable Gaussians." >&2
    exit 1
  }
  mkdir -p "$(dirname "$METHOD_REPO")"
  git clone --no-checkout "$METHOD_REPO_URL" "$METHOD_REPO"
  git -C "$METHOD_REPO" checkout --detach "$METHOD_REPO_COMMIT"
fi
if [[ -e "$METHOD_REPO/.git" ]]; then
  git -C "$METHOD_REPO" submodule update --init --recursive
fi

[[ -f "$METHOD_REPO/requirements.txt" ]] || {
  echo "Animatable Gaussians checkout is invalid or incomplete: $METHOD_REPO" >&2
  echo "Remove or repair that directory, then rerun this script." >&2
  exit 1
}

bash "$REPO_ROOT/scripts/setup/link_smplx_model.sh" "$METHOD_REPO/smpl_files/smplx/SMPLX_NEUTRAL.npz"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda install -y -n "$ENV_NAME" python=3.8 pip
else
  conda create -y -n "$ENV_NAME" python=3.8 pip
fi

echo "Installing the isolated CUDA 11.7 build toolchain"
conda install -y -n "$ENV_NAME" -c nvidia \
  cuda-version=11.7 cuda-nvcc=11.7 cuda-cudart=11.7 cuda-cudart-dev=11.7 cuda-cccl=11.7 cuda-libraries-dev=11.7 gcc_linux-64=11 gxx_linux-64=11
# Python 3.8 includes crypt.h from Python.h when compiling native extensions.
conda install -y -n "$ENV_NAME" -c conda-forge libxcrypt
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
export CUDA_HOME="$ENV_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"

echo "Installing the pinned PyTorch 2.0.1 / CUDA 11.7 stack"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
  --index-url https://download.pytorch.org/whl/cu117

# pytorch3d 0.7.4 is not distributed through PyPI. Keep the upstream pins, but
# install the binary-stack packages from their official wheel indexes and keep
# the current Python-3.8-compatible Setuptools (the upstream build tag is
# no longer published on PyPI).
FILTERED_REQUIREMENTS="$(mktemp)"
BUILD_ROOT="$(mktemp -d)"
trap 'rm -f "$FILTERED_REQUIREMENTS"; rm -rf "$BUILD_ROOT"' EXIT
grep -Ev '^(torch|torchvision|pytorch3d|setuptools)==' \
  "$METHOD_REPO/requirements.txt" > "$FILTERED_REQUIREMENTS"

echo "Installing the remaining upstream requirements"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  -r "$FILTERED_REQUIREMENTS"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  iopath fvcore ninja pyyaml tensorboard

echo "Installing the PyTorch3D 0.7.4 wheel for Python 3.8 / CUDA 11.7 / PyTorch 2.0.1"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  --no-index --no-cache-dir --no-deps pytorch3d==0.7.4 \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt201/download.html

export MAX_JOBS="${MAX_JOBS:-4}"
echo "Staging native extension sources outside the upstream checkout"
cp -a "$METHOD_REPO/gaussians/diff_gaussian_rasterization_depth_alpha" "$BUILD_ROOT/rasterizer"
cp -a "$METHOD_REPO/network/styleunet" "$BUILD_ROOT/styleunet"
cp -a "$METHOD_REPO/utils/posevocab_custom_ops" "$BUILD_ROOT/posevocab_custom_ops"
cp -a "$METHOD_REPO/utils/root_finding" "$BUILD_ROOT/root_finding"

conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  "$BUILD_ROOT/rasterizer"
(
  cd "$BUILD_ROOT/styleunet"
  conda run --no-capture-output -n "$ENV_NAME" python setup.py install
)
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  "$BUILD_ROOT/posevocab_custom_ops" "$BUILD_ROOT/root_finding"

echo "Validating the Animatable Gaussians environment"
conda run --no-capture-output -n "$ENV_NAME" python -c \
  "import cv2, diff_gaussian_rasterization_depth_alpha, fused, posevocab_custom_ops, pytorch3d, root_finding, torch, upfirdn2d, yaml; assert torch.__version__.startswith('2.0.1')"

echo "Animatable Gaussians environment is ready. Add SMPL-X assets as described in README.md."
