#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_REPO="$REPO_ROOT/external/mmlphuman"
METHOD_REPO_URL="${MMLPHUMAN_REPO_URL:-https://github.com/1231234zhan/mmlphuman.git}"
METHOD_REPO_COMMIT="${MMLPHUMAN_REPO_COMMIT:-6668509284c85fcb0f93cd7365ec8f39390ff251}"
ENV_NAME="mmlphuman"

command -v conda >/dev/null 2>&1 || {
  echo "conda was not found on PATH. Install Miniconda or Miniforge first." >&2
  exit 1
}
if [[ ! -e "$METHOD_REPO" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git was not found on PATH. It is required to clone MMLP Human." >&2
    exit 1
  }
  mkdir -p "$(dirname "$METHOD_REPO")"
  git clone --no-checkout "$METHOD_REPO_URL" "$METHOD_REPO"
  git -C "$METHOD_REPO" checkout --detach "$METHOD_REPO_COMMIT"
fi
if [[ -e "$METHOD_REPO/.git" ]]; then
  git -C "$METHOD_REPO" submodule update --init --recursive
fi

[[ -f "$METHOD_REPO/train.py" ]] || {
  echo "MMLP Human checkout is invalid or incomplete: $METHOD_REPO" >&2
  echo "Remove or repair that directory, then rerun this script." >&2
  exit 1
}

LPIPS_PATCH="$REPO_ROOT/scripts/setup/patches/mmlphuman-lpips-reset.patch"
if git -C "$METHOD_REPO" apply --reverse --check "$LPIPS_PATCH" >/dev/null 2>&1; then
  echo "MMLP Human LPIPS training fix is already applied."
elif git -C "$METHOD_REPO" apply --check "$LPIPS_PATCH"; then
  git -C "$METHOD_REPO" apply "$LPIPS_PATCH"
  echo "Applied MMLP Human LPIPS training fix."
else
  echo "Could not apply the MMLP Human LPIPS training fix to $METHOD_REPO." >&2
  echo "Restore its utils/loss_utils.py to pinned revision $METHOD_REPO_COMMIT and rerun this script." >&2
  exit 1
fi

bash "$REPO_ROOT/scripts/setup/link_smplx_model.sh" "$METHOD_REPO/smpl_model/smplx/SMPLX_NEUTRAL.npz"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda install -y -n "$ENV_NAME" python=3.10 pip
else
  conda create -y -n "$ENV_NAME" python=3.10 pip
fi

POISSON_RECON_COMMIT="262b0f539d404057d1f36e1adc07fc9388678899"
conda install -y -n "$ENV_NAME" -c conda-forge make gxx_linux-64=11 libjpeg-turbo libpng zlib
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
POINT_INTERPOLANT="$ENV_PREFIX/bin/PointInterpolant"
if [[ ! -x "$POINT_INTERPOLANT" ]]; then
  BUILD_ROOT="$(mktemp -d)"
  trap 'rm -rf "$BUILD_ROOT"' EXIT
  git init "$BUILD_ROOT/PoissonRecon"
  git -C "$BUILD_ROOT/PoissonRecon" remote add origin https://github.com/mkazhdan/PoissonRecon.git
  git -C "$BUILD_ROOT/PoissonRecon" fetch --depth 1 origin "$POISSON_RECON_COMMIT"
  git -C "$BUILD_ROOT/PoissonRecon" checkout --detach FETCH_HEAD
  make -C "$BUILD_ROOT/PoissonRecon" \
    CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
    CFLAGS="-I$ENV_PREFIX/include" \
    LFLAGS="-L$ENV_PREFIX/lib -Wl,-rpath,$ENV_PREFIX/lib" \
    pointinterpolant
  cp "$BUILD_ROOT/PoissonRecon/Bin/Linux/PointInterpolant" "$POINT_INTERPOLANT"
  chmod +x "$POINT_INTERPOLANT"
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  torch==2.4.1 torchvision==0.19.1 "numpy<2.0" \
  --index-url https://download.pytorch.org/whl/cu121
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  h5py iopath ninja jaxtyping rich imageio numba omegaconf open3d opencv-python==4.10.0.84 \
  scipy smplx scikit-image tensorboardx tensorboard trimesh websockets \
  lpips==0.1.4 torch-fidelity torchmetrics websocket-client dearpygui plyfile==1.0.3 torch_pca pyyaml
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "numpy<2.0"

echo "MMLP Human environment is ready."
