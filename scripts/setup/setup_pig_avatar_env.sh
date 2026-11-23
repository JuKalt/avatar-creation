#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIG_REPO="$REPO_ROOT/external/neu-pig-smplx"
PIG_REPO_URL="${PIG_REPO_URL:-https://github.com/JuKalt/neu-pig-smplx.git}"
PIG_REPO_COMMIT="${PIG_REPO_COMMIT:-9ec01d2ecde940f6cb3c572c387fa27fada67796}"
ENV_NAME="pig-avatar"
PYTORCH_VERSION="${PIG_PYTORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${PIG_TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${PIG_TORCHAUDIO_VERSION:-2.7.1}"
CUDA_VERSION="${PIG_CUDA_VERSION:-12.8}"
CHECK_DEVICE="${PIG_CHECK_DEVICE:-cuda:0}"

# Apply limits before any installer can build native dependencies, including
# pcgrid's cholespy dependency and packages installed by conda env update.
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"

prepend_path() {
  local variable="$1"
  local value="$2"
  local current="${!variable:-}"
  if [[ -n "$current" ]]; then
    export "$variable=$value:$current"
  else
    export "$variable=$value"
  fi
}

validate_torch_cuda() {
  local stage="$1"
  echo "Validating PyTorch/GPU compatibility ($stage)"
  conda run --no-capture-output -n "$ENV_NAME" python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; capability = torch.cuda.get_device_capability(); required = f"sm_{capability[0]}{capability[1]}"; supported = torch.cuda.get_arch_list(); assert required in supported, f"GPU requires {required}, but this PyTorch wheel supports only {supported}"; tensor = torch.zeros(1, device="cuda"); print(f"PyTorch {torch.__version__} supports {torch.cuda.get_device_name()} ({required})")'
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda or Miniforge first." >&2
  exit 1
fi

command -v git >/dev/null 2>&1 || {
  echo "git was not found on PATH. It is required to clone PiG-Avatar." >&2
  exit 1
}

if [[ ! -e "$PIG_REPO" ]]; then
  mkdir -p "$(dirname "$PIG_REPO")"
  git clone --no-checkout "$PIG_REPO_URL" "$PIG_REPO"
elif [[ ! -d "$PIG_REPO/.git" ]] || ! git -C "$PIG_REPO" diff --quiet || \
     ! git -C "$PIG_REPO" diff --cached --quiet; then
  echo "PiG-Avatar path must be a Git checkout with no tracked edits: $PIG_REPO" >&2
  exit 1
fi
if ! git -C "$PIG_REPO" cat-file -e "$PIG_REPO_COMMIT^{commit}" 2>/dev/null; then
  git -C "$PIG_REPO" fetch origin "$PIG_REPO_COMMIT"
fi
git -C "$PIG_REPO" checkout --detach "$PIG_REPO_COMMIT"

if [[ ! -f "$PIG_REPO/environment.yml" ]]; then
  echo "PiG-Avatar checkout is invalid or incomplete: $PIG_REPO" >&2
  echo "Remove or repair that directory, then rerun this script." >&2
  exit 1
fi

bash "$REPO_ROOT/scripts/setup/link_smplx_model.sh" "$PIG_REPO/models/smplx/SMPLX_NEUTRAL.npz"

cd "$PIG_REPO"

echo "Initializing nested PiG-Avatar dependencies"
git submodule sync --recursive
git submodule update --init --recursive

if [[ ! -f "ext/pcgrid/pyproject.toml" && ! -f "ext/pcgrid/setup.py" ]]; then
  echo "The PiG commit declares pcgrid but does not track its gitlink; cloning it directly."
  mkdir -p ext
  git clone https://github.com/vc-bonn/pcgrid ext/pcgrid
fi

if [[ ! -f "ext/pcgrid/pyproject.toml" && ! -f "ext/pcgrid/setup.py" ]]; then
  echo "PiG-Avatar pcgrid source is unavailable: $PIG_REPO/ext/pcgrid" >&2
  exit 1
fi

echo "Creating or updating Conda environment: $ENV_NAME"
conda env update -f environment.yml --prune

# Merely requesting the `openblas` package does not constrain Conda's virtual
# BLAS implementation. The solver can otherwise combine BLIS libblas with
# Netlib LAPACK, whose CMake target references a missing versioned libblas file.
# cholespy discovers that broken target while building pcgrid. Pin both virtual
# packages to the same OpenBLAS implementation before compiling native wheels.
echo "Pinning a consistent OpenBLAS/LAPACK implementation"
conda install -y -n "$ENV_NAME" -c conda-forge \
  'libblas=*=*openblas' \
  'liblapack=*=*openblas'

echo "Installing the CUDA $CUDA_VERSION compiler/runtime for Blackwell support"
conda install -y -n "$ENV_NAME" -c nvidia -c conda-forge \
  "cuda-nvcc=$CUDA_VERSION" \
  "cuda-nvcc_linux-64=$CUDA_VERSION" \
  "cuda-cudart=$CUDA_VERSION" \
  "cuda-cudart-dev=$CUDA_VERSION" \
  "cuda-libraries-dev=$CUDA_VERSION" \
  "cuda-cccl=$CUDA_VERSION"

# Conda's CUDA headers live below targets/x86_64-linux rather than directly in
# $CONDA_PREFIX/include. PyTorch JIT extensions (including gsplat) otherwise
# fail with "cuda_runtime_api.h: No such file or directory".
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
CUDA_TARGET="$ENV_PREFIX/targets/x86_64-linux"
if [[ ! -f "$CUDA_TARGET/include/cuda_runtime_api.h" ]]; then
  echo "CUDA runtime headers are missing: $CUDA_TARGET/include/cuda_runtime_api.h" >&2
  exit 1
fi
export CUDA_HOME="$ENV_PREFIX"
prepend_path CPATH "$CUDA_TARGET/include"
prepend_path CPLUS_INCLUDE_PATH "$CUDA_TARGET/include"
prepend_path LIBRARY_PATH "$CUDA_TARGET/lib"
prepend_path LD_LIBRARY_PATH "$CUDA_TARGET/lib"

echo "Installing PyTorch $PYTORCH_VERSION with CUDA $CUDA_VERSION support"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  "torch==$PYTORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url https://download.pytorch.org/whl/cu128

# Fail before the expensive source builds if this wheel cannot execute kernels
# for the installed GPU (for example, an sm_120 RTX 5090 with a pre-Blackwell
# PyTorch wheel).
validate_torch_cuda "before native dependency builds"
TORCH_CUDA_ARCH_LIST="$(conda run -n "$ENV_NAME" python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}.{minor}")')"
export TORCH_CUDA_ARCH_LIST
echo "Building CUDA extensions for architecture: $TORCH_CUDA_ARCH_LIST"

echo "Installing PiG-Avatar Python dependencies"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  -r requirements.txt torchmetrics torch-fidelity debugpy
# Cholespy uses nanobind's pre-3 internal keep_alive API. Constrain the
# isolated build environment; installing nanobind into Conda alone cannot do so.
conda run --no-capture-output -n "$ENV_NAME" python -m pip install ext/pcgrid \
  --build-constraint "$REPO_ROOT/scripts/setup/pig-build-constraints.txt"

echo "Installing pinned PiG-Avatar native dependencies"
(
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install --no-deps "git+https://github.com/nghorbani/human_body_prior.git"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
    --force-reinstall --no-deps "gsplat==1.5.3"
) &
build_pid=$!
while kill -0 "$build_pid" 2>/dev/null; do
  sleep 30
  if kill -0 "$build_pid" 2>/dev/null; then
    echo "Still building PiG-Avatar native dependencies (PID $build_pid, MAX_JOBS=$MAX_JOBS)..."
  fi
done
wait "$build_pid"

echo "Validating PiG-Avatar environment"
validate_torch_cuda "after native dependency builds"
conda run --no-capture-output -n "$ENV_NAME" python "$REPO_ROOT/scripts/diagnostics/check_pig_avatar_env.py" --device "$CHECK_DEVICE"

echo
echo "PiG-Avatar environment is ready."
