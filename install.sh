#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

# Build from a fresh activation context; the caller's shell is unchanged.
unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV

usage() {
  cat <<'EOF'
Usage: ./install.sh [--cli-only | pig-avatar | mmlphuman | animatable-gaussians]

Install the CLI, MMLPHuman and Animatable Gaussians by default.
Also install PiG-Avatar when its private source is accessible.
Choose one method to install only its runtime, or --cli-only for the CLI.
Set VCI_CLI_ENV to choose the CLI Conda environment, MULTIVIEW_DATA_CHECKOUT to use
an editable dependency checkout, and VCI_SMPLX_MODEL to link a licensed model.
EOF
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "${1:-}" in
  '') methods=(mmlphuman animatable_gaussians) ;;
  --cli-only) methods=() ;;
  pig-avatar|neu-pig-smplx) methods=(pig_avatar) ;;
  mmlphuman) methods=(mmlphuman) ;;
  animatable-gaussians) methods=(animatable_gaussians) ;;
  *) usage >&2; exit 2 ;;
esac

# Also works when distributed as a source archive with no declared submodules.
if [[ -e .git || -f .gitmodules ]]; then
  git submodule update --init --recursive
fi

data_repo="${MULTIVIEW_DATA_CHECKOUT:-$root/../multiview-data}"
packages=(-e '.[ui]')
if [[ -f "$data_repo/pyproject.toml" ]]; then
  packages+=(-e "$data_repo[native]")
elif [[ -n "${MULTIVIEW_DATA_CHECKOUT:-}" ]]; then
  echo "Invalid MULTIVIEW_DATA_CHECKOUT: $data_repo" >&2
  exit 1
else
  packages+=('multiview-data[native]>=0.1,<0.2')
fi
if [[ -n "${VCI_CONDA_EXE:-}" ]]; then
  conda="$VCI_CONDA_EXE"
elif [[ -x "$root/../.miniforge/bin/conda" ]]; then
  conda="$root/../.miniforge/bin/conda"
else
  conda="$(command -v conda || true)"
fi
if [[ -z "$conda" ]] || ! command -v -- "$conda" >/dev/null 2>&1; then
  echo 'Conda is required. Put Miniforge or Miniconda on PATH, or set VCI_CONDA_EXE.' >&2
  exit 1
fi
conda="$("$conda" info --base)/bin/conda"
export PATH="$(dirname -- "$conda"):$PATH"
cli_env="${VCI_CLI_ENV:-vci-avatar-creation}"
envs="$("$conda" env list)"
if ! awk -v name="$cli_env" '$1 == name {found=1} END {exit !found}' <<< "$envs"; then
  "$conda" create -y -n "$cli_env" "python=${PYTHON_VERSION:-3.10}" pip
fi
prefix="$("$conda" run -n "$cli_env" python -c 'import sys; print(sys.prefix)')"
"$conda" run --no-capture-output -p "$prefix" python -m pip install "${packages[@]}"
"$conda" run --no-capture-output -p "$prefix" avatar-create --list-methods >/dev/null
printf '%s\n%s\n' "$conda" "$prefix" > .cli-conda
printf '\nInstalled CLI environment: %s\n' "$prefix"
if [[ $# -eq 0 ]]; then
  export PIG_REPO_URL="${PIG_REPO_URL:-https://github.com/JuKalt/neu-pig-smplx.git}"
  if [[ -e external/neu-pig-smplx/.git ]] || \
     GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}" \
       git ls-remote --exit-code "$PIG_REPO_URL" HEAD >/dev/null 2>&1; then
    methods+=(pig_avatar)
  else
    printf '\nSkipping optional PiG-Avatar: its repository is not accessible with the current Git credentials.\n' >&2
    printf 'Install it later with ./install.sh pig-avatar once you have access.\n' >&2
  fi
fi
for method in "${methods[@]}"; do
  bash "scripts/setup/setup_${method}_env.sh"
done
printf '\nTry: ./run.sh --list-methods\n'
