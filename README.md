# Avatar Creation

[Usage examples](examples/README.md): preview and train each method, reuse a run config, or schedule subjects.

Create avatars from the Multiview Data catalog using PiG-Avatar,
MMLPHuman, or Animatable Gaussians. Each adapter owns
method-specific staging, training and export configuration. The method runtimes
are isolated from the main CLI; they have different CUDA and Python requirements.

For standalone setup, put Conda on `PATH`. The installer uses `vci-avatar-creation`
by default; set `VCI_CLI_ENV` to choose another environment. If you installed
the method through VCI Suite, skip the install command and run the remaining
commands from the suite's `avatar-creation/` directory.

```bash
VCI_SMPLX_MODEL=/path/to/SMPLX_NEUTRAL.npz ./install.sh
./run.sh --list-methods
./run.sh mmlphuman --dataset-kind processed --dataset-path /data/processed \
  --subjects example-subject --output-dir /data/avatars --no-tui --dry-run -- \
  --views 4 --time-steps 16 --resolution 512
```

Replace the dataset path and subject ID, then remove `--dry-run` after reviewing
the preview. `--list-methods` lists registered adapters, including ones whose
runtime has not been installed.

The launcher uses the locally installed CLI without shell activation. The
installer provisions GPU runtimes in separate Conda environments. It fetches pinned upstream checkouts
and initializes their nested submodules. Full setup needs Linux, Conda on
`PATH`, a compatible NVIDIA GPU and driver, and network access. The default
install skips PiG-Avatar if the private source is inaccessible and still
installs the other two methods. If you gain access later, run `./install.sh pig-avatar`; `PIG_REPO_URL` can select an
authorized mirror.
Provide an authorized SMPL-X model via `VCI_SMPLX_MODEL` or install it in the
method checkout before training. No model weights or datasets are bundled.

Existing jobs can still use the legacy method ID `neu-pig-smplx`. Both names
use the same output directory so existing runs can resume.

Run `./install.sh mmlphuman` (or `pig-avatar` or `animatable-gaussians`) to
install just that method and the CLI. The methods use different CUDA versions,
so choose the method your machine supports. Use `./install.sh --cli-only` to
inspect datasets and methods without building GPU runtimes. The installer uses a
sibling `multiview-data` checkout if present; set `MULTIVIEW_DATA_CHECKOUT` to use
a different checkout. Otherwise it installs the compatible package from your
configured Python package index once it is published.

Choose one dataset and subject selection as described in [methods](docs/methods.md).
For multi-machine runs, adapt [the Device Jobs example](examples/avatar-job.json).
The same `avatar-create` command can run independently of the scheduler.

For synthetic adapter checks, use a development environment with the `test`
extra and run `python -m unittest discover -v` from the checkout. Tests that
compare against upstream source need the corresponding checkout. Training and
render checks also need the isolated method runtime and authorized model assets.
