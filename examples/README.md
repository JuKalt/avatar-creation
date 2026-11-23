# Avatar Creation examples

Run from the Avatar Creation root after installing the method you want.
`./run.sh` uses the installed CLI with no shell activation. The
examples use processed VCI output containing calibrated images, masks and
synchronized SMPL-X. Use a subject with enough complete views and frames for
the chosen method. The [method guide](../docs/methods.md) explains native
reader support and method-specific staging. You can leave another Conda
environment active in the terminal. The launcher uses the installed CLI,
and the adapter uses its own method environment.

## Preview and train MMLPHuman on processed VCI

MMLPHuman and Animatable Gaussians do not require access to the private
PiG-Avatar repository. Set `VCI_SMPLX_MODEL` to the authorized neutral model
installed by Capture Processing or another authorized copy. Skip the
`./install.sh` step if the suite already installed the method.

```bash
export VCI_SMPLX_MODEL=/path/to/SMPLX_NEUTRAL.npz
./install.sh mmlphuman  # Skip if MMLPHuman is already installed.
./run.sh mmlphuman --dataset-kind processed --dataset-path /data/processed \
  --subjects example-subject --output-dir /data/avatars --no-tui --dry-run -- \
  --views 4 --time-steps 16 --resolution 512
```

Check `/data/avatars/mmlphuman/dataset_manifest.json` for the selected subject
and `run_summary.log` beside it for the rendered run settings. The dry run
checks the required method files and environment imports before writing these
files. It does not launch the adapter or verify the GPU training path. When
the preview looks right, launch:

```bash
./run.sh mmlphuman --dataset-kind processed --dataset-path /data/processed \
  --subjects example-subject --output-dir /data/avatars --no-tui -- \
  --views 4 --time-steps 16 --resolution 512
```

The live command records its output in `runner.log` alongside the manifest.
MMLPHuman prepares the LBS grid when needed. The view, frame, and resolution
settings bound the input size; they do not limit the number of training steps.

## Use the same capture with Animatable Gaussians or PiG-Avatar

```bash
./install.sh animatable-gaussians  # Skip if already installed.
./run.sh animatable-gaussians --dataset-kind processed --dataset-path /data/processed \
  --subjects example-subject --output-dir /data/avatars --no-tui --dry-run -- \
  --views 4 --time-steps 16 --resolution 512
```

Remove `--dry-run` to train. Animatable Gaussians generates position maps
before training unless existing maps are supplied. If you have Git access
to PiG-Avatar, run `./install.sh pig-avatar` and substitute `pig-avatar` in
the command. The same input limits do not guarantee identical selections
across methods. For comparisons, inspect each run's manifest and evaluation
split files; see the [method guide](../docs/methods.md) for training controls.

## Reuse a run config for THuman

```bash
cp examples/thuman-run.json /tmp/thuman-run.json
# Edit dataset_path and subjects in the copy.
./run.sh mmlphuman --run-config /tmp/thuman-run.json \
  --output-dir /data/avatars --no-tui --dry-run
```

Use THuman input with numeric image filenames, such as `00000000.jpg`. The
[sample config](thuman-run.json) keeps the subject selection and input
settings together so you can reuse them with a different method. Run configs
accept `thuman`, `dna`, `dancenet3d`, and `actorshq`, subject to each adapter's
native input support. For `processed`, use the direct CLI flags above.

## Schedule multiple subjects

Copy [avatar-job.json](avatar-job.json) and edit the example subjects, processed
input path, and shared run and output roots. Set `machines[].cwd` to the
Avatar Creation checkout. On each worker, use the output of
`./run.sh --python -c 'import sys; print(sys.executable)'` for `machines[].python`.
Install MMLPHuman on every worker and ensure `conda` is available to the job.
To use another method, change its name and adapter arguments in `task.command`.

```bash
DEVICE_JOBS=/path/to/device-jobs/run.sh  # In the suite: "$PWD/../device-jobs/run.sh"
cp examples/avatar-job.json /tmp/avatar-job.json
# Edit /tmp/avatar-job.json as described above.
"$DEVICE_JOBS" --config /tmp/avatar-job.json --dry-run
"$DEVICE_JOBS" --config /tmp/avatar-job.json --preflight-only
"$DEVICE_JOBS" --config /tmp/avatar-job.json
```

The preview renders one subject per job. On launch, each subject gets
its own directory, with the manifest and runner log under
`output_root/<subject>/mmlphuman/`. See the shared run root for scheduler logs.
