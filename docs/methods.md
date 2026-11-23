# Method inputs and outputs

Reading a dataset and training a method are separate capabilities. Select
calibrated RGB, masks and synchronized SMPL-X. Methods that need numeric
filenames, an undistorted camera or auxiliary grids produce their staging
inside the selected run output.

| Input | PiG-Avatar | MMLPHuman | Animatable Gaussians |
| --- | --- | --- | --- |
| Processed VCI or THuman-style data | Manifest-backed reader | Run-local THuman staging | Run-local THuman staging |
| Prepared DanceNet3D | THuman-style media and fitted parameters | THuman-style staging | THuman-style staging |
| DNA native SMC | Native adapter | Native export using authorized SMCReader | Reprocess/export first |
| TalkBody4D or 4D-DRESS | Native adapter | Reprocess/export first | Reprocess/export first |
| ActorsHQ | Run-local exporter and fit sidecar | Run-local exporter and fit sidecar | Run-local exporter and fit sidecar |

Dataset-specific camera/frame split support varies by method. Shared examples
in `configs/datasets` are templates with fictional subject IDs. The CLI accepts
`--subjects` for explicit selection and passes further arguments after `--` to
the adapter. Store checkpoints in the output root, alongside method configuration,
a subject manifest for the run, logs and evaluation split information.
Additional research adapters (MMLPHuman LOD and HumanGS) remain available as
experimental methods with their own runtime entry points.

PIG's training step count is controlled by method configuration as well as
`--epochs`. Set `optimization.finetuning.enabled` to false in a separate
method config for a short smoke test. The MMLPHuman adapter prepares the LBS
weight grid using PointInterpolant unless already present. Animatable Gaussians
generates its position maps before training unless existing maps are supplied.
Its upstream trainer controls the training length. Use explicit run timeouts
or the upstream stopping controls where available.
