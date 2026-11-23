#!/usr/bin/env python3
"""Run PiG-Avatar's checkpoint evaluators with the recorded VCI camera/frame selections."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

MODULES = {
    'novel-view': 'src.opt_pipeline.checkpoint_eval',
    'novel-pose': 'src.opt_pipeline.checkpoint_eval_novel_pose_pca',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--split', choices=MODULES, required=True)
    parser.add_argument('--path', type=Path, required=True, help='Checkpoint, subject output, or run directory')
    args, extra = parser.parse_known_args(argv)
    if any(value.split('=')[0] == '--dataset-root' for value in extra):
        parser.error('Evaluation uses recorded source paths; --dataset-root would not relocate the VCI manifest')
    checkpoint = args.path.expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    upstream = root / 'external/neu-pig-smplx'
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location('vci_checkpoint_evaluation', root / 'avatar_creation/methods/pig_checkpoint_evaluation.py')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    # Import the runtime before upstream model construction initializes thread pools.
    runtime = bridge._load_local('pig_avatar_run')
    from src.model_classes.avatar import AvatarModel
    from src.opt_pipeline import evaluation, novel_pose_evaluation
    bridge.install_checkpoint_evaluation_hooks(AvatarModel, evaluation, novel_pose_evaluation, runtime)
    module = MODULES[args.split]
    sys.argv = [module, '--path', str(checkpoint), *extra]
    runpy.run_module(module, run_name='__main__')
    return 0


if __name__ == '__main__':
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
