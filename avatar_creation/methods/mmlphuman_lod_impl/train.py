"""Run the original MMLPHuman training loop with the separate LOD model.

Only this privately loaded trainer module receives the LOD model factory and
viewer adapter. No upstream source or globally imported model class is modified.
"""

import argparse
from functools import partial
import importlib.util
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = PROJECT_ROOT / "external" / "mmlphuman"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UPSTREAM))


def load_trainer(lod):
    from scene.net_vis import Visualizer
    from avatar_creation.methods.mmlphuman_lod_impl.model import LODGaussianModel

    class LODVisualizer(Visualizer):
        def visualizing(self):
            # The viewer computes override colors before calling render().
            self.gaussians.set_lod(self.gaussians.eval_lod_fraction)
            return super().visualizing()

    spec = importlib.util.spec_from_file_location("_mmlphuman_lod_trainer", UPSTREAM / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.GaussianModel = partial(LODGaussianModel, **lod)
    trainer.Visualizer = LODVisualizer
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23456)
    pargs = parser.parse_args(argv)

    import torch
    from omegaconf import OmegaConf
    from utils.general_utils import safe_state
    from avatar_creation.methods.mmlphuman_lod_impl.lod import lod_config

    # Match upstream train.py's setup, including final checkpoint/test scheduling.
    args = OmegaConf.load(pargs.config)
    lod = lod_config(**dict(args.get("lod", {})))
    args.lod = lod
    args.data_dir, args.out_dir = pargs.data_dir, pargs.out_dir
    args.ip, args.port = pargs.ip, pargs.port
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, str(Path(args.out_dir) / "config.yaml"))
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    trainer = load_trainer(lod)
    safe_state(False, args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer.training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
