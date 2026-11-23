"""Launch unmodified HumanGS training with VCI data and black backgrounds."""
from pathlib import Path
import json
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
UPSTREAM = PROJECT_ROOT / 'external/HumanGS'
sys.path.insert(0, str(UPSTREAM))


def install_black_renderer():
    import torch
    import model.HumanGS as module
    original = module.render_predicted_gaussians
    def render(*args, **kwargs):
        kwargs['bg_color'] = torch.zeros(3, device=args[0].device)
        return original(*args, **kwargs)
    module.render_predicted_gaussians = render


def main():
    from setup import init_config
    from avatar_creation.methods.humangs_dataset import prepare_poses
    config = init_config()
    record_path = Path(config.training.train_dataset_path)
    record = json.loads(record_path.read_text())
    prepare_poses(record['source'], sorted({p['frame_id'] for p in record['training_pairs']}),
                  record_path.parent / 'poses', Path(config.model.smpl_model_path))
    from avatar_creation.methods.humangs import warmup_cuda_convolution
    warmup_cuda_convolution()
    install_black_renderer()
    import torch
    from avatar_creation.methods.humangs import input_signature
    checkpoints = sorted(Path(config.training.checkpoint_dir).glob('ckpt_*.pt'))
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1], map_location='cpu', weights_only=True)
        if checkpoint.get('vci_input_signature') != input_signature(record):
            raise ValueError('Resume checkpoint disagrees with recorded VCI training inputs')
        del checkpoint
    original_save = torch.save
    def save_checkpoint(value, destination, *args, **kwargs):
        if isinstance(value, dict) and 'model' in value and 'optimizer' in value:
            value = {**value, 'vci_input_signature': input_signature(record)}
        return original_save(value, destination, *args, **kwargs)
    torch.save = save_checkpoint
    runpy.run_path(str(UPSTREAM / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    from avatar_creation.methods.humangs import ensure_runtime_libraries
    ensure_runtime_libraries()
    main()
