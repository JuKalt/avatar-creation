"""Manifest adapter and strict source split planning for HumanGS."""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
import os
import site
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from avatar_creation.methods.external_data import load_subjects, safe_subject_id
from avatar_creation.methods.evaluation_split import write_evaluation_split
from avatar_creation.methods.training_inputs import selection_frames
from avatar_creation.methods.experiment_config import validate_selection

UPSTREAM = PROJECT_ROOT / 'external/HumanGS'
REVISION = 'b97d2f24beeda76da8fafe9c17ba7434a23def97'
MEDIA_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def media_files(root: Path) -> dict[int, Path]:
    result = {}
    if not root.is_dir():
        return result
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in MEDIA_EXTENSIONS and path.stem.isdigit():
            frame = int(path.stem)
            if frame in result:
                raise ValueError(f'Duplicate numeric frame ID in {root}: {frame}')
            result[frame] = path.resolve()
    return result


def observations(subject: dict, selection: dict) -> tuple[list[dict], list[list[int]]]:
    calibration = json.loads(Path(subject['calibration']).read_text())
    by_id = {int(name[3:]): name for name in calibration
             if name.startswith('cam') and name[3:].isdigit()}
    # Native staging can renumber cameras; resolve physical IDs before selecting.
    mapping = subject.get('staged_camera_mapping')
    if mapping:
        by_id = {int(source): name for name, source in mapping.items()}
    pairs, missing = [], []
    for camera in selection['camera_ids']:
        if camera not in by_id:
            raise ValueError(f'Camera {camera} is unavailable for {subject["subject_id"]}')
        name = by_id[camera]
        images = media_files(Path(subject['images']) / name)
        masks = media_files(Path(subject['masks']) / name)
        for frame in selection_frames(selection):
            if frame not in images or frame not in masks:
                missing.append([camera, frame])
            else:
                pairs.append({'camera_id': camera, 'frame_id': frame,
                              'image': str(images[frame]), 'mask': str(masks[frame]),
                              'calibration': calibration[name]})
    if not pairs:
        raise ValueError('Selected split has no valid image/mask pairs')
    return pairs, missing


def canonical_source(subject: dict, destination: Path, selection: dict) -> dict:
    kind = subject.get('dataset')
    if kind == 'dna':
        from avatar_creation.methods.mmlphuman import stage_dna_subject
        from avatar_creation.methods.humangs_smc import HumanGSSMCReader
        data = stage_dna_subject({**subject, 'training_config': {**selection, 'resolution': 0},
                                  'evaluation_config': {}, 'evaluation_selection': True},
                                 destination, resolution=0, reader_class=HumanGSSMCReader, lossless=True)
        mapping = json.loads((data / 'camera_id_mapping.json').read_text())
    elif kind == 'actorshq':
        from avatar_creation.methods.actorshq_data import stage_actorshq_subject
        data = stage_actorshq_subject({**subject, 'training_config': {**selection, 'resolution': 0}},
                                     destination, resolution=0, reader_class=HumanGSSMCReader, lossless=True)
        mapping = json.loads((data / 'camera_id_mapping.json').read_text())
    else:
        return subject
    return {**subject, 'images': str(data / 'images'), 'masks': str(data / 'masks'),
            'calibration': str(data / 'calibration.json'), 'smplx': str(data / 'smpl_params.npz'),
            'staged_camera_mapping': mapping}


def prepare_subject(subject: dict, output: Path, template: dict, *, views=None,
                    time_steps=None, resolution=0, num_input_views=4,
                    num_target_views=4, image_size=512, train_steps=None, batch_size=1) -> Path:
    owner = output.resolve() / 'subjects' / safe_subject_id(subject['subject_id'])
    selection = copy.deepcopy(subject.get('training_config', {}))
    if not selection:
        if subject.get('dataset') in {'dna', 'actorshq'}:
            raise ValueError('HumanGS native inputs require an explicit --run-config')
        calibration = json.loads(Path(subject['calibration']).read_text())
        cameras = sorted(int(n[3:]) for n in calibration if n.startswith('cam') and n[3:].isdigit())
        cameras = cameras[:views] if views else cameras
        frames = sorted({frame for camera in Path(subject['images']).iterdir()
                         if camera.is_dir() for frame in media_files(camera)})
        selection = {'camera_ids': cameras, 'frame_range': {'start': 0, 'stop': time_steps or frames[-1] + 1},
                     'resolution': resolution}
    if 'camera_ids' not in selection:
        if subject.get('dataset') in {'dna', 'actorshq'}:
            raise ValueError('HumanGS native inputs require explicit training camera IDs')
        calibration = json.loads(Path(subject['calibration']).read_text())
        available = sorted(int(n[3:]) for n in calibration if n.startswith('cam') and n[3:].isdigit())
        count = selection.pop('views')
        if len(available) < count:
            raise ValueError(f'Requested {count} training cameras, found {len(available)}')
        selection['camera_ids'] = available[:count]
    if 'frame_range' not in selection:
        selection['frame_range'] = {'start': 0, 'stop': selection.pop('time_steps')}
    validate_selection(selection, 'train', training=True)
    if min(num_input_views, num_target_views, batch_size) < 1 or image_size < 8 or image_size % 8:
        raise ValueError('View counts and batch size must be positive; image size must be a multiple of 8')
    original = {**subject, 'training_config': selection}
    source = canonical_source(original, owner / 'data', selection)
    pairs, missing = observations(source, selection)
    if len(pairs) < num_input_views + num_target_views:
        raise ValueError('Insufficient distinct training observations for context and targets')
    owner.mkdir(parents=True, exist_ok=True)
    record = {'source': source, 'original_subject': original, 'training_config': selection,
              'training_pairs': pairs, 'missing_pairs': missing, 'upstream_revision': REVISION,
              'image_size': image_size, 'num_input_views': num_input_views,
              'num_target_views': num_target_views}
    record_path = owner / 'humangs_inputs.json'
    if record_path.is_file() and json.loads(record_path.read_text()) != record:
        raise ValueError(f'Input selection changed for existing run {owner}; use a new output directory')
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    write_evaluation_split(owner, original, train_cameras=selection['camera_ids'],
                           resolution=selection['resolution'])
    config = copy.deepcopy(template)
    config['model']['smpl_model_path'] = str(UPSTREAM / 'body_models/smplx/SMPLX_NEUTRAL.npz')
    config['model']['image_tokenizer']['image_size'] = image_size
    config['training'].update({'dataset_name': 'avatar_creation.methods.humangs_dataset.HumanGSDataset',
        'train_dataset_path': str(record_path), 'test_dataset_path': str(record_path),
        'checkpoint_dir': str(owner / 'model'), 'num_input_views': num_input_views,
        'num_target_views': num_target_views, 'num_views': num_input_views + num_target_views,
        'batch_size_per_gpu': batch_size, 'use_wandb': False, 'target_has_input': False,
        'dynamic_input_view_num': False, 'num_workers': 2, 'prefetch_factor': 2})
    if train_steps is not None:
        if train_steps < 1:
            raise ValueError('--train-steps must be positive')
        config['training']['train_steps'] = train_steps
        config['training']['warmup'] = min(config['training']['warmup'], max(0, train_steps // 10))
    config['inference']['if_inference'] = False
    import yaml
    config_path = owner / 'humangs.yaml'
    if config_path.is_file() and yaml.safe_load(config_path.read_text()) != config:
        raise ValueError(f'HumanGS config changed for existing run {owner}; use a new output directory')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def evaluation_plan(path: Path, split: str) -> dict:
    path = path.expanduser().resolve()
    start = path.parent if path.is_file() else path
    owner = next((p for p in (start, *start.parents) if (p / 'humangs_inputs.json').is_file()), None)
    if owner is None:
        raise ValueError(f'Cannot find HumanGS run metadata above {path}')
    record = json.loads((owner / 'humangs_inputs.json').read_text())
    metadata = json.loads((owner / 'evaluation_split.json').read_text())
    train = record['training_config']
    if metadata.get('training_config') != train:
        raise ValueError('Training split sidecar disagrees with recorded HumanGS inputs')
    name = split.replace('-', '_')
    selection = metadata.get('evaluation', {}).get(name)
    if selection is None:
        raise ValueError(f'No explicit {name} split recorded')
    validate_selection(selection, name)
    train_frames, frames = set(selection_frames(train)), set(selection_frames(selection))
    if name == 'novel_view':
        if set(selection['camera_ids']) & set(train['camera_ids']) or not frames <= train_frames:
            raise ValueError('Novel-view split leaks training cameras or uses held-out poses')
    elif name == 'novel_pose':
        if frames & train_frames:
            raise ValueError('Novel-pose split overlaps training frames')
    else:
        raise ValueError(f'Unknown split {split}')
    training_pairs = record['training_pairs']
    if any(p['camera_id'] not in train['camera_ids'] or p['frame_id'] not in train_frames for p in training_pairs):
        raise ValueError('Recorded context observations fall outside training selection')
    # Select distinct cameras first, then additional frames, deterministically.
    candidates = sorted(training_pairs, key=lambda p: (p['frame_id'], p['camera_id']))
    context = []
    for pair in candidates:
        if pair['camera_id'] not in {p['camera_id'] for p in context}:
            context.append(pair)
    count = record['num_input_views']
    if len(context) > count:
        # Spread context across the available camera IDs rather than selecting
        # only the first adjacent cameras in a dense multiview rig.
        context = [context[i * len(context) // count] for i in range(count)]
    context += [p for p in candidates if p not in context]
    context = context[:count]
    if len(context) != record['num_input_views']:
        raise ValueError('Not enough recorded context observations')
    checkpoints = sorted((owner / 'model').glob('ckpt_*.pt'))
    checkpoint = path if path.is_file() and path.suffix == '.pt' else (checkpoints[-1] if checkpoints else None)
    return {'owner': str(owner), 'split': name, 'checkpoint': str(checkpoint) if checkpoint else None,
            'context_pairs': context, 'context_policy': 'evenly spaced training camera IDs, earliest valid frames',
            'target_selection': selection, 'record': record,
            'protocol': {'background': 'black', 'mask': 'round(mask/255)',
                         'resolution': train['resolution'], 'context_image_size': record['image_size'],
                         'crop': False, 'undistort': False, 'lpips': 'vgg', 'fid_features': 2048}}


def input_signature(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def upstream_environment() -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('OMP_NUM_THREADS', '8')
    env['CUDNN_LIB_CONFIG'] = 'FULL'
    # cuDNN dlopens its component libraries; inherited system CUDA paths can
    # otherwise mix the wheel's main library with older system sublibraries.
    libraries = [str(Path(root) / 'nvidia/cudnn/lib') for root in site.getsitepackages()
                 if (Path(root) / 'nvidia/cudnn/lib').is_dir()]
    if libraries:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(libraries + [env.get('LD_LIBRARY_PATH', '')])
    return env


def warmup_cuda_convolution(device='cuda:0') -> None:
    """Initialize cuDNN before PyTorch3D/rasterizer CUDA kernels run."""
    import torch
    with torch.no_grad():
        image = torch.zeros(1, 3, 16, 16, device=device)
        weights = torch.zeros(4, 3, 3, 3, device=device)
        torch.nn.functional.conv2d(image, weights)
        torch.cuda.synchronize(device)


def ensure_runtime_libraries() -> None:
    env = upstream_environment()
    desired = env.get('LD_LIBRARY_PATH', '')
    first = desired.split(os.pathsep)[0]
    if (os.environ.get('CUDNN_LIB_CONFIG') != 'FULL'
            or (first and os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)[0] != first)):
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--views', type=int)
    parser.add_argument('--time-steps', type=int)
    parser.add_argument('--resolution', type=int, default=0)
    parser.add_argument('--config', type=Path, default=UPSTREAM / 'configs/HumanGS_config.yaml')
    parser.add_argument('--num-input-views', type=int, default=4)
    parser.add_argument('--num-target-views', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--train-steps', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    import yaml
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    template = yaml.safe_load(config_path.read_text())
    for subject in load_subjects(args.dataset_manifest):
        config = prepare_subject(subject, args.output_dir, template, views=args.views,
            time_steps=args.time_steps, resolution=args.resolution, num_input_views=args.num_input_views,
            num_target_views=args.num_target_views, image_size=args.image_size,
            train_steps=args.train_steps, batch_size=args.batch_size)
        if not args.prepare_only:
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=1', str(PROJECT_ROOT / 'avatar_creation/methods/humangs_runtime.py'),
                '--config', str(config)], cwd=UPSTREAM, env=upstream_environment(), check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
