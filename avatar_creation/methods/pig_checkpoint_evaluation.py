"""Reconnect standalone upstream checkpoint evaluation to the recorded VCI split.

This module is loaded by file path because PiG owns the ``src`` package.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import torch


_LEGACY_EMPTY_STATE_KEYS = frozenset({
    'splat_model.binder.surface_vertices',
    'splat_model.binder.surface_faces',
    'splat_model.binder.surface_adjacency',
    'splat_model.binder.initial_face_ids',
    'splat_model.anchors.initial_positions',
})


def _load_local(name):
    spec = importlib.util.spec_from_file_location(f'vci_{name}', Path(__file__).with_name(f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_path(path):
    path = Path(path)
    return path / 'avatar_model.pth' if path.suffix == '' else path


def _local_smplx_model(saved_path):
    """Relocate a machine-specific SMPL-X path embedded in a checkpoint."""
    saved = Path(str(saved_path)).expanduser()
    if saved.is_file():
        return saved.resolve()

    root = Path(__file__).resolve().parents[2]
    override = os.environ.get('VCI_SMPLX_MODEL')
    candidates = [
        Path(override).expanduser() if override else None,
        root / 'external/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz',
        root / 'external/neu-pig-smplx/models/smplx/SMPLX_NEUTRAL.npz',
    ]
    return next((candidate.resolve() for candidate in candidates
                 if candidate is not None and candidate.is_file()), saved)


def _relocate_checkpoint_model_args(checkpoint, supplied=None, supplied_args=None):
    """Return model args with unavailable saved asset paths made local."""
    if supplied is None:
        payload = torch.load(_checkpoint_path(checkpoint), map_location='cpu', weights_only=False)
        supplied = payload.get('model_args')
        if supplied is None:
            args = supplied_args or payload.get('args')
            supplied = getattr(args, 'method_args', {}).get('model_args') if args is not None else None
    if supplied is None:
        return None

    model_args = copy.deepcopy(supplied)
    smplx_args = model_args.get('model_args_smplx', {})
    if smplx_args.get('model_path'):
        smplx_args['model_path'] = str(_local_smplx_model(smplx_args['model_path']))
    return model_args


def _without_legacy_empty_state(state_dict):
    obsolete = {key for key in _LEGACY_EMPTY_STATE_KEYS
                if key in state_dict and state_dict[key].numel() == 0}
    if not obsolete:
        return state_dict
    cleaned = copy.copy(state_dict)
    for key in obsolete:
        cleaned.pop(key)
    return cleaned


def subject_for_checkpoint(checkpoint, args):
    checkpoint = Path(checkpoint).resolve()
    owner = checkpoint.parent.parent if checkpoint.parent.name == 'model' else checkpoint.parent
    embedded = getattr(args, 'vci_subject', None)
    path = owner / 'evaluation_split.json'
    if embedded is not None:
        subject = copy.deepcopy(embedded)
    else:
        if not path.is_file():
            raise ValueError(f'Missing recorded VCI split: {path}; cannot infer camera or frame IDs')
        metadata = json.loads(path.read_text())
        subject = {**metadata['original_data'], 'subject_id': metadata['subject_id'],
                   'dataset': metadata['dataset_kind'], 'evaluation_config': metadata.get('evaluation', {})}
        train = metadata.get('training_config')
        if train is None:
            frames = metadata.get('train_frame_ids', [])
            cameras = metadata.get('train_source_camera_ids', metadata.get('train_cameras'))
            if not frames or not cameras:
                raise ValueError('Recorded split lacks explicit training camera/frame IDs')
            step = frames[1] - frames[0] if len(frames) > 1 else 1
            if step < 1 or frames != list(range(frames[0], frames[-1] + 1, step)):
                raise ValueError('Recorded training frames must form an increasing regular range')
            train = {'camera_ids': cameras,
                     'frame_range': {'start': frames[0], 'stop': frames[-1] + 1, 'step': step},
                     'resolution': 0 if metadata['resolution'] == 'source' else metadata['resolution']}
        subject['training_config'] = train
    train = subject['training_config']
    config = _load_local('training_inputs')
    if 'camera_ids' not in train:
        raise ValueError('Checkpoint evaluation requires recorded source camera IDs')
    targets = args.method_args['target']
    if targets['views'] != len(train['camera_ids']) or targets['time_steps'] != len(config.selection_frames(train)):
        raise ValueError('Recorded split camera/frame counts disagree with the checkpoint')
    if str(args.io_args['directory']) != str(subject['subject_id']):
        raise ValueError('Recorded split subject disagrees with the checkpoint')
    if not subject.get('evaluation_config'):
        raise ValueError('No explicit evaluation splits recorded for this checkpoint')
    args.io_args['out_path'] = str(owner)
    args.io_args['input_directory'] = subject['subject_path']
    args.io_args['directory_path'] = subject['subject_path']
    return subject


def install_checkpoint_evaluation_hooks(model_class, evaluation_module, pose_module, runtime=None):
    """Install once before running the unchanged upstream checkpoint CLI module."""
    runtime = runtime or _load_local('pig_avatar_run')
    original = model_class.from_checkpoint

    original_state_loader = getattr(model_class, 'load_state_dict', None)
    if original_state_loader is not None:
        def load_state_dict(self, state_dict, *positional, **kwargs):
            return original_state_loader(
                self, _without_legacy_empty_state(state_dict), *positional, **kwargs)
        model_class.load_state_dict = load_state_dict

    def from_checkpoint(cls, checkpoint, *positional, **kwargs):
        positional = list(positional)
        # AvatarModel.from_checkpoint accepts model_args as its sixth argument.
        # Supply a relocated copy before AvatarModel constructs SMPLX_Model_Predefined.
        supplied_args = kwargs.get('args')
        if supplied_args is None and len(positional) > 3:
            supplied_args = positional[3]
        if len(positional) > 4:
            positional[4] = _relocate_checkpoint_model_args(
                checkpoint, positional[4], supplied_args)
        else:
            relocated = _relocate_checkpoint_model_args(
                checkpoint, kwargs.get('model_args'), supplied_args)
            if relocated is not None:
                kwargs['model_args'] = relocated
        model = original(checkpoint, *positional, **kwargs)
        model.args.vci_evaluation_subject = subject_for_checkpoint(checkpoint, model.args)
        return model

    def novel_dataset(args, kind):
        subject = args.vci_evaluation_subject
        dataset = runtime.configured_novel_dataset(args, subject, kind)
        reference = {
            'subject_id': subject['subject_id'], 'split': f'novel_{kind}',
            'source_camera_ids': dataset.camera_ids, 'source_frame_ids': dataset.frame_ids,
            'training_config': subject['training_config'],
            'original_data': {key: subject[key] for key in ('images', 'masks', 'calibration', 'smplx')},
        }
        root = Path(args.io_args['out_path']) / 'evaluation'
        root.mkdir(parents=True, exist_ok=True)
        (root / f'input_reference_novel_{kind}.json').write_text(json.dumps(reference, indent=2, default=str) + '\n')
        return dataset

    model_class.from_checkpoint = classmethod(from_checkpoint)
    evaluation_module._novel_dataset = novel_dataset
    pose_module._novel_pose_dataset = lambda args: novel_dataset(args, 'pose')
