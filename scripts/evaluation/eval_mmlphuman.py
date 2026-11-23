#!/usr/bin/env python3
"""Evaluate recorded MMLP splits with PiG-Avatar's image and metric protocol."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def checkpoint_for(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = []
    for directory in (path, path / 'model'):
        if directory.is_dir():
            for candidate in directory.iterdir():
                match = re.fullmatch(r'chkpnt(\d+)\.pth', candidate.name)
                if match and candidate.is_file():
                    candidates.append((int(match[1]), candidate))
    if not candidates:
        raise ValueError(f'No chkpnt*.pth checkpoint found under {path}')
    return max(candidates)[1]


def lod_fraction_for_checkpoint(checkpoint_data, requested):
    """Validate an optional LOD fraction against the checkpoint format."""
    input_width = checkpoint_data['encoder_feat_params']['layers.0.weight'].shape[-1]
    lod = checkpoint_data.get('lod')
    if input_width == 63 and lod is None:
        if requested is not None:
            raise ValueError('--lod-fraction requires an MMLPHuman LOD checkpoint')
        return None
    if input_width != 64 or not isinstance(lod, dict) or lod.get('model_type') != 'mmlphuman-lod' or lod.get('version') != 1:
        raise ValueError(f'Unsupported MMLPHuman checkpoint encoder/LOD format: input width {input_width}, LOD {lod is not None}')
    config = lod['config']
    minimum, maximum = float(config['min_fraction']), float(config['max_fraction'])
    fraction = maximum if requested is None else requested
    if not 0 < minimum <= fraction <= maximum <= 1:
        raise ValueError(f'LOD fraction must be between {minimum} and {maximum}')
    return fraction


def lod_indices_for_checkpoint(checkpoint_data, fraction, device):
    """Select the saved prefix of Gaussians at the requested LOD."""
    import torch

    permutation = checkpoint_data['lod']['permutation']
    count = len(checkpoint_data['_xyz'])
    if permutation.ndim != 1 or len(permutation) != count:
        raise ValueError('LOD permutation length does not match the checkpoint Gaussians')
    selected = max(1, int(count * fraction))
    return permutation[:selected].to(device=device, dtype=torch.long)


def model_class_for_checkpoint(base_class, checkpoint_data, lod_fraction=None):
    """Append the requested LOD condition to the pose encoder input."""
    import torch

    fraction = lod_fraction_for_checkpoint(checkpoint_data, lod_fraction)
    if fraction is None:
        return base_class

    class LODModel(base_class):
        @property
        def get_joint_features(self):
            pose = super().get_joint_features
            return torch.cat((pose, pose.new_tensor([fraction])))

    return LODModel


def render_lod(model, cam, indices, background):
    """Render all Gaussians or a saved LOD prefix with PiG-Avatar's raster settings."""
    import torch
    from gsplat import rasterization

    indices = slice(None) if indices is None else indices
    cam_pos = torch.linalg.inv_ex(cam['w2c'])[0][:3, 3]
    image, alpha, info = rasterization(
        means=model.get_xyz[indices],
        quats=None,
        scales=None,
        opacities=model.get_opacity[indices],
        colors=model.get_color(cam_pos)[indices],
        viewmats=cam['w2c'][None],
        Ks=cam['K'][None],
        width=cam['width'],
        height=cam['height'],
        packed=False,
        near_plane=0.01,
        render_mode='RGB',
        rasterize_mode='antialiased',
        backgrounds=background[None],
        covars=model.get_covariance(1.0)[indices],
    )
    return image[0], alpha[0], info


def media_index(directory: Path) -> dict[int, Path]:
    result = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png'} and path.stem.isdigit():
                frame = int(path.stem)
                if frame in result:
                    raise ValueError(f'Duplicate frame {frame} in {directory}')
                result[frame] = path
    return result


def dna_source(source, staged_names):
    """Index native SMC images without decoding or staging entire sequences."""
    import h5py
    import numpy as np
    calibration, indices, camera_groups = {}, {}, {}
    with h5py.File(source['images'], 'r') as rgb, h5py.File(source['masks'], 'r') as masks, h5py.File(source['calibration'], 'r') as annotation:
        five_mp_ids = sorted(int(key) for key in rgb.get('Camera_5mp', {}) if key.isdigit())
        if len(staged_names) > len(five_mp_ids):
            raise ValueError('Staged DNA camera count exceeds source 5MP cameras')
        # stage_dna_subject enumerates sorted 5MP IDs into cam00, cam01, ...
        mapping = {name: f'cam{camera_id:02d}' for name, camera_id in zip(staged_names, five_mp_ids)}
        for group in ('Camera_5mp', 'Camera_12mp'):
            for raw_id in rgb.get(group, {}):
                if not raw_id.isdigit():
                    continue
                name = f'cam{int(raw_id):02d}'
                image_key = f'{group}/{raw_id}/color'
                mask_key = f'Mask/{int(raw_id)}/mask'
                calibration_key = f'Camera_Parameter/{int(raw_id):02d}'
                if image_key not in rgb or mask_key not in masks or calibration_key not in annotation:
                    continue
                cam = annotation[calibration_key]
                w2c = np.linalg.inv(np.asarray(cam['RT'], dtype=np.float32))
                calibration[name] = {'K': cam['K'][()].reshape(-1).tolist(), 'R': w2c[:3, :3].reshape(-1).tolist(), 'T': w2c[:3, 3].tolist(), 'distCoeff': cam['D'][()].reshape(-1).tolist()}
                indices[name] = (
                    {int(key): f'{image_key}/{key}' for key in rgb[image_key] if key.isdigit()},
                    {int(key): f'{mask_key}/{key}' for key in masks[mask_key] if key.isdigit()},
                )
                camera_groups[name] = group
    return calibration, indices, mapping, camera_groups


def read_pair(image_path, mask_path, readers=None):
    """Decode RGB and masks exactly as PiG-Avatar's THuman/DNA loaders do."""
    import cv2
    import numpy as np
    if readers is None:
        from torchvision.io import read_image
        return (read_image(str(image_path)).permute(1, 2, 0),
                read_image(str(mask_path)).permute(1, 2, 0))
    import torch
    image = cv2.imdecode(np.asarray(readers[0][image_path], dtype=np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.asarray(readers[1][mask_path], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or mask is None:
        raise ValueError(f'Cannot decode {image_path} or {mask_path}')
    return (torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            torch.from_numpy(mask.max(axis=-1))[..., None])


def prepare_target(rgb, mask, camera, resolution=0):
    """Match BaseDataset resizing and _prepare_eval_data; do not undistort."""
    import torch
    import torch.nn.functional as F

    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError('Image/mask dimensions differ')
    height, width = rgb.shape[:2]
    scale = min(1., resolution / max(height, width)) if resolution else 1.
    size = (int(height * scale), int(width * scale))
    if min(size) < 1:
        raise ValueError('Evaluation resolution produces an empty image dimension')
    if size != (height, width):
        def resize(value):
            return F.interpolate(value.permute(2, 0, 1)[None], size=size,
                                 mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        rgb, mask = resize(rgb), resize(mask)
    K = torch.tensor(camera['K'], dtype=torch.float32).reshape(3, 3)
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    return rgb * (mask / 255).round().to(torch.uint8), K


def create_lpips_model(device):
    """Use the same package, backbone, and weights as PiG-Avatar."""
    import lpips

    model = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()
    model.requires_grad_(False)
    return model


def eval_images(pred, target, fid, lpips_model):
    """PiG-Avatar's full-image metrics for one HWC image pair in [0, 1]."""
    import torch
    from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

    pred = pred[None].clamp(0, 1).permute(0, 3, 1, 2)
    target = target[None].clamp(0, 1).permute(0, 3, 1, 2)
    values = {
        'psnr': peak_signal_noise_ratio(pred, target, data_range=1.).item(),
        'ssim': structural_similarity_index_measure(pred, target, data_range=1.).item(),
        'lpips': lpips_model(pred * 2 - 1, target * 2 - 1).mean().item(),
    }
    fid.update((target * 255).to(torch.uint8), real=True)
    fid.update((pred * 255).to(torch.uint8), real=False)
    return values


def evaluation_plan(checkpoint: Path, split: str, cameras=None, resolution=None, frames=None):
    import yaml
    owner = checkpoint.parent.parent if checkpoint.parent.name == 'model' else checkpoint.parent
    metadata = json.loads((owner / 'evaluation_split.json').read_text())
    config = yaml.safe_load((owner / 'mmlphuman.yaml').read_text())
    source = metadata['original_data']
    staged_calibration = json.loads((owner / 'data/calibration.json').read_text())
    staged_names = list(staged_calibration)
    native_dna = metadata.get("dataset_kind") == "dna"
    native_actorshq = metadata.get("dataset_kind") == "actorshq"
    indices, camera_groups = {}, {}
    if native_dna:
        calibration, indices, mapping, camera_groups = dna_source(source, staged_names)
        mapping_path = owner / 'data/camera_id_mapping.json'
        if mapping_path.is_file():
            mapping = {name: f'cam{int(camera):02d}' for name, camera in json.loads(mapping_path.read_text()).items()}
        train = {mapping[staged_names[int(i)]] for i in config['train_cam_ids']}
        available = set(indices)
    else:
        if not Path(source['images']).is_dir() or not Path(source['masks']).is_dir():
            raise ValueError('Evaluation requires canonical image/mask directories or native DNA SMC files.')
        calibration = json.loads(Path(source['calibration']).read_text())
        if native_actorshq:
            mapping_path = owner / "data/camera_id_mapping.json"
            mapping = json.loads(mapping_path.read_text())
            train_source_ids = {int(mapping[staged_names[int(i)]]) for i in config["train_cam_ids"]}
            train = {f"cam{camera_id:02d}" for camera_id in train_source_ids}
        else:
            train = {staged_names[int(i)] for i in config["train_cam_ids"]}
            train_source_ids = {int(name[3:]) for name in train}
        available = {
            name for name in calibration
            if (Path(source['images']) / name).is_dir() and (Path(source['masks']) / name).is_dir()
        }
    recorded_cameras = metadata.get('train_source_camera_ids')
    if recorded_cameras is not None and set(recorded_cameras) != (train_source_ids if native_actorshq else {int(name[3:]) for name in train}):
        raise ValueError('Recorded training cameras disagree with staged camera mapping')
    configured = metadata.get('evaluation', {})
    split_name = {'heldout': 'novel_view', 'novel-view': 'novel_view', 'novel-pose': 'novel_pose'}.get(split)
    selection = configured.get(split_name)
    if split == 'novel-pose' and selection is None:
        raise ValueError('No novel_pose split configured for this run')
    if configured and split_name and selection is None:
        raise ValueError(f'No {split_name} split configured for this run')
    if selection:
        names_by_id = {int(name[3:]): name for name in available}
        missing = set(selection['camera_ids']) - set(names_by_id)
        if missing:
            raise ValueError(f'Configured evaluation cameras unavailable: {sorted(missing)}')
        allowed = {names_by_id[i] for i in selection['camera_ids']}
        if split_name == 'novel_view' and allowed & train:
            raise ValueError('Novel-view cameras overlap training cameras')
    else:
        allowed = available - train if split in ('heldout', 'novel-view') else available & train
    selected = sorted(allowed) if cameras is None else list(dict.fromkeys(cameras))
    if not selected:
        raise ValueError('No cameras available for this split. For a run trained on all cameras, use --split train for reconstruction metrics.')
    if set(selected) - allowed:
        raise ValueError(f'Cameras outside the {split} split: {sorted(set(selected) - allowed)}')
    begin = int(config['begin_ith_frame'])
    interval = int(config['frame_interval'])
    training_frames = list(range(begin, begin + int(config['num_train_frame']) * interval, interval))
    if not training_frames or interval < 1:
        raise ValueError('Invalid training frame range')
    if metadata.get('train_frame_ids', training_frames) != training_frames:
        raise ValueError('Recorded training frames disagree with mmlphuman.yaml')
    allowed_frames = training_frames
    if selection:
        r = selection['frame_range']
        allowed_frames = list(range(r['start'], r['stop'], r.get('step', 1)))
        overlap = set(allowed_frames) & set(training_frames)
        if split_name == 'novel_pose' and overlap:
            raise ValueError('Novel-pose frames overlap training frames')
        if split_name == 'novel_view' and set(allowed_frames) - set(training_frames):
            raise ValueError('Novel-view frames must belong to training frames')
    frames = allowed_frames if frames is None else list(dict.fromkeys(frames))
    if not frames or set(frames) - set(allowed_frames):
        raise ValueError('Evaluation frames must belong to the selected split frame range' if selection else 'Evaluation frames must belong to the training frame range')
    pairs, missing = [], []
    for camera in selected:
        if native_dna:
            images, masks = indices[camera]
        else:
            images = media_index(Path(source['images']) / camera)
            masks = media_index(Path(source['masks']) / camera)
        for frame in frames:
            if frame in images and frame in masks:
                pairs.append((camera, frame, images[frame], masks[frame]))
            else:
                missing.append((camera, frame))
    if not pairs:
        raise ValueError('No valid image/mask pairs in the evaluation split')
    size = 0 if resolution is None else resolution
    size = 0 if size == 'source' else int(size)
    if size < 0:
        raise ValueError('Resolution must be non-negative')
    return owner, calibration, pairs, {
        'checkpoint': str(checkpoint), 'split': split, 'train_cameras': sorted(train),
        'evaluation_cameras': selected, 'frames': frames, 'resolution': size,
        'background': 'black', 'missing_pairs': missing,
        'source_format': 'dna_smc' if native_dna else 'directories',
        'camera_groups': camera_groups,
        'original_data': source, 'pose_source': str(owner / 'data/smpl_params.npz'),
        'protocol': 'neupig', 'undistort': False, 'mask_threshold': 'round(mask / 255)',
        'lpips': 'lpips-vgg', 'fid_features': 2048,
        'rasterize_mode': 'antialiased', 'near_plane': 0.01,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmlp-path', type=Path, required=True)
    parser.add_argument('--split', choices=['heldout', 'train', 'novel-view', 'novel-pose'], default='heldout')
    parser.add_argument('--cameras', nargs='+', help='Exact source camera names, e.g. cam23')
    parser.add_argument('--frames', nargs='+', type=int, help='Subset of configured split frame IDs to evaluate')
    parser.add_argument('--resolution', type=int, help='Maximum dimension; default: source resolution, matching direct PiG-Avatar evaluation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lod-fraction', type=float, help='Fraction of Gaussians for an MMLPHuman LOD checkpoint; default: checkpoint maximum')
    parser.add_argument('--output-dir', type=Path, help='Default: <subject>/evaluation/<split>')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--save-images', action=argparse.BooleanOptionalAction, default=True,
                        help='Save targets and white/black renders (default: enabled)')
    parser.add_argument('--fid', action='store_true', help='Accepted for compatibility; FID is always computed')
    args = parser.parse_args(argv)
    checkpoint = checkpoint_for(args.mmlp_path)
    owner, calibration, pairs, audit = evaluation_plan(checkpoint, args.split, args.cameras, args.resolution, args.frames)
    import numpy as np
    with np.load(audit['pose_source']) as data:
        poses = {key: data[key] for key in ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl')}
    for key, values in poses.items():
        if min(audit['frames']) < 0 or max(audit['frames']) >= len(values):
            raise ValueError(f'Evaluation frame range exceeds staged {key} poses')
    print(f"Evaluation: {len(pairs)} pairs, {len(audit['evaluation_cameras'])} {args.split} cameras; {len(audit['missing_pairs'])} missing pairs skipped")
    if args.validate_only:
        print(json.dumps(audit, indent=2))
        return 0
    if len(pairs) < 2:
        raise ValueError('FID requires at least two evaluation pairs')

    import cv2
    import torch
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('MMLP rendering requires --device cuda:N')
    torch.cuda.set_device(device)
    device = torch.device('cuda', torch.cuda.current_device())
    upstream = Path(__file__).resolve().parents[2] / 'external/mmlphuman'
    sys.path.insert(0, str(upstream))
    from scene.gaussian_model import GaussianModel
    from utils.smpl_utils import init_smpl_pose
    init_smpl_pose()
    # Preserve CPU poses/joints while remapping saved CUDA tensors.
    def remap(storage, location):
        return storage.cuda(device.index) if location.startswith('cuda') else storage
    checkpoint_data = torch.load(checkpoint, map_location=remap, weights_only=False)
    lod_fraction = lod_fraction_for_checkpoint(checkpoint_data, args.lod_fraction)
    model = model_class_for_checkpoint(GaussianModel, checkpoint_data, lod_fraction)()
    model.restore(checkpoint_data)
    lod_indices = None if lod_fraction is None else lod_indices_for_checkpoint(checkpoint_data, lod_fraction, device)
    if lod_fraction is not None:
        audit['lod_fraction'] = lod_fraction
        audit['lod_gaussians'] = len(lod_indices)
    model.is_test = False
    lpips_model = create_lpips_model(device)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(device)
    default_output = owner / 'evaluation' / args.split
    if args.lod_fraction is not None:
        default_output /= f'lod-{args.lod_fraction:g}'
    output = (args.output_dir or default_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'evaluation_reference.json').write_text(json.dumps(audit, indent=2) + '\n')
    rows = []
    audit['empty_pairs'] = []
    from contextlib import ExitStack
    with ExitStack() as stack, torch.inference_mode():
        readers = None
        if audit['source_format'] == 'dna_smc':
            import h5py
            readers = tuple(stack.enter_context(h5py.File(audit['original_data'][key], 'r')) for key in ('images', 'masks'))
        for index, (camera, frame, image_path, mask_path) in enumerate(pairs):
            rgb, mask = read_pair(image_path, mask_path, readers)
            cam = calibration[camera]
            target_u8, K = prepare_target(rgb, mask, cam, audit['resolution'])
            if not (target_u8 >= 0.05).any():
                audit['empty_pairs'].append([camera, frame])
                print(f'Skipping empty frame {camera}/{frame}', flush=True)
                continue
            height, width = target_u8.shape[:2]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = np.array(cam['R']).reshape(3, 3)
            w2c[:3, 3] = np.array(cam['T']).reshape(3)
            pose = np.concatenate([poses['global_orient'][frame].ravel(), poses['body_pose'][frame].ravel(), np.zeros(9), poses['left_hand_pose'][frame].ravel(), poses['right_hand_pose'][frame].ravel()])
            # Match upstream training's pose conversion, including zero face pose.
            model.smpl_poses = torch.tensor(pose, dtype=torch.float32)
            model.Th = torch.tensor(poses['transl'][frame], dtype=torch.float32)
            model.Rh = torch.eye(3)
            render_camera = {'K': K.to(device), 'w2c': torch.tensor(w2c, device=device), 'width': width, 'height': height}
            def save_image(kind, value):
                dest = output / 'renders' / camera / kind
                dest.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest / f'{frame:08d}.png'),
                            cv2.cvtColor(value.to(torch.uint8).cpu().numpy(), cv2.COLOR_RGB2BGR))
            if args.save_images:
                save_image('target', target_u8)
            for background_name, background_value in (('white', 1.), ('black', 0.)):
                background = torch.full((3,), background_value, device=device)
                image, _, _ = render_lod(model, render_camera, lod_indices, background)
                image = image.clamp(0, 1)
                if args.save_images:
                    save_image(background_name, image * 255)
                if background_name == 'black':
                    values = eval_images(image, target_u8.to(device=device, dtype=torch.float32) / 255,
                                         fid, lpips_model)
                    rows.append({'camera': camera, 'frame': frame, **values})
            print(f'[{index + 1}/{len(pairs)}] {camera}/{frame}: {values}', flush=True)
    (output / 'evaluation_reference.json').write_text(json.dumps(audit, indent=2) + '\n')
    if not rows:
        raise RuntimeError('Evaluation produced no valid frames')
    if len(rows) < 2:
        raise ValueError('FID requires at least two nonempty evaluation pairs')
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in ('psnr', 'ssim', 'lpips')}
    metrics['fid'] = fid.compute().item()
    (output / 'metrics.json').write_text(json.dumps({'split': args.split, 'pairs': len(rows), **metrics}, indent=2) + '\n')
    (output / 'per_frame_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Results: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
