"""Strict HumanGS samples with source-indexed SMPL-X and square padded media."""
from __future__ import annotations
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_image(pair: dict, *, image_size: int | None, resolution: int = 0,
               square: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(Image.open(pair['image']).convert('RGB')).copy()
    mask = np.asarray(Image.open(pair['mask']))
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    if mask.shape != image.shape[:2]:
        raise ValueError(f'Mask dimensions disagree with image: {pair["mask"]}')
    if not square:
        from scripts.evaluation.eval_mmlphuman import prepare_target
        target, K = prepare_target(torch.from_numpy(image), torch.from_numpy(mask.copy())[..., None],
                                   pair['calibration'], resolution)
        return target.permute(2, 0, 1).float().contiguous() / 255, torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    mask = np.round(mask.astype(np.float32) / 255)[..., None]
    # Match the shared binary mask protocol, black background.
    image = image * mask
    height, width = image.shape[:2]
    K = np.asarray(pair['calibration']['K'], dtype=np.float32).reshape(3, 3).copy()
    if square:
        side = max(height, width)
        top, left = (side - height) // 2, (side - width) // 2
        padded = np.zeros((side, side, 3), dtype=np.float32)
        padded[top:top + height, left:left + width] = image
        image = padded
        K[0, 2] += left
        K[1, 2] += top
        height = width = side
        new_h = new_w = image_size
    else:
        scale = min(resolution / max(height, width), 1.) if resolution else 1.
        new_h, new_w = max(1, int(height * scale)), max(1, int(width * scale))
    tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255
    tensor = torch.nn.functional.interpolate(tensor[None], size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
    K[0] *= new_w / width
    K[1] *= new_h / height
    return tensor, torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])


def prepare_poses(source: dict, frame_ids: list[int], directory: Path, model_path: Path) -> None:
    """Generate metric-space vertices and rest-to-world LBS transforms on CPU."""
    import smplx
    from smplx.lbs import batch_rodrigues, batch_rigid_transform
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(source['smplx'], allow_pickle=False) as archive:
        params = {key: np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    if max(frame_ids) >= len(params['body_pose']):
        raise ValueError('SMPL-X parameters do not cover the requested source frame IDs')
    model = smplx.SMPLX(str(model_path), gender='neutral', use_pca=False,
                        flat_hand_mean=True, num_betas=10).eval()
    dimensions = {'global_orient': 3, 'body_pose': 63, 'jaw_pose': 3, 'leye_pose': 3,
                  'reye_pose': 3, 'left_hand_pose': 45, 'right_hand_pose': 45,
                  'expression': 10, 'transl': 3}
    def parameter(key, frame):
        if key not in params:
            return torch.zeros(1, dimensions[key])
        values = params[key]
        row = values if values.ndim == 1 else values[frame if len(values) > 1 else 0]
        row = row.reshape(-1)
        if row.size < dimensions[key]:
            raise ValueError(f'{key} requires {dimensions[key]} axis-angle values; PCA hand poses are unsupported')
        return torch.from_numpy(row[:dimensions[key]].copy())[None]
    with torch.no_grad():
        for frame in frame_ids:
            # No stale cache reuse: regenerate from the currently referenced source poses.
            betas = params['betas']
            row = betas if betas.ndim == 1 else betas[frame if len(betas) > 1 else 0]
            betas_tensor = torch.from_numpy(row.reshape(-1)[:10].copy())[None]
            zeros = {key: torch.zeros(1, dim) for key, dim in dimensions.items()}
            canonical = model(betas=betas_tensor, **zeros)
            posed = model(betas=betas_tensor, **{key: parameter(key, frame) for key in dimensions}, return_full_pose=True)
            rotation = batch_rodrigues(posed.full_pose.reshape(-1, 3)).reshape(1, 55, 3, 3)
            _, transforms = batch_rigid_transform(rotation, canonical.joints[:, :55], model.parents)
            transforms[:, :, :3, 3] += parameter('transl', frame)[:, None]
            np.savez(directory / f'{frame:08d}.npz', v_posed=posed.vertices[0].numpy(),
                     v_cano=canonical.vertices[0].numpy(), faces=model.faces,
                     joint_transforms=transforms[0].numpy())


def sample(pairs: list[dict], pose_directory: Path, *, image_size: int,
           resolution: int = 0, target_native: bool = False) -> dict:
    images, intrinsics, cameras, posed, canonical, faces, transforms = [], [], [], [], [], [], []
    for index, pair in enumerate(pairs):
        image, intrinsic = load_image(pair, image_size=image_size, resolution=resolution,
                                      square=not (target_native and index == len(pairs) - 1))
        # Evaluation puts targets on the native canvas after ProcessData splits views.
        if target_native and index == len(pairs) - 1:
            image = torch.zeros_like(images[0])
        images.append(image)
        intrinsics.append(intrinsic)
        cam = pair['calibration']
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = np.asarray(cam['R']).reshape(3, 3)
        w2c[:3, 3] = np.asarray(cam['T']).reshape(3)
        cameras.append(torch.from_numpy(np.linalg.inv(w2c)))
        with np.load(pose_directory / f'{pair["frame_id"]:08d}.npz') as data:
            posed.append(torch.from_numpy(data['v_posed']).float())
            canonical.append(torch.from_numpy(data['v_cano']).float())
            faces.append(torch.from_numpy(data['faces'].astype(np.int64)))
            transforms.append(torch.from_numpy(data['joint_transforms']).float())
    return {'image': torch.stack(images), 'fxfycxcy': torch.stack(intrinsics),
            'c2w': torch.stack(cameras), 'smpl_posed_vertices': torch.stack(posed),
            'smpl_canonical_vertices': torch.stack(canonical), 'smpl_faces': torch.stack(faces),
            'joint_transforms': torch.stack(transforms),
            'index': torch.tensor([[p['camera_id'], p['frame_id']] for p in pairs]),
            'scene_name': 'vci'}


class HumanGSDataset(Dataset):
    def __init__(self, config):
        self.config = config
        record_path = Path(config.training.train_dataset_path)
        self.record = json.loads(record_path.read_text())
        self.pairs = self.record['training_pairs']
        self.pose_directory = record_path.parent / 'poses'
        self.count = config.training.num_input_views + config.training.num_target_views
        if len(self.pairs) < self.count:
            raise ValueError('Insufficient distinct training pairs')
        # The launch wrapper generates poses once before workers start.
        if any(not (self.pose_directory / f'{p["frame_id"]:08d}.npz').is_file() for p in self.pairs):
            raise ValueError('Missing generated training poses; launch through the VCI wrapper')

    def __len__(self):
        return max(1000, len(self.pairs))

    def __getitem__(self, index):
        pairs = random.sample(self.pairs, self.count)
        return sample(pairs, self.pose_directory, image_size=self.record['image_size'])
