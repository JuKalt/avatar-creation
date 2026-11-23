#!/usr/bin/env python3
"""Evaluate HumanGS with fixed training-only context and recorded VCI target splits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from avatar_creation.methods.humangs import UPSTREAM, canonical_source, evaluation_plan, observations, input_signature, ensure_runtime_libraries, warmup_cuda_convolution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--split', choices=['novel-view', 'novel-pose'], required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-save-images', action='store_true')
    args = parser.parse_args(argv)
    plan = evaluation_plan(args.path, args.split)
    owner = Path(plan['owner'])
    output = (args.output_dir or owner / 'evaluation' / plan['split']).resolve()
    subject = plan['record']['original_subject']
    if args.validate_only:
        if subject.get('dataset') not in {'dna', 'actorshq'}:
            pairs, missing = observations(subject, plan['target_selection'])
            plan.update(target_pairs=pairs, missing_pairs=missing)
        print(json.dumps(plan, indent=2))
        return 0
    source = canonical_source(subject, owner / 'evaluation_inputs' / plan['split'], plan['target_selection'])
    pairs, missing = observations(source, plan['target_selection'])
    plan.update(target_pairs=pairs, missing_pairs=missing)
    output.mkdir(parents=True, exist_ok=True)
    reference = output / 'evaluation_reference.json'
    reference.write_text(json.dumps(plan, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return 0
    if plan['checkpoint'] is None:
        raise ValueError('No HumanGS checkpoint found; train first or pass a checkpoint file')
    import torch
    import torch.distributed as dist
    from easydict import EasyDict
    import yaml
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)
    from avatar_creation.methods.humangs_dataset import prepare_poses, sample, load_image
    from avatar_creation.methods.humangs_runtime import install_black_renderer
    from scripts.evaluation.eval_mmlphuman import eval_images, create_lpips_model
    from torchmetrics.image.fid import FrechetInceptionDistance
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    warmup_cuda_convolution(device)
    config = EasyDict(yaml.safe_load((owner / 'humangs.yaml').read_text()))
    if config.training.num_input_views != plan['record']['num_input_views']:
        raise ValueError('Saved config context count disagrees with recorded inputs')
    config.training.num_target_views = 1
    config.training.num_views = config.training.num_input_views + 1
    config.training.target_has_input = False
    config.training.lpips_loss_weight = 0.
    config.training.perceptual_loss_weight = 0.
    config.inference.if_inference = True
    poses = owner / 'evaluation_poses' / plan['split']
    prepare_poses(source, sorted({p['frame_id'] for p in pairs}), poses, Path(config.model.smpl_model_path))
    prepare_poses(plan['record']['source'], sorted({p['frame_id'] for p in plan['context_pairs']}),
                  poses, Path(config.model.smpl_model_path))
    # HumanGS assumes a distributed process group even for a single-GPU forward.
    with tempfile.TemporaryDirectory(prefix='humangs-dist-') as rendezvous:
        dist.init_process_group('nccl', init_method=f'file://{rendezvous}/init', rank=0, world_size=1)
        try:
            install_black_renderer()
            from model.HumanGS import HumanGS
            model = HumanGS(config).to(device).eval()
            # Upstream load_ckpt silently swallows errors; enforce strict loading here.
            checkpoint = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
            if checkpoint.get('vci_input_signature') != input_signature(plan['record']):
                raise ValueError('Checkpoint does not match the recorded VCI training inputs')
            # Frozen training loss networks are omitted in evaluation. Require
            # an exact match for every reconstruction parameter and buffer.
            weights = {key: value for key, value in checkpoint['model'].items()
                       if not key.startswith('loss_computer.')}
            model.load_state_dict(weights, strict=True)
            del checkpoint, weights
            lpips = create_lpips_model(device)
            fid = FrechetInceptionDistance(feature=2048).to(device)
            rows, empty = [], []
            original_forward = model.process_data.forward
            target_canvas = None
            def process_data(*a, **kw):
                context, target = original_forward(*a, **kw)
                target.image = target_canvas
                target.image_h_w = target_canvas.shape[-2:]
                return context, target
            model.process_data.forward = process_data
            def save(camera, frame, kind, value):
                destination = output / 'renders' / f'cam{camera:02d}' / kind
                destination.mkdir(parents=True, exist_ok=True)
                image = (value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(image).save(destination / f'{frame:08d}.png')
            with torch.no_grad():
                for index, pair in enumerate(pairs):
                    target, _ = load_image(pair, image_size=None,
                        resolution=plan['protocol']['resolution'], square=False)
                    if not (target * 255 >= .05).any():
                        empty.append([pair['camera_id'], pair['frame_id']])
                        continue
                    batch = sample([*plan['context_pairs'], pair], poses,
                        image_size=plan['record']['image_size'], resolution=plan['protocol']['resolution'], target_native=True)
                    batch = {k: v[None].to(device) if isinstance(v, torch.Tensor) else [v] for k, v in batch.items()}
                    # Targets are separate from encoder input; only cameras/poses reach reconstruction.
                    target_canvas = target[None, None].to(device)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.training.use_amp):
                        result = model(batch)
                    prediction = result.render[0, 0].float()
                    values = eval_images(prediction.permute(1, 2, 0), target.to(device).permute(1, 2, 0), fid, lpips)
                    rows.append({'camera': pair['camera_id'], 'frame': pair['frame_id'], **values})
                    if not args.no_save_images:
                        save(pair['camera_id'], pair['frame_id'], 'target', target)
                        save(pair['camera_id'], pair['frame_id'], 'black', prediction)
                    print(f'[{index + 1}/{len(pairs)}] cam{pair["camera_id"]}/{pair["frame_id"]}: {values}', flush=True)
            plan['empty_pairs'] = empty
            reference.write_text(json.dumps(plan, indent=2) + '\n')
            (output / 'per_frame_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
            if len(rows) < 2:
                raise ValueError('FID requires at least two nonempty evaluation pairs')
            metrics = {key: float(np.mean([r[key] for r in rows])) for key in ('psnr', 'ssim', 'lpips')}
            metrics['fid'] = float(fid.compute())
            (output / 'metrics.json').write_text(json.dumps({'split': plan['split'], 'pairs': len(rows), **metrics}, indent=2) + '\n')
        finally:
            dist.destroy_process_group()
    print(f'Results: {output}')
    return 0


if __name__ == '__main__':
    ensure_runtime_libraries()
    raise SystemExit(main())
