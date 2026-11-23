#!/usr/bin/env python3
"""Render a final PiG-Avatar checkpoint over its complete training motion timeline.

Example (pig-avatar environment):
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/render_neupig_showcase.py --run RUN
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MethodType

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--fps', type=float, default=12)
    p.add_argument('--playback-fps', type=float, default=60,
                   help='Training frames per second of playback, independent of output fps')
    p.add_argument('--source-fps', type=float, default=25,
                   help='Original capture rate for speed label (this DZNE capture: 25 fps)')
    p.add_argument('--preview', action='store_true', help='Render six timeline samples only')
    p.add_argument('--disable-pose-refinement', action='store_true',
                   help='Use interpolated input SMPL-X parameters without learned residual updates')
    p.add_argument('--camera-manifest', type=Path,
                   help='Reuse exact cameras from a previous render with matching timing')
    return p.parse_args()


def quaternion_slerp(q0, q1, weight):
    """Shortest-path unit-quaternion interpolation; q and -q are equivalent."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0, 1)
    angle = torch.acos(dot)
    sine = torch.sin(angle).clamp_min(1e-7)
    spherical = (torch.sin((1-weight)*angle)*q0 + torch.sin(weight*angle)*q1) / sine
    linear = torch.lerp(q0, q1, weight)
    return F.normalize(torch.where(dot > .9995, linear, spherical), dim=-1)


class PoseSampler:
    """Interpolate base SMPL-X poses while retaining the model's residual updates.

    Hook only this loaded instance's base-parameter lookup. This also keeps the
    mesh, splat binding, and returned conditioning parameters on the same pose,
    unlike the upstream override path, which handles some residuals differently.
    """
    rotation_keys = {'body_pose', 'global_orient', 'jaw_pose', 'left_hand_pose',
                     'right_hand_pose', 'leye_pose', 'reye_pose'}

    def __init__(self, human):
        from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
        self.to_axis_angle = quaternion_to_axis_angle
        self.position = 0.
        self.original = human._continous_init
        self.keys = {id(value): key for key, value in human.args.tmp_args['init_smplx_params'].items()}
        self.quaternions = {
            key: axis_angle_to_quaternion(value.reshape(value.shape[0], -1, 3))
            for key, value in human.args.tmp_args['init_smplx_params'].items()
            if key in self.rotation_keys
        }
        sampler = self

        def lookup(instance, indices, params):
            if instance.model_args.get('continous', False):
                return sampler.original(indices, params)
            position = min(max(sampler.position, 0.), len(params)-1)
            lo = int(math.floor(position))
            hi = min(lo+1, len(params)-1)
            weight = position-lo
            if weight == 0:
                return params[lo:lo+1]
            key = sampler.keys[id(params)]
            if key in sampler.rotation_keys:
                q = sampler.quaternions[key]
                return sampler.to_axis_angle(quaternion_slerp(q[lo:lo+1], q[hi:hi+1], weight)).reshape(1, -1)
            return torch.lerp(params[lo:lo+1], params[hi:hi+1], weight)

        human._continous_init = MethodType(lookup, human)


def camera(center, phase, radius, device):
    # Y points down in this capture's world coordinates. CV camera: x right, y down, z forward.
    angle = math.pi + 2 * math.pi * phase + 0.28 * math.sin(2 * math.pi * phase)
    elevation = 0.10 + 0.09 * math.sin(2 * math.pi * phase - 0.4)
    distance = radius * (1 + 0.045 * math.sin(4 * math.pi * phase + 0.6))
    eye = center + np.array([math.sin(angle), -elevation, math.cos(angle)]) * distance
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, -1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.stack([right, down, forward])
    view[:3, 3] = -view[:3, :3] @ eye
    return torch.tensor(view, device=device), torch.tensor(eye, device=device, dtype=torch.float32)


class Video:
    def __init__(self, path, size, fps):
        self.proc = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '4', str(path)
        ], stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError('ffmpeg encoding failed')


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def layout(mesh, splats, index, count, fps, source_position, source_count, pose_refinement=True,
           playback_fps=15, source_fps=25):
    image = Image.new('RGB', (1920, 1080), '#f4f5f7')
    image.paste(Image.fromarray(mesh), (36, 113))
    image.paste(Image.fromarray(splats), (984, 113))
    d = ImageDraw.Draw(image)
    d.text((38, 20), 'PiG-Avatar  /  Motion reconstruction', font=font(30), fill='#172332')
    if not pose_refinement:
        d.text((1340, 28), 'Pose refinement disabled', font=font(22), fill='#44617f')
    d.text((38, 70), 'SMPL-X geometry', font=font(23), fill='#44617f')
    d.text((986, 70), 'Gaussian splat reconstruction', font=font(23), fill='#44617f')
    d.text((38, 1025), 'DZNE pilot 007  |  50,000 splats  |  Final checkpoint', font=font(19), fill='#546271')
    d.text((1050, 1025), f'Playback ×{playback_fps / source_fps:.1f}', font=font(19), fill='#546271')
    label = f'Sequence {source_position + 1:05.1f} / {source_count}  |  {index / fps:05.2f} s  |  {fps:g} fps'
    d.text((1310, 1025), label, font=font(17), fill='#546271')
    d.rectangle((38, 1061, 1882, 1064), fill='#dce1e6')
    d.rectangle((38, 1061, 38 + round(1844 * (index + 1) / count), 1064), fill='#527eab')
    return np.asarray(image)


@torch.no_grad()
def main():
    a = arguments()
    if a.fps <= 0 or a.playback_fps <= 0 or a.source_fps <= 0:
        raise ValueError('fps, playback-fps, and source-fps must be positive')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'external/neu-pig-smplx'))
    from src.model_classes.avatar import AvatarModel
    from src.rendering.meshes import Meshes
    import nvdiffrast.torch as dr

    checkpoints = sorted(a.run.resolve().rglob('avatar_model.pth'))
    if len(checkpoints) != 1:
        raise ValueError(f'Expected one final checkpoint, found {len(checkpoints)}')
    checkpoint = checkpoints[0]
    out = a.out or a.run.resolve() / 'showcase'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    device = 'cuda:0'
    model = AvatarModel.from_checkpoint(checkpoint, map_location=device).eval().to(device)
    count = int(model.args.T)
    output_count = round(count * a.fps / a.playback_fps)
    if output_count < 1:
        raise ValueError('Playback settings produce no output frames')
    camera_manifest = None
    if a.camera_manifest:
        camera_manifest = json.loads(a.camera_manifest.read_text())
        if (camera_manifest['fps'] != a.fps
                or camera_manifest.get('playback_fps', 15) != a.playback_fps
                or camera_manifest['frame_count'] != count
                or len(camera_manifest['cameras']) != output_count
                or [record['frame'] for record in camera_manifest['cameras']] != list(range(output_count))):
            raise ValueError('Camera manifest must contain the full matching timeline and playback rates')
    sampler = PoseSampler(model.human_model)
    # Cache just the mesh geometry to frame and track the entire performance.
    ids = model.lods.lod_indices(1.)
    state = model.splat_model.anchor_state(ids)
    latent = model._grid_query(state, torch.tensor([1.], device=device, dtype=torch.float64))[..., :-1]
    meshes = []
    for i in range(count):
        sampler.position = float(i)
        # Tiny positive bias prevents float32 rounding from flooring to the previous frame.
        human, _ = model.human_model(None, latent, torch.tensor([(i + 1e-4) / count], device=device),
                                     eval_loss=False, apply_residual_update=not a.disable_pose_refinement)
        meshes.append(human['human_model'].verts_list()[0].cpu().numpy())
    meshes = np.stack(meshes)
    centers = (meshes.min(1) + meshes.max(1)) * .5
    # A symmetric Gaussian filter avoids jerky tracking and preserves endpoints.
    x = np.arange(-12, 13)
    kernel = np.exp(-.5 * (x / 5.) ** 2)
    kernel /= kernel.sum()
    centers = np.stack([np.convolve(np.pad(centers[:, j], (12, 12), mode='edge'), kernel, mode='valid') for j in range(3)], 1)
    size = 900
    focal = 1150.
    K = torch.tensor([[focal, 0, size/2], [0, focal, size/2], [0, 0, 1]], device=device)
    radius = 2.8
    # Fit every mesh at every actual orbit position, retaining a 12% border.
    for _ in range(8):
        largest = 0.
        for i in range(count):
            v, _ = camera(centers[i], i / max(1, count-1), radius, device)
            cv = meshes[i] @ v[:3, :3].cpu().numpy().T + v[:3, 3].cpu().numpy()
            largest = max(largest, float(np.max(np.abs(cv[:, :2] / cv[:, 2:]) * focal)))
        if largest < size * .39:
            break
        radius *= 1.08
    if camera_manifest:
        K = torch.tensor(camera_manifest['intrinsics'], device=device, dtype=torch.float32)
        focal = float(K[0, 0])
        radius = camera_manifest['radius']
    print(f'Source frames={count}; output frames={output_count}; fps={a.fps}; duration={output_count/a.fps:.2f}s; orbit radius={radius:.3f}', flush=True)
    context = dr.RasterizeCudaContext(device=device)
    faces = model.human_model.faces.int().contiguous()
    projection = torch.zeros((4,4), device=device)
    projection[0,0] = 2*focal/size
    projection[1,1] = -2*focal/size
    projection[2,2] = 100.01/99.99
    projection[2,3] = -2*100*.01/99.99
    projection[3,2] = 1
    selected = np.linspace(0, output_count-1, 6).astype(int).tolist() if a.preview else list(range(output_count))
    videos = {} if a.preview else {
        'comparison': Video(out/'showcase.mp4', (1920,1080), a.fps),
        'smplx': Video(out/'smplx_orbit.mp4', (size,size), a.fps),
        'splats': Video(out/'splats_orbit.mp4', (size,size), a.fps),
    }
    bg = torch.tensor([244/255,245/255,247/255], device=device)
    camera_records = []
    try:
        for i in selected:
            position = min(i * a.playback_fps / a.fps, count-1)
            sampler.position = position
            center = np.array([np.interp(position, np.arange(count), centers[:,j]) for j in range(3)])
            view, eye = camera(center, i/max(1,output_count-1), radius, device)
            if camera_manifest:
                record = camera_manifest['cameras'][i]
                view = torch.tensor(record['view_cv'], device=device, dtype=torch.float32)
                eye = torch.tensor(record['eye'], device=device, dtype=torch.float32)
            target_time = (position + 1e-4) / count
            outputs,_ = model({'target_time': torch.tensor([target_time],device=device)}, lod_fraction=1., lod_indices=ids,
                              log_latents=False, apply_smplx_residual_update=not a.disable_pose_refinement)
            data = {'V_CV':view[None,None], 'K_CV':K[None,None], 'camera_center':eye[None,None]}
            rgb,_,_ = outputs['splat_model'].render_rgb(data,(size,size),background=bg)
            splats = rgb.squeeze().mul(255).round().byte().cpu().numpy()
            verts = outputs['human_model'].verts_list()[0]
            normals = Meshes([verts], [faces.long()]).compute_vertex_normals()
            vh = torch.cat([verts,torch.ones_like(verts[:,:1])],-1)
            clip = (vh @ (projection @ view).T)[None].contiguous()
            rast,_ = dr.rasterize(context,clip,faces,resolution=(size,size))
            normal,_ = dr.interpolate(normals[None],rast,faces)
            normal = F.normalize(normal,dim=-1)
            light = F.normalize(-view[2,:3] + .6*(-view[1,:3]) - .35*view[0,:3],dim=0)
            diffuse = (normal*light).sum(-1,keepdim=True).clamp(0,1)
            color = torch.tensor([.40,.57,.73],device=device) * (.42 + .58*diffuse)
            mask = (rast[...,3:] > 0).float()
            rgb_mesh = dr.antialias(color*mask + bg*(1-mask),rast,clip,faces)
            mesh = rgb_mesh[0].flip(0).clamp(0,1).mul(255).round().byte().cpu().numpy()
            combined = layout(mesh,splats,i,output_count,a.fps,position,count,not a.disable_pose_refinement,
                              a.playback_fps,a.source_fps)
            if i in [selected[0], selected[len(selected)//2], selected[-1]] or a.preview:
                Image.fromarray(combined).save(out/f'preview_{i:04d}.jpg',quality=95)
            if videos:
                videos['comparison'].write(combined)
                videos['smplx'].write(mesh)
                videos['splats'].write(splats)
            camera_records.append({'frame':i,'source_position':position,'target_time':target_time,'view_cv':view.cpu().tolist(),'eye':eye.cpu().tolist()})
            if i % 20 == 0 or i == selected[-1]:
                print(f'Rendered {i+1}/{output_count}',flush=True)
    finally:
        for writer in videos.values():
            writer.close()
    metadata = {'checkpoint':str(checkpoint),'frame_count':count,'rendered_frames':len(selected),
                'fps':a.fps,'duration_seconds':len(selected)/a.fps,'sequence':'Full training motion, final model',
                'playback_fps':a.playback_fps,'output_frame_count':output_count,
                'source_fps':a.source_fps,'playback_speed_multiplier':a.playback_fps/a.source_fps,
                'playback_speed_label':f'×{a.playback_fps/a.source_fps:.1f}',
                'pose_refinement_enabled':not a.disable_pose_refinement,
                'camera_manifest_source':str(a.camera_manifest.resolve()) if a.camera_manifest else None,
                'playback':'Fixed training frames per second; final pose held for its remaining frame interval',
                'pose_interpolation':'Shortest-path quaternion SLERP for base joint rotations; linear translation/expression; '
                                     + ('learned pose residuals disabled; ' if a.disable_pose_refinement else 'continuous learned pose residuals; ')
                                     + 'learned splat networks',
                'camera':'One full orbit with sinusoidal speed, elevation and distance variation; smoothed subject tracking',
                'intrinsics':K.cpu().tolist(),'radius':radius,'cameras':camera_records}
    (out/('preview_manifest.json' if a.preview else 'render_manifest.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Output: {out}',flush=True)


if __name__ == '__main__':
    main()
