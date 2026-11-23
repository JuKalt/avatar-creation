#!/usr/bin/env python3
"""Check isolated HumanGS native imports and a CUDA convolution."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from avatar_creation.methods.humangs import ensure_runtime_libraries


def main():
    import torch
    import xformers.ops
    import pytorch3d._C
    import diff_gaussian_rasterization
    import smplx
    import yaml
    import h5py
    import torchmetrics
    import torch_fidelity
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    image = torch.zeros(1, 3, 16, 16, device='cuda')
    torch.nn.Conv2d(3, 4, 3).cuda()(image)
    torch.cuda.synchronize()
    print(f'HumanGS ready: torch={torch.__version__}, cuDNN={torch.backends.cudnn.version()}')


if __name__ == '__main__':
    ensure_runtime_libraries()
    main()
