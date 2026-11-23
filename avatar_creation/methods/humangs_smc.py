"""Bounded, read-only access to native DNA-Rendering HDF5 SMC observations."""
from __future__ import annotations
import numpy as np


class HumanGSSMCReader:
    def __init__(self, path):
        import h5py
        self.smc = h5py.File(path, 'r', rdcc_nbytes=1024 * 1024)

    def get_Calibration_all(self):
        result = {}
        for name, group in self.smc['Camera_Parameter'].items():
            if name.isdigit():
                result[name] = {key: group[key][()] for key in ('K', 'RT')}
                result[name]['D'] = group['D'][()] if 'D' in group else np.zeros(5)
        return result

    def get_Camera_5mp_info(self):
        counts = [max(map(int, group['color'].keys()), default=-1) + 1
                  for name, group in self.smc['Camera_5mp'].items()
                  if name.isdigit() and 'color' in group]
        return {'num_frame': min(counts, default=0)}

    def get_SMPLx(self):
        group = self.smc['SMPLx']
        return {key: group[key][()] for key in ('betas', 'expression', 'fullpose', 'transl')}

    def _decode(self, key, mask=False):
        import cv2
        if key not in self.smc:
            raise ValueError(f'Missing native DNA observation: {key}')
        image = cv2.imdecode(np.asarray(self.smc[key], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode native DNA observation: {key}')
        return image.max(axis=-1) if mask else image

    def get_img(self, group, camera, image_type, *, Frame_id, disable_tqdm=True):
        return self._decode(f'{group}/{int(camera)}/{image_type}/{int(Frame_id)}')

    def get_mask(self, camera, *, Frame_id, disable_tqdm=True):
        return self._decode(f'Mask/{int(camera)}/mask/{int(Frame_id)}', mask=True)

    def release(self):
        self.smc.close()
