"""Check HumanGS image/camera alignment and metric-target preprocessing."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

from avatar_creation.methods.humangs_dataset import load_image, prepare_poses, sample
from avatar_creation.methods.humangs import canonical_source, observations
from scripts.evaluation.eval_mmlphuman import prepare_target

MODEL = Path(__file__).resolve().parents[1] / 'external/HumanGS/body_models/smplx/SMPLX_NEUTRAL.npz'


class HumanGSImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = np.full((20, 10, 3), 128, dtype=np.uint8)
        self.mask = np.zeros((20, 10), dtype=np.uint8)
        self.mask[3:17, 2:8] = 255
        Image.fromarray(self.image).save(self.root / 'image.png')
        Image.fromarray(self.mask).save(self.root / 'mask.png')
        self.pair = {'image': str(self.root / 'image.png'), 'mask': str(self.root / 'mask.png'),
                     'calibration': {'K': [12, 0, 5, 0, 14, 10, 0, 0, 1]}}

    def test_padding_preserves_subject_and_shifts_intrinsics(self):
        image, intrinsics = load_image(self.pair, image_size=40)
        self.assertEqual(tuple(image.shape), (3, 40, 40))
        torch.testing.assert_close(intrinsics, torch.tensor([24., 28., 20., 20.]))
        self.assertEqual(float(image[:, :, :10].abs().max()), 0.)
        self.assertGreater(float(image.sum()), 0.)

    def test_native_target_matches_shared_metric_protocol(self):
        actual, intrinsics = load_image(self.pair, image_size=None, resolution=13, square=False)
        expected, K = prepare_target(torch.from_numpy(self.image), torch.from_numpy(self.mask)[..., None],
                                     self.pair['calibration'], 13)
        torch.testing.assert_close(actual, expected.permute(2, 0, 1).float() / 255)
        torch.testing.assert_close(intrinsics, torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
        self.assertEqual(tuple(actual.shape), (3, 13, 6))

    def test_sample_preserves_rotated_camera_world_position(self):
        self.pair.update(camera_id=7, frame_id=0)
        self.pair['calibration'].update(
            R=[0, -1, 0, 1, 0, 0, 0, 0, 1], T=[1, 2, 3],
        )
        np.savez(self.root / '00000000.npz',
                 v_posed=np.zeros((1, 3)), v_cano=np.zeros((1, 3)),
                 faces=np.zeros((1, 3), dtype=np.int64), joint_transforms=np.eye(4)[None])
        data = sample([self.pair], self.root, image_size=20)
        camera_to_world = data['c2w'][0]
        torch.testing.assert_close(camera_to_world[:3, 3], torch.tensor([-2., 1., -3.]))
        world_to_camera = torch.tensor([[0., -1., 0., 1.], [1., 0., 0., 2.],
                                        [0., 0., 1., 3.], [0., 0., 0., 1.]])
        torch.testing.assert_close(world_to_camera @ camera_to_world, torch.eye(4))

    def test_native_dna_12mp_is_lossless_and_preserves_ids(self):
        import h5py
        import cv2
        rgb = self.root / 'rgb.smc'
        annotation = self.root / 'annotation.smc'
        image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        mask = np.full((8, 10), 255, dtype=np.uint8)
        with h5py.File(rgb, 'w') as f:
            for camera, group in ((0, 'Camera_5mp'), (48, 'Camera_12mp')):
                for frame in range(4):
                    f.create_dataset(f'{group}/{camera}/color/{frame}', data=cv2.imencode('.png', image)[1])
        with h5py.File(annotation, 'w') as f:
            for camera in (0, 48):
                f.create_dataset(f'Camera_Parameter/{camera:02d}/K', data=np.eye(3))
                f.create_dataset(f'Camera_Parameter/{camera:02d}/RT', data=np.eye(4))
                f.create_dataset(f'Camera_Parameter/{camera:02d}/D', data=np.zeros(5))
                for frame in range(4):
                    f.create_dataset(f'Mask/{camera}/mask/{frame}', data=cv2.imencode('.png', mask)[1])
            f.create_dataset('SMPLx/fullpose', data=np.zeros((4, 55, 3)))
            f.create_dataset('SMPLx/betas', data=np.zeros((4, 10)))
            f.create_dataset('SMPLx/expression', data=np.zeros((4, 10)))
            f.create_dataset('SMPLx/transl', data=np.zeros((4, 3)))
        subject = {'dataset': 'dna', 'subject_id': 'test', 'images': str(rgb),
                   'masks': str(annotation), 'calibration': str(annotation), 'smplx': str(annotation)}
        selection = {'camera_ids': [48], 'frame_range': {'start': 1, 'stop': 4, 'step': 2}}
        source = canonical_source(subject, self.root / 'stage', selection)
        pairs, missing = observations(source, selection)
        self.assertEqual([(p['camera_id'], p['frame_id']) for p in pairs], [(48, 1), (48, 3)])
        self.assertEqual(missing, [])
        self.assertEqual(Path(pairs[0]['image']).suffix, '.png')
        np.testing.assert_array_equal(cv2.imread(pairs[0]['image']), image)
        np.testing.assert_array_equal(cv2.imread(pairs[0]['mask'], cv2.IMREAD_GRAYSCALE), mask)

    @unittest.skipUnless(MODEL.is_file(), 'Licensed SMPL-X model not installed')
    def test_rest_to_world_translation_and_sparse_frame_ids(self):
        try:
            import smplx  # noqa: F401  # availability probe
        except ImportError:
            self.skipTest('smplx not installed in this test environment')
        source = self.root / 'smpl.npz'
        np.savez(source, betas=np.zeros((1, 10), dtype=np.float32),
                 body_pose=np.zeros((3, 63), dtype=np.float32),
                 global_orient=np.zeros((3, 3), dtype=np.float32),
                 transl=np.tile(np.array([[1., 2., 3.]], dtype=np.float32), (3, 1)))
        prepare_poses({'smplx': str(source)}, [0, 2], self.root / 'poses', MODEL)
        self.assertFalse((self.root / 'poses/00000001.npz').exists())
        with np.load(self.root / 'poses/00000002.npz') as data:
            np.testing.assert_allclose(data['v_posed'], data['v_cano'] + [1., 2., 3.], atol=1e-6)
            expected = np.tile(np.eye(4), (55, 1, 1))
            expected[:, :3, 3] = [1., 2., 3.]
            np.testing.assert_allclose(data['joint_transforms'], expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
