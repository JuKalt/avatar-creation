"""Standalone evaluation must select real held-out cameras without a PiG run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('mmlp_eval', Path(__file__).resolve().parents[1] / 'scripts/evaluation/eval_mmlphuman.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class MMLPEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'subject'
        (self.owner / 'model').mkdir(parents=True)
        (self.owner / 'data').mkdir()
        self.checkpoint = self.owner / 'model/chkpnt10.pth'
        self.checkpoint.touch()
        source = self.root / 'source'
        for kind in ('images', 'masks'):
            for cam in ('cam04', 'cam19'):
                directory = source / kind / cam
                directory.mkdir(parents=True)
                for frame in (2, 4):
                    (directory / f'{frame:08d}.png').touch()
        (source / 'calibration.json').write_text(json.dumps({'cam04': {}, 'cam19': {}}))
        (self.owner / 'data/calibration.json').write_text(json.dumps({'cam04': {}}))
        (self.owner / 'evaluation_split.json').write_text(json.dumps({
            'resolution': 1024, 'train_cameras': [0],
            'original_data': {key: str(source / value) for key, value in {'images': 'images', 'masks': 'masks', 'calibration': 'calibration.json'}.items()},
        }))
        (self.owner / 'mmlphuman.yaml').write_text(json.dumps({'train_cam_ids': [0], 'begin_ith_frame': 2, 'num_train_frame': 2, 'frame_interval': 2}))

    def test_heldout_uses_source_names_and_training_timeline(self):
        _, _, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'heldout')
        self.assertEqual(audit['evaluation_cameras'], ['cam19'])
        self.assertEqual([(p[0], p[1]) for p in pairs], [('cam19', 2), ('cam19', 4)])
        self.assertEqual(audit['train_cameras'], ['cam04'])
        self.assertEqual(audit['resolution'], 0)
        self.assertEqual(evaluation.evaluation_plan(self.checkpoint, 'heldout', resolution=512)[3]['resolution'], 512)

    def test_rejects_training_camera_as_heldout(self):
        with self.assertRaisesRegex(ValueError, 'outside the heldout split'):
            evaluation.evaluation_plan(self.checkpoint, 'heldout', ['cam04'])

    def test_no_heldout_requires_explicit_train_split(self):
        (self.root / 'source/calibration.json').write_text(json.dumps({'cam04': {}}))
        with self.assertRaisesRegex(ValueError, 'No cameras available'):
            evaluation.evaluation_plan(self.checkpoint, 'heldout')
        _, _, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'train')
        self.assertEqual(len(pairs), 2)
        self.assertEqual(audit['split'], 'train')

    def test_missing_mask_is_recorded_and_omitted(self):
        (self.root / 'source/masks/cam19/00000004.png').unlink()
        _, _, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'heldout')
        self.assertEqual(len(pairs), 1)
        self.assertEqual(audit['missing_pairs'], [('cam19', 4)])

    def test_validate_only_needs_no_pig_or_checkpoint_payload(self):
        import contextlib
        import io
        import numpy as np
        np.savez(self.owner / 'data/smpl_params.npz',
                 global_orient=np.zeros((5, 3)), body_pose=np.zeros((5, 63)),
                 left_hand_pose=np.zeros((5, 45)), right_hand_pose=np.zeros((5, 45)),
                 transl=np.zeros((5, 3)))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = evaluation.main(['--mmlp-path', str(self.owner), '--validate-only'])
        self.assertEqual(result, 0)
        self.assertIn('2 pairs, 1 heldout cameras', output.getvalue())
        self.assertFalse((self.owner / 'evaluation').exists())

    def test_lod_checkpoint_selects_gaussians_and_conditions_encoder(self):
        import torch

        class BaseModel:
            @property
            def get_joint_features(self):
                return torch.arange(63, dtype=torch.float32)

        plain = {'encoder_feat_params': {'layers.0.weight': torch.empty(1, 1, 63)}}
        self.assertIs(evaluation.model_class_for_checkpoint(BaseModel, plain), BaseModel)
        with self.assertRaisesRegex(ValueError, 'requires an MMLPHuman LOD checkpoint'):
            evaluation.model_class_for_checkpoint(BaseModel, plain, 0.5)

        lod = {'encoder_feat_params': {'layers.0.weight': torch.empty(1, 1, 64)},
               '_xyz': torch.empty(10, 3),
               'lod': {'model_type': 'mmlphuman-lod', 'version': 1,
                       'config': {'min_fraction': 0.1, 'max_fraction': 1.0},
                       'permutation': torch.tensor([9, 2, 7, 0, 8, 1, 4, 3, 6, 5])}}
        self.assertEqual(evaluation.lod_fraction_for_checkpoint(lod, None), 1.0)
        model = evaluation.model_class_for_checkpoint(BaseModel, lod, 0.3)()
        self.assertEqual(model.get_joint_features.shape, (64,))
        self.assertAlmostEqual(model.get_joint_features[-1].item(), 0.3)
        indices = evaluation.lod_indices_for_checkpoint(lod, 0.3, 'cpu')
        self.assertEqual(indices.tolist(), [9, 2, 7])
        with self.assertRaisesRegex(ValueError, 'LOD fraction must be between'):
            evaluation.lod_fraction_for_checkpoint(lod, 0.05)
        lod['lod']['version'] = 2
        with self.assertRaisesRegex(ValueError, 'Unsupported MMLPHuman checkpoint'):
            evaluation.model_class_for_checkpoint(BaseModel, lod)

    def test_latest_checkpoint_is_numeric(self):
        (self.owner / 'model/chkpnt9.pth').touch()
        self.assertEqual(evaluation.checkpoint_for(self.owner), self.checkpoint)
        self.assertEqual(evaluation.checkpoint_for(self.owner / 'model'), self.checkpoint)

class DNAEvaluationTests(MMLPEvaluationTests):
    def setUp(self):
        super().setUp()
        import h5py
        import numpy as np
        import cv2
        self.rgb_path = self.root / 'rgb.smc'
        self.annot_path = self.root / 'annot.smc'
        with h5py.File(self.rgb_path, 'w') as rgb, h5py.File(self.annot_path, 'w') as annot:
            for camera in (2, 7, 48):
                group = 'Camera_12mp' if camera >= 48 else 'Camera_5mp'
                image = np.full((16, 16, 3), (10, 20, 30), dtype=np.uint8)
                mask = np.full((16, 16), 255, dtype=np.uint8)
                for frame in (2, 4):
                    rgb.create_dataset(f'{group}/{camera}/color/{frame}', data=cv2.imencode('.png', image)[1])
                    annot.create_dataset(f'Mask/{camera}/mask/{frame}', data=cv2.imencode('.png', mask)[1])
                cam = annot.create_group(f'Camera_Parameter/{camera:02d}')
                c2w = np.eye(4); c2w[0, 3] = 2
                cam['RT'] = c2w
                cam['K'] = np.eye(3)
                cam['D'] = np.zeros(5)
        meta = json.loads((self.owner / 'evaluation_split.json').read_text())
        meta.update(dataset_kind='dna', original_data={'images': str(self.rgb_path), 'masks': str(self.annot_path), 'calibration': str(self.annot_path)})
        (self.owner / 'evaluation_split.json').write_text(json.dumps(meta))
        (self.owner / 'data/calibration.json').write_text(json.dumps({'cam00': {}, 'cam01': {}}))
        config = json.loads((self.owner / 'mmlphuman.yaml').read_text())
        config['train_cam_ids'] = [0, 1]
        (self.owner / 'mmlphuman.yaml').write_text(json.dumps(config))

    def test_heldout_uses_source_names_and_training_timeline(self):
        _, calibration, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'heldout')
        self.assertEqual(audit['evaluation_cameras'], ['cam48'])
        self.assertEqual(audit['train_cameras'], ['cam02', 'cam07'])
        self.assertEqual(audit['camera_groups']['cam48'], 'Camera_12mp')
        self.assertEqual(calibration['cam48']['T'], [-2., 0., 0.])
        self.assertEqual([p[1] for p in pairs], [2, 4])

    def test_rejects_training_camera_as_heldout(self):
        with self.assertRaisesRegex(ValueError, 'outside the heldout split'):
            evaluation.evaluation_plan(self.checkpoint, 'heldout', ['cam07'])

    def test_no_heldout_requires_explicit_train_split(self):
        import h5py
        with h5py.File(self.rgb_path, 'a') as rgb:
            del rgb['Camera_12mp']
        with self.assertRaisesRegex(ValueError, 'No cameras available'):
            evaluation.evaluation_plan(self.checkpoint, 'heldout')
        _, _, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'train')
        self.assertEqual(len(pairs), 4)
        self.assertEqual(audit['evaluation_cameras'], ['cam02', 'cam07'])

    def test_missing_mask_is_recorded_and_omitted(self):
        import h5py
        with h5py.File(self.annot_path, 'a') as annot:
            del annot['Mask/48/mask/4']
        _, _, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'heldout')
        self.assertEqual(len(pairs), 1)
        self.assertEqual(audit['missing_pairs'], [('cam48', 4)])

    def test_decodes_native_pair_without_rgb_swap(self):
        import h5py
        _, _, pairs, _ = evaluation.evaluation_plan(self.checkpoint, 'heldout', frames=[4])
        with h5py.File(self.rgb_path, 'r') as rgb, h5py.File(self.annot_path, 'r') as annot:
            image, mask = evaluation.read_pair(pairs[0][2], pairs[0][3], (rgb, annot))
        self.assertEqual(image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(int(mask[0, 0]), 255)
        with self.assertRaisesRegex(ValueError, 'training frame range'):
            evaluation.evaluation_plan(self.checkpoint, 'heldout', frames=[3])


class NeuPiGProtocolTests(unittest.TestCase):
    @staticmethod
    def upstream_functions(relative_path, names, namespace, class_name=None):
        # Execute only the reference functions, avoiding Neu-PiG's model/CUDA imports.
        import ast
        path = Path(__file__).resolve().parents[1] / 'external/neu-pig-smplx' / relative_path
        if not path.is_file():
            raise unittest.SkipTest('Neu-PiG checkout is required for reference parity')
        tree = ast.parse(path.read_text())
        nodes = tree.body
        if class_name:
            nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
        selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in selected} != set(names):
            raise AssertionError(f'Neu-PiG reference functions changed: {names}')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        return {name: namespace[name] for name in names}

    def test_targets_and_intrinsics_match_upstream_with_and_without_resize(self):
        import torch
        namespace = {'torch': torch}
        methods = self.upstream_functions('src/io/datasets/base.py',
                    ['_resolution', '_get_resized_image', '_get_resized_intrinsics'], namespace, 'BaseDataset')
        Reference = type('Reference', (), methods)
        prepare = self.upstream_functions('src/opt_pipeline/evaluation.py',
                                          ['_prepare_eval_data'], namespace)['_prepare_eval_data']
        rgb = torch.arange(17 * 23 * 3).reshape(17, 23, 3).remainder(256).to(torch.uint8)
        mask = torch.tensor([0, 127, 128, 129, 255], dtype=torch.uint8).repeat(79)[:17 * 23].reshape(17, 23, 1)
        camera = {'K': [100., 0., 11., 0., 110., 8., 0., 0., 1.],
                  'distCoeff': [-0.1, 0.08, 0.001, 0.002, 0.]}
        for resolution in (0, 13):
            with self.subTest(resolution=resolution):
                ref = Reference()
                ref.resolution = resolution or float('inf')
                expected_rgb = ref._get_resized_image(rgb.permute(2, 0, 1)).permute(1, 2, 0)
                expected_mask = ref._get_resized_image(mask.permute(2, 0, 1)).permute(1, 2, 0)
                expected = prepare({'rgb': expected_rgb[None, None], 'mask': expected_mask[None, None]})['rgb'][0, 0]
                target, K = evaluation.prepare_target(rgb, mask, camera, resolution)
                self.assertTrue(torch.equal(target, expected))
                self.assertTrue(torch.equal(K, ref._get_resized_intrinsics(torch.tensor(camera['K']).reshape(3, 3))))
        target, _ = evaluation.prepare_target(rgb, mask, camera)
        self.assertTrue(torch.equal(target[mask[..., 0] == 128], rgb[mask[..., 0] == 128]))
        empty, _ = evaluation.prepare_target(rgb, torch.zeros_like(mask), camera)
        self.assertFalse((empty >= 0.05).any())

    def test_dna_mask_uses_max_channel(self):
        import cv2
        import numpy as np
        pixels = np.array([[[0, 0, 255], [255, 0, 0], [0, 128, 0]]], dtype=np.uint8)
        encoded = cv2.imencode('.png', pixels)[1]
        image, mask = evaluation.read_pair('image', 'mask', ({'image': encoded}, {'mask': encoded}))
        self.assertEqual(image[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(mask[..., 0].tolist(), [[255, 255, 128]])

    def test_metrics_and_fid_inputs_match_upstream(self):
        import torch
        from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
        namespace = {'torch': torch, 'FrechetInceptionDistance': object,
                     'peak_signal_noise_ratio': peak_signal_noise_ratio,
                     'structural_similarity_index_measure': structural_similarity_index_measure}
        reference = self.upstream_functions('src/metrics/evaluation.py', ['eval_images'], namespace)['eval_images']
        class FIDCapture:
            def __init__(self):
                self.inputs = []
            def update(self, value, real):
                self.inputs.append((real, value.clone()))
        class LPIPSCapture:
            def __init__(self):
                self.inputs = []
            def __call__(self, pred, target):
                self.inputs.append((pred.clone(), target.clone()))
                return (pred - target).square().mean().reshape(1)
        generator = torch.Generator().manual_seed(123)
        pred = torch.rand(32, 40, 3, generator=generator) * 1.4 - 0.2
        target = torch.rand(32, 40, 3, generator=generator)
        actual_fid, expected_fid = FIDCapture(), FIDCapture()
        actual_lpips, expected_lpips = LPIPSCapture(), LPIPSCapture()
        actual = evaluation.eval_images(pred, target, actual_fid, actual_lpips)
        expected = reference(pred[None, None], target[None, None], expected_fid, expected_lpips)
        self.assertEqual(actual, expected)
        for (a_real, a), (b_real, b) in zip(actual_fid.inputs, expected_fid.inputs):
            self.assertEqual(a_real, b_real)
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(actual_lpips.inputs[0], expected_lpips.inputs[0]):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.stride(), b.stride())

    def test_full_and_lod_renders_use_neupig_settings(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        model = SimpleNamespace(get_xyz=torch.arange(9).reshape(3, 3).float(),
                                get_opacity=torch.ones(3),
                                get_color=lambda _: torch.ones(3, 3),
                                get_covariance=lambda _: torch.eye(3).repeat(3, 1, 1))
        cam = {'w2c': torch.eye(4), 'K': torch.eye(3), 'width': 12, 'height': 10}
        for indices in (None, torch.tensor([2, 0])):
            with patch('gsplat.rasterization', return_value=(torch.zeros(1, 10, 12, 3), torch.zeros(1, 10, 12, 1), {})) as render:
                evaluation.render_lod(model, cam, indices, torch.zeros(3))
                kwargs = render.call_args.kwargs
                self.assertEqual(kwargs['rasterize_mode'], 'antialiased')
                self.assertEqual(kwargs['near_plane'], 0.01)
                expected = model.get_xyz if indices is None else model.get_xyz[indices]
                self.assertTrue(torch.equal(kwargs['means'], expected))


if __name__ == '__main__':
    unittest.main()
