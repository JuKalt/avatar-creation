"""HumanGS must keep context strictly inside training and preserve source IDs."""
import json
from pathlib import Path
import tempfile
import unittest
import yaml

from avatar_creation.methods.humangs import prepare_subject, evaluation_plan, observations
from avatar_creation.methods.registry import load_method_spec


class HumanGSTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        calibration = {}
        for camera in [1, 4, 7]:
            name = f'cam{camera:02d}'
            calibration[name] = {'K': [10, 0, 4, 0, 10, 4, 0, 0, 1],
                                  'R': [1, 0, 0, 0, 1, 0, 0, 0, 1], 'T': [0, 0, 3]}
            for kind in ['images', 'masks']:
                folder = self.source / kind / name
                folder.mkdir(parents=True)
                for frame in [2, 4, 6, 8]:
                    (folder / f'{frame:08d}.jpg').touch()
        (self.source / 'calibration.json').write_text(json.dumps(calibration))
        (self.source / 'smpl_params.npz').touch()
        self.subject = {'subject_id': 'test', 'dataset': 'thuman', 'subject_path': str(self.source),
                        'images': str(self.source / 'images'), 'masks': str(self.source / 'masks'),
                        'calibration': str(self.source / 'calibration.json'), 'smplx': str(self.source / 'smpl_params.npz'),
                        'training_config': {'camera_ids': [4, 1], 'frame_range': {'start': 2, 'stop': 7, 'step': 2}, 'resolution': 0},
                        'evaluation_config': {'novel_view': {'camera_ids': [7], 'frame_range': {'start': 2, 'stop': 7, 'step': 2}},
                                              'novel_pose': {'camera_ids': [1, 4], 'frame_range': {'start': 8, 'stop': 9}}}}
        # Minimal upstream config keeps tests independent from the ignored checkout.
        self.template = {'model': {'image_tokenizer': {}}, 'training': {'warmup': 3000}, 'inference': {}}
        self.config = prepare_subject(self.subject, self.root / 'run', self.template,
                                      num_input_views=2, num_target_views=2)
        self.owner = self.config.parent

    def test_training_never_includes_heldout_pairs(self):
        record = json.loads((self.owner / 'humangs_inputs.json').read_text())
        self.assertEqual({p['camera_id'] for p in record['training_pairs']}, {1, 4})
        self.assertEqual({p['frame_id'] for p in record['training_pairs']}, {2, 4, 6})
        config = yaml.safe_load(self.config.read_text())
        self.assertFalse(config['training']['target_has_input'])
        self.assertFalse(config['training']['use_wandb'])

    def test_both_splits_use_identical_training_only_context(self):
        view = evaluation_plan(self.owner, 'novel-view')
        pose = evaluation_plan(self.owner, 'novel-pose')
        self.assertEqual(view['context_pairs'], pose['context_pairs'])
        self.assertEqual([(p['camera_id'], p['frame_id']) for p in view['context_pairs']], [(1, 2), (4, 2)])
        pairs, _ = observations(self.subject, pose['target_selection'])
        self.assertEqual([(p['camera_id'], p['frame_id']) for p in pairs], [(1, 8), (4, 8)])

    def test_rejects_pose_and_view_leakage(self):
        file = self.owner / 'evaluation_split.json'
        metadata = json.loads(file.read_text())
        metadata['evaluation']['novel_pose']['frame_range']['start'] = 6
        file.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            evaluation_plan(self.owner, 'novel-pose')
        metadata['evaluation']['novel_view']['camera_ids'] = [4]
        file.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'leaks'):
            evaluation_plan(self.owner, 'novel-view')

    def test_missing_pairs_do_not_compact_timeline(self):
        (self.source / 'masks/cam07/00000004.jpg').unlink()
        pairs, missing = observations(self.subject, self.subject['evaluation_config']['novel_view'])
        self.assertEqual([p['frame_id'] for p in pairs], [2, 6])
        self.assertEqual(missing, [[7, 4]])

    def test_changed_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            prepare_subject(self.subject, self.root / 'run', self.template,
                            num_input_views=1, num_target_views=2)

    def test_staged_camera_mapping_preserves_physical_ids(self):
        subject = {**self.subject, 'staged_camera_mapping': {'cam01': 4, 'cam04': 1, 'cam07': 7}}
        pairs, _ = observations(subject, {'camera_ids': [4], 'frame_range': {'start': 2, 'stop': 3}})
        self.assertEqual(pairs[0]['camera_id'], 4)
        self.assertIn('cam01', pairs[0]['image'])

    def test_registry_supports_training_and_evaluation(self):
        spec = load_method_spec('humangs')
        self.assertEqual(spec.conda_env, 'humangs')
        self.assertIn('--split novel-pose', spec.evaluation_scripts['novel_pose'])


if __name__ == '__main__':
    unittest.main()
