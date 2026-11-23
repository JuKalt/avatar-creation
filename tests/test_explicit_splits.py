"""Explicit source IDs must survive validation, staging, and evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from avatar_creation.methods.experiment_config import load_experiment_config, training_arguments
from avatar_creation.methods.mmlphuman import prepare_subject
from avatar_creation.methods.training_inputs import validate_dna_inputs
from tests.test_external_methods import _subject
from tests.test_mmlphuman_evaluation import MMLPEvaluationTests, evaluation


class ExplicitSplitTests(unittest.TestCase):
    def config(self):
        return {'schema_version': 1, 'dataset_kind': 'thuman', 'subjects': ['subject01'],
                'train': {'camera_ids': [4, 0], 'frame_range': {'start': 2, 'stop': 8, 'step': 2}, 'resolution': 0},
                'evaluation': {
                    'novel_view': {'camera_ids': [19], 'frame_range': {'start': 2, 'stop': 8, 'step': 2}},
                    'novel_pose': {'camera_ids': [4], 'frame_range': {'start': 8, 'stop': 12}}}}

    def load(self, config):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(config))
            return load_experiment_config(path)

    def test_counts_derived_without_losing_ids(self):
        config = self.load(self.config())
        self.assertEqual(config, self.config())
        self.assertEqual(training_arguments(config, []), ['--views', '2', '--time-steps', '3', '--resolution', '0'])

    def test_invalid_or_leaking_splits_rejected(self):
        for field, value in [('camera_ids', []), ('camera_ids', [1, 1]), ('camera_ids', [True]),
                             ('frame_range', {'start': 2, 'stop': 2}),
                             ('frame_range', {'start': 0, 'stop': 4, 'step': 0})]:
            config = self.config()
            config['train'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SystemExit):
                self.load(config)
        for split, field, value in [('novel_view', 'camera_ids', [4]),
                                     ('novel_view', 'frame_range', {'start': 9, 'stop': 10}),
                                     ('novel_pose', 'frame_range', {'start': 6, 'stop': 10})]:
            config = self.config()
            config['evaluation'][split][field] = value
            with self.subTest(split=split), self.assertRaises(SystemExit):
                self.load(config)

    def test_staging_uses_source_frames_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            subject = _subject(source)
            (source / 'calibration.json').write_text(json.dumps({'cam00': {}, 'cam01': {}}))
            (source / 'gaussian').mkdir()
            (source / 'gaussian/lbs_weights_grid.npz').touch()
            for kind in ('images', 'masks'):
                for frame in (2, 4):
                    (source / kind / 'cam01' / f'{frame:08d}.jpg').touch()
            subject['training_config'] = {'camera_ids': [1], 'frame_range': {'start': 2, 'stop': 6, 'step': 2}, 'resolution': 0}
            subject['evaluation_config'] = {'novel_pose': {'camera_ids': [1], 'frame_range': {'start': 6, 'stop': 8}}}
            data, config_path = prepare_subject(subject, root / 'runs', {'test': {}}, views=1, time_steps=2)
            import yaml
            config = yaml.safe_load(config_path.read_text())
            self.assertEqual([p.name for p in (data / 'images').iterdir()], ['cam01'])
            self.assertEqual(sorted(p.stem for p in (data / 'images/cam01').iterdir()), ['00000002', '00000004'])
            self.assertEqual((config['begin_ith_frame'], config['num_train_frame'], config['frame_interval']), (2, 2, 2))
            metadata = json.loads((config_path.parent / 'evaluation_split.json').read_text())
            self.assertEqual(metadata['train_source_camera_ids'], [1])
            self.assertEqual(metadata['train_frame_ids'], [2, 4])
            self.assertEqual(metadata['evaluation'], subject['evaluation_config'])
            subject['training_config']['frame_range'] = {'start': 0, 'stop': 2}
            prepare_subject(subject, root / 'runs', {'test': {}}, views=1, time_steps=2)
            self.assertEqual(sorted(p.stem for p in (data / 'images/cam01').iterdir()), ['00000000', '00000001'])

    def test_dna_accepts_nonconsecutive_ids_checks_absolute_frame_bounds(self):
        rgb, annot = Mock(), Mock()
        rgb.get_Camera_5mp_info.return_value = {'num_frame': 12}
        annot.get_SMPLx.return_value = {'transl': [0] * 12}
        annot.get_Calibration_all.return_value = {'00': {}, '04': {}, '19': {}}
        config = self.config()
        subject = {'subject_id': 'dna', 'training_config': config['train'], 'evaluation_config': config['evaluation']}
        validate_dna_inputs(subject, rgb, annot)
        annot.get_SMPLx.return_value = {'transl': [0] * 10}
        with self.assertRaisesRegex(SystemExit, 'synchronized frames'):
            validate_dna_inputs(subject, rgb, annot)


class ExplicitEvaluationTests(MMLPEvaluationTests):
    def test_configured_view_and_pose_ranges(self):
        path = self.owner / 'evaluation_split.json'
        metadata = json.loads(path.read_text())
        metadata['evaluation'] = {
            'novel_view': {'camera_ids': [19], 'frame_range': {'start': 4, 'stop': 5}},
            'novel_pose': {'camera_ids': [4], 'frame_range': {'start': 6, 'stop': 8}}}
        path.write_text(json.dumps(metadata))
        for kind in ('images', 'masks'):
            for frame in (6, 7):
                (self.root / 'source' / kind / 'cam04' / f'{frame:08d}.png').touch()
        _, _, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'heldout')
        self.assertEqual([(p[0], p[1]) for p in pairs], [('cam19', 4)])
        _, _, pairs, audit = evaluation.evaluation_plan(self.checkpoint, 'novel-pose')
        self.assertEqual([(p[0], p[1]) for p in pairs], [('cam04', 6), ('cam04', 7)])
        with self.assertRaisesRegex(ValueError, 'selected split'):
            evaluation.evaluation_plan(self.checkpoint, 'novel-pose', frames=[2])
