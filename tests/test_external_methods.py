from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import yaml
except ModuleNotFoundError:
    yaml = types.ModuleType("yaml")
    yaml.safe_dump = lambda value, sort_keys=False: json.dumps(value)
    yaml.safe_load = json.loads
    sys.modules["yaml"] = yaml

from avatar_creation.methods.animatable_gaussians import (
    _upstream_environment,
    prepare_subject as prepare_animatable,
)
from avatar_creation.methods.mmlphuman import prepare_subject as prepare_mmlphuman
from avatar_creation.methods.mmlphuman import stage_dna_subject

from avatar_creation.methods.external_data import (
    _ignore_missing_rmtree_entry,
    camera_ids,
    numeric_frame_ids,
    numeric_frame_ids_union,
    safe_subject_id,
    stage_thuman_subject,
)
from avatar_creation.methods.registry import MethodSpec, discover_methods, required_file_errors


def _subject(root: Path) -> dict:
    images = root / "images"
    masks = root / "masks"
    for camera in ("cam00", "cam01"):
        (images / camera).mkdir(parents=True)
        (masks / camera).mkdir(parents=True)
        for frame in (0, 1):
            (images / camera / f"{frame:08d}.jpg").write_bytes(b"P6\n1 1\n255\n\0\0\0")
            (masks / camera / f"{frame:08d}.jpg").write_bytes(b"P5\n1 1\n255\n\xff")
    (root / "calibration.json").write_text("{}", encoding="utf-8")
    np.savez(root / "smpl_params.npz", betas=np.zeros((1, 10), dtype=np.float32))
    return {
        "subject_id": "subject/01",
        "subject_path": str(root),
        "images": str(images),
        "masks": str(masks),
        "calibration": str(root / "calibration.json"),
        "smplx": str(root / "smpl_params.npz"),
    }


class ExternalMethodAdapterTests(unittest.TestCase):
    def test_staging_cleanup_ignores_entries_already_removed(self) -> None:
        missing = FileNotFoundError(2, "No such file or directory", "frame.jpg")
        _ignore_missing_rmtree_entry(
            os.unlink, "frame.jpg", (FileNotFoundError, missing, None)
        )

        denied = PermissionError(13, "Permission denied", "frame.jpg")
        with self.assertRaises(PermissionError):
            _ignore_missing_rmtree_entry(
                os.unlink, "frame.jpg", (PermissionError, denied, None)
            )

    def test_shared_staging_is_read_only_and_dependency_free(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            data_dir = stage_thuman_subject(subject, Path(output_tmp) / "data")
            self.assertTrue((data_dir / "images").is_symlink())
            self.assertTrue((data_dir / "masks").is_symlink())
            self.assertEqual(camera_ids(data_dir / "images"), [0, 1])
            self.assertEqual(numeric_frame_ids(data_dir / "images"), [0, 1])
            self.assertEqual(safe_subject_id("subject/01"), "subject_01")
            self.assertFalse((source / "smpl_pos_map").exists())

    def test_thuman_style_staging_limits_views_and_synchronized_frames(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            calibration = {
                "cam00": {"K": [1]},
                "cam01": {"K": [2]},
            }
            (source / "calibration.json").write_text(json.dumps(calibration))
            destination = Path(output_tmp) / "data"

            staged = stage_thuman_subject(
                subject, destination, views=1, time_steps=1
            )

            self.assertEqual([path.name for path in (staged / "images").iterdir()], ["cam00"])
            self.assertEqual(
                [path.name for path in (staged / "images" / "cam00").iterdir()],
                ["00000000.jpg"],
            )
            self.assertEqual(
                [path.name for path in (staged / "masks" / "cam00").iterdir()],
                ["00000000.jpg"],
            )
            self.assertEqual(
                list(json.loads((staged / "calibration.json").read_text())),
                ["cam00"],
            )
            self.assertTrue((staged / "smpl_params.npz").is_symlink())
            self.assertIs(
                stage_thuman_subject(subject, destination, views=1, time_steps=1),
                destination,
            )

    def test_thuman_staging_keeps_fixed_timeline_and_invalid_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            calibration = {"cam00": {"K": [1]}, "cam01": {"K": [2]}}
            (source / "calibration.json").write_text(json.dumps(calibration))
            (source / "gaussian").mkdir()
            (source / "gaussian/lbs_weights_grid.npz").touch()
            (source / "images/cam00/00000001.jpg").unlink()
            (source / "masks/cam00/00000001.jpg").unlink()
            for camera in ("cam00", "cam01"):
                (source / f"images/{camera}/00000002.jpg").touch()
                (source / f"masks/{camera}/00000002.jpg").touch()

            staged, config_path = prepare_mmlphuman(
                subject, Path(output_tmp), {"test": {}}, views=2, time_steps=3
            )

            self.assertEqual(
                sorted(path.name for path in (staged / "images/cam00").iterdir()),
                ["00000000.jpg", "00000002.jpg"],
            )
            self.assertEqual(
                sorted(path.name for path in (staged / "images/cam01").iterdir()),
                ["00000000.jpg", "00000001.jpg", "00000002.jpg"],
            )
            self.assertEqual(numeric_frame_ids_union(staged / "images"), [0, 1, 2])
            validity = json.loads((staged / "frame_validity.json").read_text())
            self.assertEqual(validity["invalid_frames"], {"cam00": [1], "cam01": []})
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["begin_ith_frame"], 0)
            self.assertEqual(config["num_train_frame"], 3)

    def test_all_staged_methods_keep_globally_missing_timesteps(self) -> None:
        for adapter in (prepare_mmlphuman, prepare_animatable):
            with self.subTest(adapter=adapter.__module__), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "source"
                subject = _subject(source)
                (source / "calibration.json").write_text(json.dumps({
                    "cam00": {"K": [1]}, "cam01": {"K": [2]},
                }))
                (source / "gaussian").mkdir()
                (source / "gaussian/lbs_weights_grid.npz").touch()
                # Frame 0 is absent everywhere; frame 2 exists but is outside T=2.
                for camera in ("cam00", "cam01"):
                    (source / f"images/{camera}/00000000.jpg").unlink()
                    (source / f"images/{camera}/00000002.jpg").touch()
                    (source / f"masks/{camera}/00000002.jpg").touch()
                result = adapter(subject, Path(tmp) / "output",
                                 {"train": {"data": {}}, "test": {"data": {}}},
                                 views=2, time_steps=2)
                staged, config_path = result[:2]
                config = yaml.safe_load(config_path.read_text())
                self.assertEqual(numeric_frame_ids_union(staged / "images"), [1])
                validity = json.loads((staged / "frame_validity.json").read_text())
                self.assertEqual(validity["frame_ids"], {"start": 0, "end": 1, "count": 2})
                self.assertEqual(validity["invalid_frames"], {"cam00": [0], "cam01": [0]})
                if adapter is prepare_mmlphuman:
                    self.assertEqual(config["begin_ith_frame"], 0)
                    self.assertEqual(config["num_train_frame"], 2)
                    self.assertEqual(config["test"]["begin_ith_frame"], 1)
                else:
                    self.assertEqual(config["train"]["data"]["frame_range"], [0, 2, 1])
                    self.assertEqual(config["train"]["eval_training_ids"], [1, 0])
                    self.assertEqual((staged / "missing_img_files.txt").read_text().splitlines(),
                                     ["images/cam00/00000000.jpg", "images/cam01/00000000.jpg"])

    def test_new_methods_are_registered_with_isolated_environments(self) -> None:
        methods = discover_methods()
        self.assertEqual(methods["mmlphuman"].conda_env, "mmlphuman")
        self.assertEqual(
            methods["animatable-gaussians"].conda_env, "animatable-gaussians"
        )
        self.assertNotEqual(
            methods["mmlphuman"].conda_env,
            methods["animatable-gaussians"].conda_env,
        )
        self.assertIn("h5py", " ".join(methods["mmlphuman"].environment_check))
        self.assertIn("PointInterpolant", " ".join(methods["mmlphuman"].environment_check))
        self.assertIn("smpl_model/smplx/SMPLX_NEUTRAL.npz", methods["mmlphuman"].required_files)
        self.assertIn("smpl_files/smplx/SMPLX_NEUTRAL.npz", methods["animatable-gaussians"].required_files)
        self.assertTrue(methods["pig-avatar"].required_any_files)

    def test_method_required_files_report_missing_assets_and_alternatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = MethodSpec(
                name="fixture",
                command=[],
                cwd=root,
                required_files=("SMPLX_NEUTRAL.npz",),
                required_any_files=(("$VCI_TEST_SMPLX", "fallback.npz"),),
            )
            errors = required_file_errors(spec)
            self.assertTrue(any("SMPLX_NEUTRAL.npz" in error for error in errors))
            self.assertTrue(any("alternative required files" in error for error in errors))
            (root / "SMPLX_NEUTRAL.npz").touch()
            (root / "fallback.npz").touch()
            self.assertEqual(required_file_errors(spec), [])

    def test_external_installers_clone_pinned_revisions(self) -> None:
        scripts = Path(__file__).resolve().parents[1] / "scripts" / "setup"
        expected = {
            "setup_mmlphuman_env.sh": "6668509284c85fcb0f93cd7365ec8f39390ff251",
            "setup_animatable_gaussians_env.sh": "2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114",
        }
        for name, revision in expected.items():
            with self.subTest(script=name):
                script = (scripts / name).read_text(encoding="utf-8")
                self.assertLess(script.index("git clone --no-checkout"), script.index("checkout --detach"))
                self.assertLess(script.index("checkout --detach"), script.index("submodule update --init --recursive"))
                self.assertIn(revision, script)

    def test_pig_avatar_installer_uses_a_pinned_checkout(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup"
            / "setup_pig_avatar_env.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('9ec01d2ecde940f6cb3c572c387fa27fada67796', script)
        self.assertIn('checkout --detach "$PIG_REPO_COMMIT"', script)
        self.assertIn('diff --cached --quiet', script)
        self.assertIn("'libblas=*=*openblas'", script)
        self.assertIn("'liblapack=*=*openblas'", script)
        self.assertIn('PYTORCH_VERSION="${PIG_PYTORCH_VERSION:-2.7.1}"', script)
        self.assertIn('CUDA_VERSION="${PIG_CUDA_VERSION:-12.8}"', script)
        self.assertIn('"cuda-nvcc_linux-64=$CUDA_VERSION"', script)
        self.assertIn("download.pytorch.org/whl/cu128", script)
        self.assertIn("torch.cuda.get_arch_list()", script)
        self.assertIn("targets/x86_64-linux", script)
        self.assertIn("cuda_runtime_api.h", script)
        self.assertIn('MAX_JOBS="${MAX_JOBS:-1}"', script)
        self.assertIn("TORCH_CUDA_ARCH_LIST", script)
        self.assertIn("torch-fidelity debugpy", script)
        self.assertEqual(script.count("validate_torch_cuda \""), 2)
        self.assertLess(
            script.index('validate_torch_cuda "before native dependency builds"'),
            script.index("Installing PiG-Avatar Python dependencies"),
        )
        self.assertLess(
            script.index("'liblapack=*=*openblas'"),
            script.index("python -m pip install ext/pcgrid"),
        )

        check = " ".join(discover_methods()["pig-avatar"].environment_check)
        self.assertIn("torch.cuda.get_device_capability()", check)
        self.assertIn("torch.cuda.get_arch_list()", check)
        self.assertIn("targets", check)
        self.assertIn("x86_64-linux", check)
        self.assertIn("CPLUS_INCLUDE_PATH", check)
        self.assertIn("TORCH_CUDA_ARCH_LIST", check)
        self.assertIn("cuda:0", check)

    def test_mmlphuman_setup_includes_dna_reader_dependencies(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "scripts" / "setup" / "setup_mmlphuman_env.sh").read_text(encoding="utf-8")
        self.assertIn("6668509284c85fcb0f93cd7365ec8f39390ff251", script)
        self.assertNotIn("git submodule update --init external/mmlphuman", script)
        self.assertIn("h5py", script)
        self.assertIn("opencv-python==4.10.0.84", script)
        self.assertIn("numpy<2.0", script)
        self.assertIn("PoissonRecon", script)
        self.assertIn("PointInterpolant", script)
        self.assertIn("262b0f539d404057d1f36e1adc07fc9388678899", script)
        self.assertIn("mmlphuman-lpips-reset.patch", script)
        patch = Path(__file__).resolve().parents[1] / "scripts" / "setup" / "patches" / "mmlphuman-lpips-reset.patch"
        self.assertIn("lpips_model.reset()", patch.read_text(encoding="utf-8"))

    def test_animatable_setup_handles_special_wheels_and_native_modules(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "scripts" / "setup" / "setup_animatable_gaussians_env.sh").read_text(encoding="utf-8")
        self.assertIn("2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114", script)
        self.assertNotIn("git submodule update --init external/AnimatableGaussians", script)
        self.assertIn("dl.fbaipublicfiles.com/pytorch3d", script)
        self.assertIn("grep -Ev", script)
        self.assertIn("pytorch3d|setuptools", script)
        check = " ".join(discover_methods()["animatable-gaussians"].environment_check)
        for module in (
            "diff_gaussian_rasterization_depth_alpha",
            "fused",
            "posevocab_custom_ops",
            "root_finding",
            "upfirdn2d",
        ):
            self.assertIn(module, check)

    def test_mmlphuman_stages_dna_smc_records_in_run_output(self) -> None:
        if importlib.util.find_spec("cv2") is None or importlib.util.find_spec("numpy") is None:
            self.skipTest("DNA staging dependencies are installed in the MMLP environment")
        import numpy as np

        class FakeReader:
            def __init__(self, _path):
                self.smc = None

            def get_Camera_5mp_info(self):
                return {"num_frame": 3}

            def get_Calibration_all(self):
                return {
                    "00": {"K": np.eye(3), "RT": np.eye(4), "D": np.zeros(5)},
                    "01": {"K": np.eye(3), "RT": np.eye(4), "D": np.zeros(5)},
                }

            def get_SMPLx(self):
                fullpose = np.zeros((3, 55, 3), dtype=np.float32)
                return {
                    "betas": np.ones((3, 10), dtype=np.float32),
                    "expression": np.zeros((3, 10), dtype=np.float32),
                    "fullpose": fullpose,
                    "transl": np.zeros((3, 3), dtype=np.float32),
                }

            def get_img(self, *_args, **_kwargs):
                return np.full((4, 5, 3), 127, dtype=np.uint8)

            def get_mask(self, *_args, **_kwargs):
                return np.full((4, 5), 255, dtype=np.uint8)

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            rgb = source / "subject.smc"
            annot = source / "subject_annots.smc"
            rgb.touch()
            annot.touch()
            subject = {
                "dataset": "dna",
                "subject_id": "0166_04",
                "subject_path": str(source),
                "images": str(rgb),
                "masks": str(annot),
                "calibration": str(annot),
                "smplx": str(annot),
            }
            destination = Path(output_tmp) / "data"
            staged = stage_dna_subject(
                subject, destination, views=1, time_steps=2, resolution=3, reader_class=FakeReader
            )
            self.assertTrue((staged / "images" / "cam00" / "00000001.jpg").is_file())
            self.assertTrue((staged / "masks" / "cam00" / "00000001.jpg").is_file())
            import cv2
            self.assertEqual(cv2.imread(str(staged / "images" / "cam00" / "00000001.jpg")).shape[:2], (2, 3))
            calibration = json.loads((staged / "calibration.json").read_text())
            self.assertEqual(list(calibration), ["cam00"])
            self.assertAlmostEqual(calibration["cam00"]["K"][0], 0.6)
            self.assertAlmostEqual(calibration["cam00"]["K"][4], 0.6)
            params = np.load(staged / "smpl_params.npz")
            self.assertEqual(params["body_pose"].shape, (2, 63))
            self.assertEqual(params["left_hand_pose"].shape, (2, 45))
            self.assertIs(
                stage_dna_subject(
                    subject, destination, views=1, time_steps=2, resolution=3, reader_class=FakeReader
                ),
                destination,
            )

            subject["training_config"] = {
                "camera_ids": [1], "frame_range": {"start": 1, "stop": 3}, "resolution": 3}
            subject["evaluation_config"] = {
                "novel_pose": {"camera_ids": [1], "frame_range": {"start": 0, "stop": 1}}}
            stage_dna_subject(subject, destination, views=1, time_steps=2,
                              resolution=3, reader_class=FakeReader)
            self.assertEqual(sorted(p.stem for p in (destination / "images/cam00").iterdir()),
                             ["00000001", "00000002"])
            self.assertEqual(json.loads((destination / "camera_id_mapping.json").read_text()),
                             {"cam00": 1})
            with np.load(destination / "smpl_params.npz") as explicit_params:
                self.assertEqual(explicit_params["body_pose"].shape, (3, 63))

    def test_mmlphuman_stages_links_and_writes_run_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            (source / "calibration.json").write_text(
                json.dumps({"cam00": {"K": [1]}, "cam01": {"K": [2]}})
            )
            (source / "gaussian").mkdir()
            (source / "gaussian" / "lbs_weights_grid.npz").touch()
            (source / "gaussian" / "template.ply").touch()
            template = {
                "train_cam_ids": [],
                "begin_ith_frame": 0,
                "num_train_frame": 1,
                "frame_interval": 1,
                "test": {},
            }

            data_dir, config_path = prepare_mmlphuman(
                subject, Path(output_tmp), template, views=1, time_steps=1
            )

            self.assertTrue((data_dir / "images" / "cam00").is_dir())
            self.assertTrue((data_dir / "gaussian" / "lbs_weights_grid.npz").is_symlink())
            self.assertFalse((data_dir / "gaussian" / "template.ply").exists())
            staged_calibration = json.loads((data_dir / "calibration.json").read_text())
            self.assertEqual(
                staged_calibration["cam00"]["distCoeff"], [0.0] * 5
            )
            source_calibration = json.loads((source / "calibration.json").read_text())
            self.assertNotIn("distCoeff", source_calibration["cam00"])
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["train_cam_ids"], [0])
            self.assertEqual(config["num_train_frame"], 1)
            split = json.loads(
                (config_path.parent / "evaluation_split.json").read_text()
            )
            self.assertEqual(split["train_cameras"], [0])
            self.assertEqual(split["train_views"], 1)
            self.assertEqual(split["resolution"], "source")
            self.assertFalse((source / "gaussian" / "init_body_points.ply").exists())

    def test_mmlphuman_aliases_png_media_to_hardcoded_jpg_names(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            for media_name in ("images", "masks"):
                jpg = source / media_name / "cam00" / "00000000.jpg"
                jpg.rename(jpg.with_suffix(".png"))
            (source / "calibration.json").write_text(
                json.dumps({"cam00": {"K": [1]}, "cam01": {"K": [2]}})
            )
            (source / "gaussian").mkdir()
            (source / "gaussian" / "lbs_weights_grid.npz").touch()

            data_dir, _ = prepare_mmlphuman(
                subject, Path(output_tmp), {"test": {}}, views=1, time_steps=1
            )

            for media_name in ("images", "masks"):
                alias = data_dir / media_name / "cam00" / "00000000.jpg"
                self.assertTrue(alias.is_symlink())
                self.assertEqual(alias.resolve().suffix, ".png")
                self.assertFalse(
                    (source / media_name / "cam00" / "00000000.jpg").exists()
                )

    def test_mmlphuman_materializes_supported_smplx_beta_count(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            source_betas = np.arange(16, dtype=np.float32)[None]
            body_pose = np.zeros((2, 63), dtype=np.float32)
            np.savez(
                source / "smpl_params.npz",
                betas=source_betas,
                body_pose=body_pose,
            )
            (source / "calibration.json").write_text(
                json.dumps({"cam00": {"K": [1]}, "cam01": {"K": [2]}})
            )
            (source / "gaussian").mkdir()
            (source / "gaussian" / "lbs_weights_grid.npz").touch()

            data_dir, _ = prepare_mmlphuman(
                subject, Path(output_tmp), {"test": {}}, views=1, time_steps=1
            )

            staged_params = data_dir / "smpl_params.npz"
            self.assertFalse(staged_params.is_symlink())
            with np.load(staged_params) as params:
                np.testing.assert_array_equal(params["betas"], source_betas[:, :10])
                np.testing.assert_array_equal(params["body_pose"], body_pose)
            with np.load(source / "smpl_params.npz") as params:
                np.testing.assert_array_equal(params["betas"], source_betas)

    def test_mmlphuman_generates_missing_preprocessing_for_thuman_style_data(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            (source / "calibration.json").write_text(
                json.dumps({"cam00": {"K": [1]}, "cam01": {"K": [2]}})
            )

            def generate(data_dir, _subject_out):
                weights = data_dir / "gaussian" / "lbs_weights_grid.npz"
                weights.touch()

            with patch("avatar_creation.methods.mmlphuman._generate_lbs_weights", side_effect=generate) as generator:
                data_dir, _ = prepare_mmlphuman(
                    subject,
                    Path(output_tmp),
                    {"test": {}},
                    views=1,
                    time_steps=1,
                )

            generator.assert_called_once()
            self.assertTrue((data_dir / "gaussian" / "lbs_weights_grid.npz").is_file())

    def test_animatable_upstream_environment_excludes_vci_pythonpath(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        retained = project_root.parent / "retained-pythonpath"
        with patch.dict(os.environ, {
            "VCI_DATASET_REPO": str(project_root),
            "PYTHONPATH": os.pathsep.join((str(project_root), str(retained))),
        }):
            env = _upstream_environment()

        self.assertEqual(env["PYTHONPATH"], str(retained))
        self.assertEqual(env["OPENCV_IO_ENABLE_OPENEXR"], "1")

    def test_animatable_applies_shared_training_selection(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            source_betas = np.arange(16, dtype=np.float32)[None]
            np.savez(source / "smpl_params.npz", betas=source_betas)
            for media_name in ("images", "masks"):
                jpg = source / media_name / "cam00" / "00000000.jpg"
                jpg.rename(jpg.with_suffix(".png"))
            (source / "calibration.json").write_text(json.dumps({
                "cam00": {"K": [1]}, "cam01": {"K": [2]},
            }))
            (source / "template.ply").touch()
            (source / "cano_weight_volume.npz").touch()
            template = {"train": {"data": {}}, "test": {"data": {}}}
            data_dir, config_path, _ = prepare_animatable(
                subject, Path(output_tmp), template, views=1, time_steps=1, resolution=0
            )
            self.assertFalse((data_dir / "template.ply").exists())
            self.assertTrue((data_dir / "cano_weight_volume.npz").is_symlink())
            staged_calibration = json.loads((data_dir / "calibration.json").read_text())
            self.assertEqual(staged_calibration["cam00"]["imgSize"], [1, 1])
            source_calibration = json.loads((source / "calibration.json").read_text())
            self.assertNotIn("imgSize", source_calibration["cam00"])
            for media_name in ("images", "masks"):
                alias = data_dir / media_name / "cam00" / "00000000.jpg"
                self.assertEqual(alias.resolve().suffix, ".png")
            with np.load(data_dir / "smpl_params.npz") as params:
                np.testing.assert_array_equal(params["betas"], source_betas[:, :10])
            with np.load(source / "smpl_params.npz") as params:
                np.testing.assert_array_equal(params["betas"], source_betas)
            config = yaml.safe_load(config_path.read_text())
            self.assertEqual(config["train"]["data"]["used_cam_ids"], [0])
            self.assertEqual(config["train"]["data"]["frame_range"], [0, 1, 1])
            self.assertEqual(numeric_frame_ids(data_dir / "images"), [0])
            self.assertEqual(camera_ids(data_dir / "images"), [0])
            self.assertTrue((source / "images/cam01/00000001.jpg").is_file())

    def test_animatable_config_points_only_at_staged_output(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            subject = _subject(source)
            (source / "calibration.json").write_text(json.dumps({
                "cam00": {"K": [1]}, "cam01": {"K": [2]},
            }))
            template = {
                "train": {
                    "data": {},
                    "pretrained_dir": "old",
                    "prev_ckpt": "old",
                },
                "test": {"data": {}},
            }
            data_dir, config_path, prepared = prepare_animatable(
                subject, Path(output_tmp), template
            )

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["train"]["data"]["data_dir"], str(data_dir))
            split = json.loads(
                (config_path.parent / "evaluation_split.json").read_text()
            )
            self.assertEqual(split["train_cameras"], [0, 1])
            self.assertEqual(split["train_views"], 2)
            self.assertEqual(split["resolution"], "source")
            self.assertEqual(config["train"]["data"]["frame_range"], [0, 2, 1])
            self.assertEqual(config["train"]["data"]["used_cam_ids"], [0, 1])
            self.assertTrue(config["train"]["net_ckpt_dir"].startswith(output_tmp))
            self.assertFalse(prepared)
            self.assertFalse((source / "smpl_pos_map").exists())


if __name__ == "__main__":
    unittest.main()
