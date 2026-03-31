import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _load_config_module(relative_path: str):
    config_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WifiArchitectureTests(unittest.TestCase):
    def test_petr_wifi_config_uses_relative_paths_and_wifi_backbone(self):
        cfg = _load_config_module("configs/wifi/petr_wifi.py")

        expected_meta_keys = ["img_shape", "ori_shape", "pad_shape", "img_name"]

        self.assertEqual(cfg.data_root, "data/wifipose")
        self.assertEqual(cfg.model["backbone"]["type"], "SpectralTokenizer")
        self.assertIsNone(cfg.model["neck"])
        self.assertEqual(cfg.data["train"]["dataset_root"], "data/wifipose/train_data")
        self.assertEqual(cfg.data["val"]["dataset_root"], "data/wifipose/test_data")
        self.assertEqual(cfg.data["test"]["dataset_root"], "data/wifipose/test_data")
        self.assertEqual(cfg.train_pipeline[-1]["meta_keys"], expected_meta_keys)
        self.assertEqual(cfg.test_pipeline[-1]["meta_keys"], expected_meta_keys)

    def test_spectral_tokenizer_is_registered_as_backbone_in_source(self):
        source = _read("opera/models/utils/spectral_tokenizer.py")

        self.assertIn("BACKBONES", source)
        self.assertIn("mmdet.models.builder", source)
        self.assertIn("@MMDET_BACKBONES.register_module()", source)
        self.assertIn("@OPERA_BACKBONES.register_module()", source)

    def test_wimamba_encoder_is_registered_in_mmcv_transformer_sequence_registry(self):
        source = _read("opera/models/backbones/wimamba.py")

        self.assertIn("mmcv.cnn.bricks.transformer", source)
        self.assertIn(
            "@MMCV_TRANSFORMER_LAYER_SEQUENCE.register_module()",
            source)
        self.assertIn("@TRANSFORMER_LAYER_SEQUENCE.register_module()", source)

    def test_wifi_pose_source_adds_required_meta_fields(self):
        source = _read("opera/datasets/wifi_pose.py")

        self.assertIn("img_shape=img_shape", source)
        self.assertIn("ori_shape=img_shape", source)
        self.assertIn("pad_shape=img_shape", source)

    def test_petr_source_uses_backbone_extract_feat_for_wifi(self):
        source = _read("opera/models/detectors/petr.py")

        self.assertIn("import warnings", source)
        self.assertIn("def extract_feat", source)
        self.assertIn("img.reshape(bs, -1, channel)", source)
        self.assertIn("self.backbone(x)", source)
        self.assertNotIn("self.head =", source)

    def test_petr_source_initializes_via_single_stage_detector_to_support_neck(self):
        source = _read("opera/models/detectors/petr.py")

        self.assertIn(
            "from mmdet.models.detectors.single_stage import SingleStageDetector",
            source)
        self.assertIn("neck=None", source)
        self.assertIn("SingleStageDetector.__init__(", source)

    def test_single_gpu_test_source_skips_rgb_visualization_for_wifi(self):
        source = _read("opera/apis/test.py")

        self.assertIn("img_tensor.dim() >= 4", source)
        self.assertIn("tensor2imgs", source)

    def test_visualization_scripts_no_longer_hardcode_linux_paths(self):
        for relative_path in (
            "tools/multiperson_visualize.py",
            "tools/one_person_visualize.py",
        ):
            source = _read(relative_path)
            self.assertNotIn("/home/winter24/", source)
            self.assertTrue(
                "PROJECT_ROOT" in source or "argparse" in source,
                msg=f"{relative_path} should derive its paths dynamically.",
            )


if __name__ == "__main__":
    unittest.main()
