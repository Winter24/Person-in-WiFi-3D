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
    def test_petr_wifi_config_uses_relative_paths_and_linear_b0_backbone(self):
        cfg = _load_config_module("configs/wifi/petr_wifi.py")

        expected_meta_keys = ["img_shape", "ori_shape", "pad_shape", "img_name"]

        self.assertEqual(cfg.data_root, "data/wifipose")
        self.assertEqual(cfg.model["backbone"]["type"], "WifiInputAdapter")
        self.assertEqual(cfg.model["backbone"]["mode"], "linear")
        self.assertEqual(cfg.model["backbone"]["in_channels"], 60)
        self.assertEqual(cfg.model["backbone"]["embed_dims"], 256)
        self.assertIsNone(cfg.model["neck"])
        self.assertIsNone(cfg.model["bbox_head"]["loss_bone"])
        self.assertEqual(cfg.data["train"]["dataset_root"], "data/wifipose/train_data")
        self.assertEqual(cfg.data["val"]["dataset_root"], "data/wifipose/test_data")
        self.assertEqual(cfg.data["test"]["dataset_root"], "data/wifipose/test_data")
        self.assertEqual(cfg.train_pipeline[-1]["meta_keys"], expected_meta_keys)
        self.assertEqual(cfg.test_pipeline[-1]["meta_keys"], expected_meta_keys)

    def test_b0_training_hparams_match_legacy_cvpr_except_epochs(self):
        cfg = _load_config_module("configs/wifi/petr_wifi.py")
        bbox_head = cfg.model["bbox_head"]
        assigner = cfg.model["train_cfg"]["assigner"]

        self.assertEqual(cfg.data["samples_per_gpu"], 64)
        self.assertEqual(bbox_head["loss_cls"]["loss_weight"], 2.0)
        self.assertEqual(bbox_head["loss_kpt"]["loss_weight"], 70.0)
        self.assertEqual(bbox_head["loss_kpt_rpn"]["loss_weight"], 70.0)
        self.assertEqual(bbox_head["loss_kpt_refine"]["loss_weight"], 80.0)
        self.assertEqual(assigner["cls_cost"]["weight"], 2.0)
        self.assertEqual(assigner["kpt_cost"]["weight"], 70.0)
        self.assertEqual(cfg.optimizer["lr"], 2e-5)
        self.assertEqual(cfg.lr_config["step"], [80])
        self.assertEqual(cfg.runner["max_epochs"], 10)

    def test_petr_wifi_mamba_config_switches_backbone_to_spectral_mode(self):
        cfg = _load_config_module("configs/wifi/petr_wifi_mamba.py")

        self.assertEqual(cfg.model["backbone"]["type"], "WifiInputAdapter")
        self.assertEqual(cfg.model["backbone"]["mode"], "spectral")
        self.assertEqual(cfg.model["bbox_head"]["transformer"]["encoder"]["num_layers"], 6)

    def test_wifi_input_adapter_supports_linear_and_spectral_modes(self):
        source = _read("opera/models/utils/spectral_tokenizer.py")

        self.assertIn("BACKBONES", source)
        self.assertIn("mmdet.models.builder", source)
        self.assertIn("@MMDET_BACKBONES.register_module()", source)
        self.assertIn("@OPERA_BACKBONES.register_module()", source)
        self.assertIn("class WifiInputAdapter", source)
        self.assertIn("mode='spectral'", source)
        self.assertIn("self.linear_proj", source)
        self.assertIn("if self.mode == 'linear'", source)

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

    def test_petr_source_normalizes_tensor_inputs_for_forward_test(self):
        source = _read("opera/models/detectors/petr.py")

        self.assertIn("def forward_test", source)
        self.assertIn("if isinstance(imgs, list):", source)
        self.assertIn("img = imgs[0]", source)
        self.assertIn("if hasattr(img_metas, 'data'):", source)
        self.assertIn("elif isinstance(img_metas, list) and img_metas and hasattr(img_metas[0], 'data'):", source)
        self.assertIn("img_meta['batch_input_shape'] = tuple(img.size()[-2:])", source)
        self.assertIn("return self.simple_test(img, img_metas, **kwargs)", source)

    def test_petr_source_remaps_legacy_linear_head_checkpoint_keys(self):
        source = _read("opera/models/detectors/petr.py")

        self.assertIn("head.weight", source)
        self.assertIn("head.bias", source)
        self.assertIn("backbone.linear_proj.weight", source)
        self.assertIn("backbone.linear_proj.bias", source)

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

    def test_bone_stats_paths_are_anchored_to_project_root(self):
        for relative_path in (
            "opera/models/dense_heads/petr_head.py",
            "opera/models/dense_heads/wi_tidar_head.py",
        ):
            source = _read(relative_path)
            self.assertIn("__file__", source)
            self.assertIn("project_root", source)
            self.assertIn("os.path.join(current_dir, '../../..')", source)
            self.assertIn(
                "os.path.join(project_root, 'gt_bone_stats.json')",
                source)

    def test_compute_bone_stats_writes_output_to_project_root(self):
        source = _read("tools/analysis/compute_bone_stats.py")

        self.assertIn("project_root", source)
        self.assertIn(
            "output_path = os.path.join(project_root, 'gt_bone_stats.json')",
            source)

    def test_selected_files_no_longer_have_colab_headers(self):
        for relative_path in (
            "configs/wifi/wi_tidir_wifi.py",
            "opera/models/dense_heads/flow_components.py",
            "opera/models/utils/rectified_flow.py",
            "tools/train.py",
        ):
            source = _read(relative_path)
            self.assertNotIn("%%writefile", source, msg=relative_path)
            self.assertNotIn("# @title", source, msg=relative_path)

    def test_witidir_config_keeps_only_bone_loss(self):
        source = _read("configs/wifi/wi_tidir_wifi.py")

        self.assertIn("loss_bone=dict(type='BoneLengthLoss', loss_weight=2.0)", source)
        self.assertNotIn("loss_limb", source)
        self.assertIn("num_layers=6", source)

    def test_witidar_head_source_removes_limb_loss_logic(self):
        source = _read("opera/models/dense_heads/wi_tidar_head.py")

        self.assertIn("losses_cls, losses_kpt, losses_bone, losses_flow", source)
        self.assertNotIn("loss_limb", source)
        self.assertNotIn("losses_limb", source)
        self.assertNotIn("LimbLoss", source)

    def test_losses_init_no_longer_imports_limb_loss(self):
        source = _read("opera/models/losses/__init__.py")

        self.assertIn("BoneLengthLoss", source)
        self.assertNotIn("limb_loss", source)
        self.assertNotIn("LimbLoss", source)


if __name__ == "__main__":
    unittest.main()
