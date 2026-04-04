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
        self.assertEqual(cfg.train_pipeline[-1]["meta_keys"], [])
        self.assertEqual(cfg.test_pipeline[0]["type"], "mmdet.MultiScaleFlipAug")
        self.assertEqual(
            cfg.test_pipeline[0]["transforms"][-1]["meta_keys"],
            [])

    def test_b0_training_hparams_match_cvpr_paper_recipe(self):
        cfg = _load_config_module("configs/wifi/petr_wifi.py")
        bbox_head = cfg.model["bbox_head"]
        assigner = cfg.model["train_cfg"]["assigner"]

        self.assertEqual(cfg.data["samples_per_gpu"], 32)
        self.assertEqual(bbox_head["loss_cls"]["loss_weight"], 4.0)
        self.assertEqual(bbox_head["loss_kpt"]["loss_weight"], 35.0)
        self.assertEqual(bbox_head["loss_kpt_rpn"]["loss_weight"], 35.0)
        self.assertEqual(bbox_head["loss_kpt_refine"]["loss_weight"], 35.0)
        self.assertEqual(bbox_head["loss_oks_refine"]["loss_weight"], 3.0)
        self.assertEqual(assigner["cls_cost"]["weight"], 4.0)
        self.assertEqual(assigner["kpt_cost"]["weight"], 35.0)
        self.assertEqual(cfg.optimizer["type"], "AdamW")
        self.assertEqual(cfg.optimizer["lr"], 2e-5)
        self.assertEqual(cfg.optimizer["betas"], (0.9, 0.999))
        self.assertEqual(cfg.optimizer["weight_decay"], 1e-4)
        self.assertEqual(cfg.lr_config["step"], [450])
        self.assertEqual(cfg.runner["max_epochs"], 500)

    def test_petr_wifi_mamba_config_switches_backbone_to_spectral_mode(self):
        cfg = _load_config_module("configs/wifi/petr_wifi_mamba.py")

        self.assertEqual(cfg.model["backbone"]["type"], "WifiInputAdapter")
        self.assertEqual(cfg.model["backbone"]["mode"], "spectral")
        self.assertEqual(cfg.model["bbox_head"]["transformer"]["encoder"]["num_layers"], 6)

    def test_petr_wifi_mamba_source_does_not_override_runtime_hparams(self):
        source = _read("configs/wifi/petr_wifi_mamba.py")

        self.assertNotIn("optimizer = dict(", source)
        self.assertNotIn("log_config = dict(", source)

    def test_petr_wifi_bone_config_enables_bone_loss_for_b0_bone(self):
        cfg = _load_config_module("configs/wifi/petr_wifi_bone.py")
        source = _read("configs/wifi/petr_wifi_bone.py")

        self.assertIn("_base_ = ['./petr_wifi.py']", source)
        self.assertEqual(cfg.model["bbox_head"]["loss_bone"]["type"], "BoneLengthLoss")
        self.assertEqual(cfg.model["bbox_head"]["loss_bone"]["loss_weight"], 2.0)

    def test_petr_wifi_bone_mamba_config_enables_bone_loss_for_b2_bone(self):
        cfg = _load_config_module("configs/wifi/petr_wifi_bone_mamba.py")
        source = _read("configs/wifi/petr_wifi_bone_mamba.py")

        self.assertIn("_base_ = ['./petr_wifi_mamba.py']", source)
        self.assertEqual(cfg.model["bbox_head"]["loss_bone"]["type"], "BoneLengthLoss")
        self.assertEqual(cfg.model["bbox_head"]["loss_bone"]["loss_weight"], 2.0)

    def test_bone_branch_commands_are_documented(self):
        checklist = _read("docs/paper/2026-03-31-experiment-daily-checklist.md")

        self.assertIn("B0_bone", checklist)
        self.assertIn("B1_bone", checklist)
        self.assertIn("B2_bone", checklist)

    def test_wifi_input_adapter_supports_linear_and_spectral_modes(self):
        source = _read("opera/models/utils/spectral_tokenizer.py")

        self.assertIn("BACKBONES", source)
        self.assertIn("mmdet.models.builder", source)
        self.assertIn("@MMDET_BACKBONES.register_module()", source)
        self.assertIn("@OPERA_BACKBONES.register_module()", source)
        self.assertIn("class WifiInputAdapter", source)
        self.assertIn("mode='spectral'", source)
        self.assertIn("self.head = nn.Linear(in_channels, embed_dims)", source)
        self.assertIn("def linear_proj(self):", source)
        self.assertIn("return self.head", source)
        self.assertIn("if self.mode == 'linear'", source)
        self.assertIn("num_spatial=9", source)
        self.assertIn("seq_len=20", source)
        self.assertIn("x.reshape(B, self.num_spatial, self.seq_len, C)", source)
        self.assertIn("torch.fft.rfft(", source)
        self.assertNotIn("torch.fft.rfft2", source)
        self.assertIn("x_out + x_linear", source)
        self.assertIn("self.complex_weight[..., 0], 1.0", source)
        self.assertIn("self.complex_weight[..., 1], 0.0", source)
        self.assertIn("Expected sequence length", source)
        self.assertIn("CRITICAL ASSUMPTION", source)
        self.assertNotIn("xavier_uniform_(self.linear_proj.weight)", source)

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
        self.assertIn("backbone.head.weight", source)
        self.assertIn("backbone.head.bias", source)
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

        self.assertIn("loss_bone=dict(_delete_=True, type='BoneLengthLoss', loss_weight=2.0)", source)
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

    def test_train_entrypoint_defaults_to_seed_42_and_deterministic(self):
        source = _read("tools/train.py")

        self.assertIn("parser.add_argument('--seed', type=int, default=42", source)
        self.assertIn("parser.set_defaults(deterministic=True)", source)
        self.assertIn("--non-deterministic", source)
        self.assertIn("PYTHONHASHSEED", source)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", source)

    def test_training_docs_show_seed_42_and_deterministic_flags(self):
        checklist = _read("docs/paper/2026-03-31-experiment-daily-checklist.md")

        self.assertIn("--seed 42 --deterministic", checklist)
        self.assertIn("PYTHONHASHSEED=42", checklist)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", checklist)

    def test_bone_warmup_hook_is_wired_globally_and_overridden_for_b5(self):
        base_cfg = _read("configs/wifi/petr_wifi.py")
        b5_cfg = _read("configs/wifi/wi_tidir_wifi.py")
        hook_source = _read("opera/core/runner/hooks/bone_warmup_hook.py")
        runner_init = _read("opera/core/runner/__init__.py")

        self.assertIn("BoneLossWarmupHook", hook_source)
        self.assertIn("loss_bone is None", hook_source)
        self.assertIn("warmup_ratio=0.1", base_cfg)
        self.assertIn("ramp_ratio=0.1", base_cfg)
        self.assertIn("target_weight=2.0", base_cfg)
        self.assertIn("target_weight=1.0", b5_cfg)
        self.assertIn("BoneLossWarmupHook", runner_init)


if __name__ == "__main__":
    unittest.main()
