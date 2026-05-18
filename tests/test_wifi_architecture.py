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

    def test_b0_training_hparams_match_current_runtime_recipe(self):
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
        self.assertEqual(cfg.evaluation["interval"], 5)
        self.assertEqual(cfg.checkpoint_config["interval"], 5)
        self.assertEqual(cfg.runner["max_epochs"], 20)

    def test_petr_wifi_mamba_config_switches_backbone_to_spectral_mode(self):
        cfg = _load_config_module("configs/wifi/petr_wifi_mamba.py")

        self.assertEqual(cfg.model["backbone"]["type"], "WifiInputAdapter")
        self.assertEqual(cfg.model["backbone"]["mode"], "spectral")
        self.assertEqual(cfg.model["bbox_head"]["transformer"]["encoder"]["num_layers"], 4)

    def test_petr_wifi_linear_mamba_config_keeps_linear_backbone_and_mamba_encoder(self):
        cfg = _load_config_module("configs/wifi/petr_wifi_linear_mamba.py")

        self.assertEqual(cfg.model["backbone"]["mode"], "linear")
        self.assertEqual(cfg.model["bbox_head"]["transformer"]["encoder"]["type"], "WiMambaEncoder")
        self.assertEqual(cfg.model["bbox_head"]["transformer"]["encoder"]["num_layers"], 4)

    def test_witidir_transformer_flow_config_uses_transformer_encoder_without_mamba(self):
        cfg = _load_config_module("configs/wifi/wi_tidir_wifi_transformer.py")

        self.assertEqual(cfg.model["backbone"]["mode"], "spectral")
        self.assertEqual(cfg.model["bbox_head"]["type"], "opera.WiTiDARHead")
        self.assertEqual(cfg.model["bbox_head"]["transformer_encoder"]["type"], "mmcv.DetrTransformerEncoder")
        self.assertEqual(cfg.model["bbox_head"]["transformer_encoder"]["num_layers"], 6)
        self.assertIsNone(cfg.model["bbox_head"]["mamba_cfg"])
        self.assertIsNone(cfg.model["bbox_head"]["loss_bone"])

    def test_witidir_linear_config_keeps_witidar_stack_and_switches_backbone_to_linear(self):
        cfg = _load_config_module("configs/wifi/wi_tidir_wifi_linear.py")
        source = _read("configs/wifi/wi_tidir_wifi_linear.py")

        self.assertIn("_base_ = ['./wi_tidir_wifi.py']", source)
        self.assertEqual(cfg.model["backbone"]["mode"], "linear")
        self.assertIn("work_dir = './work_dirs/wi_tidir_wifi_linear'", source)

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

        self.assertIn("M0_bone", checklist)
        self.assertIn("M1_bone", checklist)
        self.assertIn("M2_bone", checklist)
        self.assertNotIn("`A1`", checklist)

    def test_paper_plan_uses_m0_to_m5_main_ladder(self):
        plan = _read("docs/paper/2026-03-31-balanced-arxiv-workshop-paper-plan.md")

        self.assertIn(
            "Fast yet Accurate: Bridging the Gap in WiFi Pose Estimation via Mamba and Rectified Flow",
            plan)
        self.assertIn(
            "| M0 | Linear Projection | Transformer | DETR Regression | No | Reproduced CVPR-style baseline |",
            plan)
        self.assertIn(
            "| M5 | Spectral Tokenizer | WiMamba (4 layers) | Draft + Rectified Flow | Yes | Full model with BoneLengthLoss |",
            plan)
        self.assertNotIn("| A1 |", plan)

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
        self.assertIn("x_linear = self.head(x)", source)
        self.assertIn("x_grid = x_linear.view(B * self.num_spatial, self.seq_len, self.embed_dims)", source)
        self.assertIn("torch.fft.rfft(", source)
        self.assertNotIn("torch.fft.rfft2", source)
        self.assertIn("x_token = self.norm(x_linear + x_enhanced)", source)
        self.assertIn("return x_token", source)
        self.assertIn('"""Doppler-Guided Wi-Fi Input Adapter.', source)
        self.assertIn("self.freq_gate = nn.Sequential(", source)
        self.assertIn("self.channel_proj = nn.Linear(embed_dims, embed_dims)", source)
        self.assertNotIn("xavier_uniform_(self.linear_proj.weight)", source)

    def test_wifi_input_adapter_has_spatial_mixing_before_temporal_fft(self):
        source = _read("opera/models/utils/spectral_tokenizer.py")

        self.assertIn("self._init_weights()", source)
        self.assertIn("def _init_weights(self):", source)
        self.assertIn("groups=embed_dims", source)
        self.assertIn("fft_bins = self.seq_len // 2 + 1", source)
        self.assertIn("self.freq_gate = nn.Sequential(", source)
        self.assertIn("nn.Linear(fft_bins, fft_bins * 2)", source)
        self.assertIn("nn.Linear(fft_bins * 2, self.seq_len)", source)
        self.assertIn("self.channel_proj = nn.Linear(embed_dims, embed_dims)", source)
        self.assertIn("nn.init.xavier_uniform_(self.freq_gate[0].weight)", source)
        self.assertIn("nn.init.constant_(self.freq_gate[0].bias, 0)", source)
        self.assertIn("nn.init.xavier_uniform_(self.freq_gate[2].weight)", source)
        self.assertIn("nn.init.constant_(self.freq_gate[2].bias, 0)", source)
        self.assertIn("nn.init.constant_(self.channel_proj.weight, 0.0)", source)
        self.assertIn("nn.init.constant_(self.channel_proj.bias, 0.0)", source)
        self.assertIn("x_time = x_grid.permute(0, 2, 1).contiguous()", source)
        self.assertIn("x_fft = torch.fft.rfft(x_grid, dim=1, norm='ortho')", source)
        self.assertIn("doppler_profile = x_fft_mag.mean(dim=-1)", source)
        self.assertIn("time_gate = self.freq_gate(doppler_profile)", source)
        self.assertIn("time_gate = torch.sigmoid(time_gate)", source)
        self.assertIn("time_gate_broadcast = time_gate.unsqueeze(1)", source)
        self.assertIn("x_enhanced = x_time * time_gate", source)
        self.assertIn("x_enhanced = self.channel_proj(x_enhanced)", source)
        self.assertNotIn("self.spatial_mixer", source)
        self.assertNotIn("self.complex_weight", source)

        idx_temporal = source.find("x_time = x_grid.permute(0, 2, 1).contiguous()")
        idx_fft = source.find("x_fft = torch.fft.rfft(")
        idx_gate = source.find("time_gate = self.freq_gate(")
        idx_fft = source.find("torch.fft.rfft(")

        self.assertTrue(idx_temporal < idx_fft)
        self.assertTrue(idx_fft < idx_gate)

    def test_wimamba_encoder_is_registered_in_mmcv_transformer_sequence_registry(self):
        source = _read("opera/models/backbones/wimamba.py")

        self.assertIn("mmcv.cnn.bricks.transformer", source)
        self.assertIn(
            "@MMCV_TRANSFORMER_LAYER_SEQUENCE.register_module()",
            source)
        self.assertIn("@TRANSFORMER_LAYER_SEQUENCE.register_module()", source)

    def test_wimamba_encoder_uses_factorized_spatiotemporal_blocks(self):
        source = _read("opera/models/backbones/wimamba.py")

        self.assertIn("class FactorizedWiMambaBlock", source)
        self.assertIn("self.norm_t = nn.LayerNorm(dim)", source)
        self.assertIn("self.mamba_t = Mamba(", source)
        self.assertIn("self.norm_s = nn.LayerNorm(dim)", source)
        self.assertIn("self.mamba_s_fwd = Mamba(", source)
        self.assertIn("self.mamba_s_bwd = Mamba(", source)
        self.assertIn("x_t = x_t.reshape(B * S, T, C)", source)
        self.assertIn("x_s = x_s.transpose(1, 2).contiguous().view(B * T, S, C)", source)
        self.assertIn("x_s_rev = torch.flip(x_s, dims=[1]).contiguous()", source)
        self.assertIn("out_bwd = torch.flip(out_bwd, dims=[1]).contiguous()", source)
        self.assertIn("out_s = out_fwd + out_bwd", source)
        self.assertIn("x = residual_t + x_t", source)
        self.assertIn("x = residual_s + out_s", source)
        self.assertIn("x = x.reshape(B, self.num_spatial, self.seq_len, C)", source)
        self.assertIn("x = x.permute(1, 0, 2).contiguous()", source)
        self.assertNotIn("class FastFactorizedWiMambaBlock", source)

    def test_wimamba2_csi_encoder_defines_dropin_and_cross_scan_variants(self):
        source = _read("opera/models/backbones/wimamba2_csi.py")

        self.assertIn("from mamba_ssm import Mamba2", source)
        self.assertNotIn("bimamba_type", source)
        self.assertIn("class FactorizedWiMamba2Block", source)
        self.assertIn("class WiMamba2DropInEncoder", source)
        self.assertIn("class CSISeparablePositionEmbedding", source)
        self.assertIn("class WiMamba2CSIEncoder", source)
        self.assertIn("@MMCV_TRANSFORMER_LAYER_SEQUENCE.register_module()", source)
        self.assertIn("@TRANSFORMER_LAYER_SEQUENCE.register_module()", source)
        self.assertIn("antenna_major", source)
        self.assertIn("time_major", source)
        self.assertIn("serpentine", source)
        self.assertIn("final_attn", source)

    def test_wimamba2_csi_encoder_is_exported(self):
        source = _read("opera/models/backbones/__init__.py")

        self.assertIn("WiMamba2DropInEncoder", source)
        self.assertIn("WiMamba2CSIEncoder", source)
        self.assertIn("CSISeparablePositionEmbedding", source)

    def test_mamba2_ablation_configs_follow_expected_ladder(self):
        dropin = _load_config_module("configs/wifi/petr_wifi_mamba2_dropin.py")
        flattened = _load_config_module("configs/wifi/petr_wifi_mamba2_flattened.py")
        crossscan = _load_config_module("configs/wifi/petr_wifi_mamba2_crossscan.py")
        pos = _load_config_module("configs/wifi/petr_wifi_mamba2_crossscan_pos.py")
        attn = _load_config_module("configs/wifi/petr_wifi_mamba2_crossscan_pos_attn.py")

        self.assertEqual(
            dropin.model["bbox_head"]["transformer"]["encoder"]["type"],
            "WiMamba2DropInEncoder")
        self.assertEqual(
            flattened.model["bbox_head"]["transformer"]["encoder"]["type"],
            "WiMamba2CSIEncoder")
        self.assertEqual(
            flattened.model["bbox_head"]["transformer"]["encoder"]["routes"],
            ("time_major",))
        self.assertEqual(
            crossscan.model["bbox_head"]["transformer"]["encoder"]["routes"],
            ("time_major", "serpentine"))
        self.assertFalse(crossscan.model["bbox_head"]["transformer"]["encoder"]["use_pos_embed"])
        self.assertTrue(pos.model["bbox_head"]["transformer"]["encoder"]["use_pos_embed"])
        self.assertTrue(attn.model["bbox_head"]["transformer"]["encoder"]["final_attn"])

    def test_wimamba_backbone_ablation_variants_have_unique_registry_names(self):
        init_source = _read("opera/models/backbones/__init__.py")

        expected_exports = (
            "WiMambaEncoder",
            "WiMambaV2Encoder",
            "WiMambaV3Encoder",
            "WiMamba1FlatEncoder",
            "WiMamba2FlatEncoder",
        )
        for name in expected_exports:
            self.assertIn(name, init_source)

        variant_sources = {
            "opera/models/backbones/wimamba.py": "class WiMambaEncoder",
            "opera/models/backbones/wimamba_v2.py": "class WiMambaV2Encoder",
            "opera/models/backbones/wimamba_v3.py": "class WiMambaV3Encoder",
            "opera/models/backbones/wimamba1_v1_flatten.py": "class WiMamba1FlatEncoder",
            "opera/models/backbones/wimamba2_flatten.py": "class WiMamba2FlatEncoder",
        }
        for path, class_decl in variant_sources.items():
            source = _read(path)
            self.assertIn(class_decl, source, msg=path)

    def test_wimamba_backbone_ablation_configs_switch_encoder_types(self):
        expected = {
            "configs/wifi/petr_wifi_mamba.py": "WiMambaEncoder",
            "configs/wifi/petr_wifi_mamba_v2.py": "WiMambaV2Encoder",
            "configs/wifi/petr_wifi_mamba_v3.py": "WiMambaV3Encoder",
            "configs/wifi/petr_wifi_mamba1_flatten.py": "WiMamba1FlatEncoder",
            "configs/wifi/petr_wifi_mamba2_flatten.py": "WiMamba2FlatEncoder",
        }
        for path, encoder_type in expected.items():
            cfg = _load_config_module(path)
            encoder = cfg.model["bbox_head"]["transformer"]["encoder"]
            self.assertEqual(encoder["type"], encoder_type, msg=path)
            self.assertEqual(cfg.model["backbone"]["type"], "WifiInputAdapter")
            self.assertEqual(cfg.model["backbone"]["mode"], "spectral")

    def test_wimamba_backbone_ablation_launcher_maps_all_variants(self):
        source = _read("scripts/run_wimamba_backbone_ablation.sh")

        for run_id in (
            "M1FCT", "M1V2", "M1V3", "M1FLAT", "M2FLAT",
            "M2D", "M2CSI", "M2C", "M2CP", "M2CPA",
        ):
            self.assertIn(run_id, source)
        for config_path in (
            "configs/wifi/petr_wifi_mamba.py",
            "configs/wifi/petr_wifi_mamba_v2.py",
            "configs/wifi/petr_wifi_mamba_v3.py",
            "configs/wifi/petr_wifi_mamba1_flatten.py",
            "configs/wifi/petr_wifi_mamba2_flatten.py",
            "configs/wifi/petr_wifi_mamba2_dropin.py",
            "configs/wifi/petr_wifi_mamba2_flattened.py",
            "configs/wifi/petr_wifi_mamba2_crossscan.py",
            "configs/wifi/petr_wifi_mamba2_crossscan_pos.py",
            "configs/wifi/petr_wifi_mamba2_crossscan_pos_attn.py",
        ):
            self.assertIn(config_path, source)
        self.assertIn("PYTHONHASHSEED_VALUE", source)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG_VALUE", source)
        self.assertIn("--seed \"$SEED\"", source)
        self.assertIn("--deterministic", source)
        self.assertIn("--auto-resume", source)
        self.assertIn("DRY_RUN", source)

    def test_benchmark_supports_segmented_profile_sections(self):
        source = _read("tools/analysis/benchmark.py")

        self.assertIn("--profile-sections", source)
        self.assertIn("section_latency_ms", source)
        self.assertIn("register_forward_pre_hook", source)
        self.assertIn("register_forward_hook", source)
        self.assertIn("torch.inference_mode()", source)

    def test_wifi_pose_source_adds_required_meta_fields(self):
        source = _read("opera/datasets/wifi_pose.py")

        self.assertIn("img_shape=img_shape", source)
        self.assertIn("ori_shape=img_shape", source)
        self.assertIn("pad_shape=img_shape", source)

    def test_dataset_worker_init_fn_seeds_torch_rng(self):
        source = _read("opera/datasets/builder.py")

        self.assertIn("def worker_init_fn", source)
        self.assertIn("torch.manual_seed(worker_seed)", source)

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
        self.assertIn("num_layers=4", source)

    def test_witidar_head_source_removes_limb_loss_logic(self):
        source = _read("opera/models/dense_heads/wi_tidar_head.py")

        self.assertIn("losses_cls, losses_kpt, losses_bone, losses_flow", source)
        self.assertNotIn("loss_limb", source)
        self.assertNotIn("losses_limb", source)
        self.assertNotIn("LimbLoss", source)

    def test_witidar_head_supports_transformer_or_mamba_encoder_selection(self):
        source = _read("opera/models/dense_heads/wi_tidar_head.py")

        self.assertIn("transformer_encoder=None", source)
        self.assertIn("Specify only one of transformer_encoder or mamba_cfg", source)
        self.assertIn("build_transformer_layer_sequence(transformer_encoder)", source)
        self.assertIn("self.encoder_type = 'transformer'", source)
        self.assertIn("if self.encoder_type == 'mamba':", source)
        self.assertIn("elif self.encoder_type == 'transformer':", source)
        self.assertIn("self.encoder(query=feat, key=None, value=None)", source)

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

    def test_parallel_5gpu_launcher_has_gpu_discovery_and_resume_logic(self):
        source = _read("scripts/run_train_5gpu.sh")

        self.assertIn("#!/usr/bin/env bash", source)
        self.assertIn("set -euo pipefail", source)
        self.assertIn("RUN_IDS=(M0 M1 M2 M4 M5)", source)
        self.assertIn("--query-gpu=index,name,memory.total,memory.used,utilization.gpu", source)
        self.assertIn("Need at least 5 visible GPUs", source)
        self.assertIn('Selected GPU IDs: ${RUN_GPU_IDS[*]}', source)
        self.assertIn('echo "$run_id -> GPU $gpu_id"', source)
        self.assertIn('CUDA_VISIBLE_DEVICES=$gpu_id', source)
        self.assertIn("PYTHONHASHSEED=42", source)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", source)
        self.assertIn("--auto-resume", source)
        self.assertIn("if [[ -f \"$work_dir/latest.pth\" ]]", source)
        self.assertIn("DRY_RUN", source)
        self.assertIn("trap terminate_children INT TERM", source)
        self.assertIn("Final Run Summary", source)
        self.assertIn("printf '%q '", source)
        self.assertIn('eval "$cmd" >"$log_path" 2>&1 &', source)

    def test_parallel_5gpu_launcher_supports_optional_m5_linear_run(self):
        source = _read("scripts/run_train_5gpu.sh")

        self.assertIn('[M5_linear]="configs/wifi/wi_tidir_wifi_linear.py"', source)
        self.assertIn('[M5_linear]=""', source)
        self.assertIn("EXTRA_RUN_IDS", source)

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

    def test_set_random_seed_enables_strict_cuda_determinism_controls(self):
        source = _read("opera/apis/train.py")

        self.assertIn("torch.backends.cuda.matmul.allow_tf32 = False", source)
        self.assertIn("torch.backends.cudnn.allow_tf32 = False", source)
        self.assertIn("torch.use_deterministic_algorithms(True, warn_only=True)", source)
        self.assertIn("os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'", source)

    def test_train_model_defaults_checkpoint_hook_to_latest_checkpoint_hook(self):
        source = _read("opera/apis/train.py")

        self.assertIn("checkpoint_config.setdefault('type', 'LatestCheckpointHook')", source)

    def test_docs_include_strict_repro_run_workflow(self):
        checklist = _read("docs/paper/2026-03-31-experiment-daily-checklist.md")

        self.assertIn("M1_rerun_strict_01", checklist)
        self.assertIn("data.workers_per_gpu=0", checklist)
        self.assertIn("epoch_5.pth", checklist)
        self.assertIn("latest.pth", checklist)


if __name__ == "__main__":
    unittest.main()

