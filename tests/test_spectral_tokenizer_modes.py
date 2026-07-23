import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "opera" / "models" / "utils" / "spectral_tokenizer.py"


class _Registry:
    def register_module(self):
        return lambda cls: cls


def _load_tokenizer_module():
    module_names = (
        "mmdet",
        "mmdet.models",
        "mmdet.models.builder",
        "opera",
        "opera.models",
        "opera.models.builder",
        "opera.models.utils",
        "opera.models.utils.spectral_tokenizer",
    )
    previous = {name: sys.modules.get(name) for name in module_names}

    mmdet = types.ModuleType("mmdet")
    mmdet.__path__ = []
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.__path__ = []
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = _Registry()

    opera = types.ModuleType("opera")
    opera.__path__ = []
    opera_models = types.ModuleType("opera.models")
    opera_models.__path__ = []
    opera_builder = types.ModuleType("opera.models.builder")
    opera_builder.BACKBONES = _Registry()
    opera_utils = types.ModuleType("opera.models.utils")
    opera_utils.__path__ = []

    replacements = {
        "mmdet": mmdet,
        "mmdet.models": mmdet_models,
        "mmdet.models.builder": mmdet_builder,
        "opera": opera,
        "opera.models": opera_models,
        "opera.models.builder": opera_builder,
        "opera.models.utils": opera_utils,
    }
    sys.modules.update(replacements)

    try:
        spec = importlib.util.spec_from_file_location(
            "opera.models.utils.spectral_tokenizer", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class SpectralTokenizerModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_tokenizer_module()
        cls.adapter_cls = cls.module.WifiInputAdapter

    def test_all_diagnostic_modes_return_expected_shape(self):
        x = torch.randn(2, 180, 60)
        for mode in (
            "linear",
            "linear_ln",
            "temporal_residual",
            "spectral_gate_residual",
            "spectral",
        ):
            with self.subTest(mode=mode):
                adapter = self.adapter_cls(60, 256, mode=mode)
                output = adapter(x)
                self.assertEqual(tuple(output.shape), (2, 180, 256))

    def test_parameter_counts_lock_the_component_ladder(self):
        expected = {
            "linear": 15616,
            "linear_ln": 16128,
            "temporal_residual": 83456,
            "spectral_gate_residual": 82644,
            "spectral": 84180,
        }
        for mode, expected_count in expected.items():
            with self.subTest(mode=mode):
                adapter = self.adapter_cls(60, 256, mode=mode)
                count = sum(parameter.numel() for parameter in adapter.parameters())
                self.assertEqual(count, expected_count)

    def test_zero_initialized_residual_starts_at_layer_normalized_projection(self):
        x = torch.randn(2, 180, 60)
        for mode in (
            "temporal_residual",
            "spectral_gate_residual",
            "spectral",
        ):
            with self.subTest(mode=mode):
                adapter = self.adapter_cls(60, 256, mode=mode)
                output, debug = adapter(x, return_debug=True)
                projected = adapter.head(x)
                expected = adapter.norm(projected)
                self.assertTrue(torch.equal(debug["residual_correction"], torch.zeros_like(projected)))
                self.assertTrue(torch.allclose(output, expected, atol=1e-6, rtol=1e-6))

    def test_spectral_gate_is_independent_for_each_spatial_link(self):
        adapter = self.adapter_cls(60, 256, mode="spectral")
        _, debug = adapter(torch.randn(2, 180, 60), return_debug=True)
        self.assertEqual(tuple(debug["spectral_descriptor"].shape), (18, 11))
        self.assertEqual(tuple(debug["temporal_gate"].shape), (18, 20))

    def test_grouped_modes_reject_incompatible_token_length(self):
        adapter = self.adapter_cls(60, 256, mode="spectral")
        with self.assertRaisesRegex(ValueError, "num_spatial.*seq_len"):
            adapter(torch.randn(1, 179, 60))

    def test_existing_mode_state_dict_keys_remain_stable(self):
        linear = self.adapter_cls(60, 256, mode="linear")
        spectral = self.adapter_cls(60, 256, mode="spectral")
        self.assertEqual(set(linear.state_dict()), {"head.weight", "head.bias"})
        self.assertEqual(
            set(spectral.state_dict()),
            {
                "head.weight",
                "head.bias",
                "time_conv.weight",
                "time_conv.bias",
                "freq_gate.0.weight",
                "freq_gate.0.bias",
                "freq_gate.2.weight",
                "freq_gate.2.bias",
                "channel_proj.weight",
                "channel_proj.bias",
                "norm.weight",
                "norm.bias",
            },
        )


if __name__ == "__main__":
    unittest.main()
