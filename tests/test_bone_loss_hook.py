import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / 'opera/core/runner/hooks/bone_warmup_hook.py'


class _DummyRegistry:
    def register_module(self):
        def decorator(cls):
            return cls
        return decorator


def _load_hook_class():
    mmcv_mod = types.ModuleType('mmcv')
    runner_mod = types.ModuleType('mmcv.runner')
    hooks_mod = types.ModuleType('mmcv.runner.hooks')
    hooks_mod.HOOKS = _DummyRegistry()
    hooks_mod.Hook = object

    spec = importlib.util.spec_from_file_location('bone_warmup_hook_test', HOOK_PATH)
    module = importlib.util.module_from_spec(spec)

    old_modules = {}
    for name, mod in (
        ('mmcv', mmcv_mod),
        ('mmcv.runner', runner_mod),
        ('mmcv.runner.hooks', hooks_mod),
    ):
        old_modules[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

    return module.BoneLossWarmupHook


class BoneLossWarmupHookTests(unittest.TestCase):
    def test_hook_bypasses_safely_when_no_bone_loss(self):
        BoneLossWarmupHook = _load_hook_class()
        hook = BoneLossWarmupHook()

        runner = MagicMock()
        runner.max_epochs = 10
        runner.epoch = 1
        runner.rank = 0
        runner.model.module.bbox_head.loss_bone = None

        try:
            hook.before_train_epoch(runner)
        except Exception as exc:  # pragma: no cover
            self.fail(f'Hook should bypass safely without loss_bone, got: {exc}')

    def test_ratio_based_warmup_calculation(self):
        BoneLossWarmupHook = _load_hook_class()
        hook = BoneLossWarmupHook(
            target_weight=2.0, warmup_ratio=0.1, ramp_ratio=0.2)

        runner = MagicMock()
        runner.max_epochs = 50
        runner.rank = 0
        mock_loss_bone = MagicMock()
        runner.model.module.bbox_head.loss_bone = mock_loss_bone

        for epoch in range(5):
            runner.epoch = epoch
            hook.before_train_epoch(runner)
            self.assertEqual(mock_loss_bone.loss_weight, 0.0)

        for epoch in range(5, 15):
            runner.epoch = epoch
            hook.before_train_epoch(runner)
            step = epoch - 5 + 1
            expected_weight = 2.0 * (step / 10.0)
            self.assertAlmostEqual(mock_loss_bone.loss_weight, expected_weight)

        runner.epoch = 15
        hook.before_train_epoch(runner)
        self.assertEqual(mock_loss_bone.loss_weight, 2.0)

    def test_hook_can_read_non_wrapped_model(self):
        BoneLossWarmupHook = _load_hook_class()
        hook = BoneLossWarmupHook(target_weight=1.0, warmup_ratio=0.1, ramp_ratio=0.1)

        runner = MagicMock()
        runner.max_epochs = 20
        runner.epoch = 2
        runner.rank = 0
        del runner.model.module
        runner.model.bbox_head.loss_bone = MagicMock()

        hook.before_train_epoch(runner)
        self.assertEqual(runner.model.bbox_head.loss_bone.loss_weight, 0.5)


if __name__ == '__main__':
    unittest.main()
