import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "opera/core/runner/hooks/latest_checkpoint_hook.py"


class _DummyRegistry:
    def register_module(self):
        def decorator(cls):
            return cls
        return decorator


class _DummyHook:
    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False


class _DummyCheckpointHook(_DummyHook):
    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 sync_buffer=False,
                 file_client_args=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args
        self.args = kwargs

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir
        self.file_client = _DummyFileClient()
        self.args.setdefault("create_symlink", self.file_client.allow_symlink)

    def _save_checkpoint(self, runner):
        filename_tmpl = self.args.get("filename_tmpl", "epoch_{}.pth")
        runner.save_checkpoint(
            self.out_dir,
            filename_tmpl=filename_tmpl,
            save_optimizer=self.save_optimizer,
            create_symlink=self.args.get("create_symlink", False))


class _DummyFileClient:
    allow_symlink = False
    name = "disk"

    @staticmethod
    def infer_client(file_client_args, out_dir):
        return _DummyFileClient()

    @staticmethod
    def join_path(*parts):
        return os.path.join(*parts)

    @staticmethod
    def isfile(path):
        return False

    @staticmethod
    def remove(path):
        return path


def _master_only(func):
    return func


def _load_hook_class():
    mmcv_mod = types.ModuleType("mmcv")
    fileio_mod = types.ModuleType("mmcv.fileio")
    fileio_mod.FileClient = _DummyFileClient
    runner_mod = types.ModuleType("mmcv.runner")
    hooks_mod = types.ModuleType("mmcv.runner.hooks")
    hooks_mod.HOOKS = _DummyRegistry()
    hooks_mod.Hook = _DummyHook
    hooks_mod.CheckpointHook = _DummyCheckpointHook
    dist_utils_mod = types.ModuleType("mmcv.runner.dist_utils")
    dist_utils_mod.allreduce_params = lambda params: None
    dist_utils_mod.master_only = _master_only

    spec = importlib.util.spec_from_file_location(
        "latest_checkpoint_hook_test", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)

    old_modules = {}
    for name, mod in (
        ("mmcv", mmcv_mod),
        ("mmcv.fileio", fileio_mod),
        ("mmcv.runner", runner_mod),
        ("mmcv.runner.hooks", hooks_mod),
        ("mmcv.runner.dist_utils", dist_utils_mod),
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

    return module.LatestCheckpointHook


class _DummyLogger:
    def info(self, msg):
        return msg


class _DummyRunner:
    def __init__(self, work_dir, max_epochs):
        self.work_dir = work_dir
        self.epoch = 0
        self.iter = 0
        self._max_epochs = max_epochs
        self.meta = {}
        self.logger = _DummyLogger()
        self.model = types.SimpleNamespace(buffers=lambda: [])
        self.saved_checkpoints = {}

    @property
    def max_epochs(self):
        return self._max_epochs

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl="epoch_{}.pth",
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        filename = filename_tmpl.format(self.epoch + 1)
        self.saved_checkpoints[filename] = self.epoch + 1
        if create_symlink:
            self.saved_checkpoints["latest.pth"] = self.epoch + 1


class LatestCheckpointHookTests(unittest.TestCase):
    def test_updates_latest_every_epoch_but_keeps_numbered_checkpoints_periodic(self):
        LatestCheckpointHook = _load_hook_class()
        hook = LatestCheckpointHook(interval=5, by_epoch=True)
        work_dir = str(REPO_ROOT / "tests" / "dummy_work_dir")
        runner = _DummyRunner(work_dir=work_dir, max_epochs=6)
        hook.before_run(runner)

        for epoch in range(6):
            runner.epoch = epoch
            hook.after_train_epoch(runner)

        self.assertEqual(runner.saved_checkpoints["latest.pth"], 6)
        self.assertEqual(runner.saved_checkpoints["epoch_5.pth"], 5)
        self.assertNotIn("epoch_1.pth", runner.saved_checkpoints)
        self.assertNotIn("epoch_2.pth", runner.saved_checkpoints)
        self.assertNotIn("epoch_3.pth", runner.saved_checkpoints)
        self.assertNotIn("epoch_4.pth", runner.saved_checkpoints)
        self.assertNotIn("epoch_6.pth", runner.saved_checkpoints)
        self.assertEqual(
            runner.meta["hook_msgs"]["last_ckpt"],
            str(Path(work_dir) / "latest.pth"))


if __name__ == "__main__":
    unittest.main()
