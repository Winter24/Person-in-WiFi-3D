import math

from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class BoneLossWarmupHook(Hook):
    """Warm up BoneLengthLoss automatically when a model enables it.

    The hook is intentionally "structurally aware": it inspects the model at
    runtime and only applies when ``loss_bone`` exists and is not ``None``.
    This lets us register the hook globally in the base config without
    affecting non-bone ablations such as B0/B1/B2/B4.
    """

    def __init__(self,
                 target_weight=1.0,
                 warmup_ratio=0.1,
                 ramp_ratio=0.1):
        self.target_weight = target_weight
        self.warmup_ratio = warmup_ratio
        self.ramp_ratio = ramp_ratio
        self.warmup_epochs = None
        self.ramp_epochs = None
        self._calculated = False

    def _get_loss_bone(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'loss_bone'):
            return model.bbox_head.loss_bone
        if hasattr(model, 'loss_bone'):
            return model.loss_bone
        return None

    def _maybe_init_schedule(self, runner):
        if self._calculated:
            return

        self.warmup_epochs = max(
            1, math.floor(runner.max_epochs * self.warmup_ratio))
        self.ramp_epochs = max(
            1, math.floor(runner.max_epochs * self.ramp_ratio))
        self._calculated = True

    def before_train_epoch(self, runner):
        self._maybe_init_schedule(runner)

        loss_bone = self._get_loss_bone(runner)
        if loss_bone is None:
            return

        epoch = runner.epoch
        if epoch < self.warmup_epochs:
            current_weight = 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            step = epoch - self.warmup_epochs + 1
            progress = step / float(self.ramp_epochs)
            current_weight = self.target_weight * progress
        else:
            current_weight = self.target_weight

        loss_bone.loss_weight = current_weight

        rank = getattr(runner, 'rank', 0)
        if rank == 0:
            runner.logger.info(
                f'[BoneLossWarmup] Epoch {epoch + 1}/{runner.max_epochs} '
                f'| Warmup: {self.warmup_epochs}e, Ramp: {self.ramp_epochs}e '
                f'| Bone weight -> {current_weight:.4f}')
