from mmcv.runner.dist_utils import allreduce_params, master_only
from mmcv.runner.hooks import CheckpointHook, HOOKS


@HOOKS.register_module()
class LatestCheckpointHook(CheckpointHook):
    """Keep `latest.pth` fresh every epoch while saving numbered checkpoints
    only on the configured interval.

    This is useful for auto-resume workflows where we want a cheap rolling
    checkpoint every epoch but do not want to accumulate `epoch_N.pth` files
    at the same cadence.
    """

    def __init__(self, latest_filename_tmpl='latest.pth', **kwargs):
        super().__init__(**kwargs)
        self.latest_filename_tmpl = latest_filename_tmpl

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())

        if self.every_n_epochs(runner, self.interval):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            self._save_checkpoint(runner)
            if not self.args.get('create_symlink', False):
                self._save_latest_checkpoint(runner)
            else:
                self._record_latest_checkpoint(runner)
        else:
            runner.logger.info(
                f'Updating latest checkpoint at {runner.epoch + 1} epochs')
            self._save_latest_checkpoint(runner)

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())

        if self.every_n_iters(runner, self.interval):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            self._save_checkpoint(runner)
            if not self.args.get('create_symlink', False):
                self._save_latest_checkpoint(runner)
            else:
                self._record_latest_checkpoint(runner)
        else:
            runner.logger.info(
                f'Updating latest checkpoint at {runner.iter + 1} iterations')
            self._save_latest_checkpoint(runner)

    @master_only
    def _save_latest_checkpoint(self, runner):
        args = self.args.copy()
        args.pop('filename_tmpl', None)
        args['create_symlink'] = False
        runner.save_checkpoint(
            self.out_dir,
            filename_tmpl=self.latest_filename_tmpl,
            save_optimizer=self.save_optimizer,
            **args)
        self._record_latest_checkpoint(runner)

    def _record_latest_checkpoint(self, runner):
        if runner.meta is None:
            runner.meta = dict()
        runner.meta.setdefault('hook_msgs', dict())
        runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
            self.out_dir, self.latest_filename_tmpl)
