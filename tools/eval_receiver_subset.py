# Copyright (c) Hikvision Research Institute. All rights reserved.
"""Receiver-subset evaluation helper.

Zero-masks out unused WiFi receivers before they reach the model and then runs
the standard single-GPU eval pipeline. When ``--rx-keep 0 1 2`` is passed the
result is numerically identical to the unmasked baseline, so the script doubles
as a sanity check.

The model-ready CSI tensor is ``(B, Tx=3, Rx=3, T=20, F=60)``. It survives the
pipeline all the way to ``PETR.extract_feat`` which does
``img.reshape(bs, -1, channel)`` and only there collapses the Tx/Rx/time cells
into 180 tokens. We monkey-patch ``extract_feat`` from outside so the mask is
applied on a fresh tensor while the original tensor stays untouched.
"""
import argparse
import json
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.utils import (build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

from opera.apis import single_gpu_test
from opera.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from opera.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a Wi-Fi pose model with a subset of receivers '
        '(unused Rx slots are zero-masked at the model input).')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--rx-keep',
        type=int,
        nargs='+',
        required=True,
        help='Receiver indices to keep (0..Rx-1). Other Rx slots are zeroed. '
        'E.g. "--rx-keep 0 1 2" keeps all three (sanity-check pass).')
    parser.add_argument(
        '--out',
        required=True,
        help='Path to write the metrics JSON.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use (single-GPU mode only).')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mpjpe'],
        help='Evaluation metric(s) passed to dataset.evaluate(). '
        'Defaults to ["mpjpe"].')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def _build_rx_mask(num_rx, rx_keep, device, dtype):
    keep_set = set(int(i) for i in rx_keep)
    invalid = [i for i in keep_set if i < 0 or i >= num_rx]
    if invalid:
        raise ValueError(
            f'--rx-keep contains out-of-range indices {invalid} '
            f'for Rx dim of size {num_rx}.')
    mask = torch.zeros(num_rx, device=device, dtype=dtype)
    for i in keep_set:
        mask[i] = 1
    # Broadcastable to (B, Tx, Rx, T, F).
    return mask.view(1, 1, num_rx, 1, 1)


def _wrap_extract_feat(detector, rx_keep):
    """Replace detector.extract_feat with a version that zero-masks unused Rx.

    Works on a fresh tensor (out-of-place) so the cached batch on the
    data loader side is never mutated. Gradient-free path is preserved
    because we are already inside ``torch.no_grad()`` during eval.
    """
    orig_extract = detector.extract_feat
    rx_keep_tuple = tuple(int(i) for i in rx_keep)

    def patched_extract_feat(img):
        # img shape: (B, Tx, Rx, T, F)
        if img.dim() != 5:
            raise RuntimeError(
                f'Expected 5-D CSI tensor (B, Tx, Rx, T, F); got shape '
                f'{tuple(img.shape)}. Cannot apply receiver-subset mask.')
        num_rx = img.shape[2]
        mask = _build_rx_mask(num_rx, rx_keep_tuple, img.device, img.dtype)
        # Out-of-place: leave the original tensor untouched.
        img_masked = img * mask
        return orig_extract(img_masked)

    detector.extract_feat = patched_extract_feat
    return orig_extract


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    # Force batch_size=1 regardless of what the config requests.
    test_loader_cfg['samples_per_gpu'] = 1

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # Install the Rx-mask wrapper on the underlying detector. Must happen
    # AFTER build_dp because build_dp wraps the detector in DataParallel.
    detector = model.module
    if not hasattr(detector, 'extract_feat'):
        raise RuntimeError(
            'Underlying detector has no extract_feat method; receiver-subset '
            'masking via this entry point is not supported for this model.')
    _wrap_extract_feat(detector, args.rx_keep)

    rx_keep_sorted = sorted(set(int(i) for i in args.rx_keep))
    print(f'\n[eval_receiver_subset] rx_keep={rx_keep_sorted} '
          f'(unused Rx slots zero-masked at PETR.extract_feat input)')

    outputs = single_gpu_test(model, data_loader, False, None, 0.3)

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval))

    metrics_out_abs = osp.abspath(args.out)
    mmcv.mkdir_or_exist(osp.dirname(metrics_out_abs))
    eval_kwargs['metrics_out'] = metrics_out_abs

    metric = dataset.evaluate(outputs, **eval_kwargs)

    # dataset.evaluate already dumps the structured JSON via metrics_out.
    # Read it back to print a tidy one-line summary.
    if not osp.isfile(metrics_out_abs):
        # Fallback: write whatever evaluate returned (e.g. older dataset).
        with open(metrics_out_abs, 'w') as f:
            json.dump({'rx_keep': rx_keep_sorted, 'metric': dict(metric)},
                      f, indent=2, default=str)
        per_joint_count = 0
    else:
        # Augment the JSON in place with the rx_keep tag for traceability.
        try:
            with open(metrics_out_abs, 'r') as f:
                exported = json.load(f)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f'Could not re-open metrics JSON for tagging: {exc}')
            exported = None
        if isinstance(exported, dict):
            exported['rx_keep'] = rx_keep_sorted
            with open(metrics_out_abs, 'w') as f:
                json.dump(exported, f, indent=2)
            per_joint_count = len(exported.get('per_joint_mpjdle', {}))
        else:
            per_joint_count = 0

    mpjpe_val = metric.get('mpjpe', float('nan')) if metric else float('nan')
    print(f'\n[eval_receiver_subset] DONE rx_keep={rx_keep_sorted} '
          f'mpjpe={mpjpe_val:.4f} mm '
          f'per_joint_mpjdle_entries={per_joint_count} '
          f'json={metrics_out_abs}')


if __name__ == '__main__':
    main()
