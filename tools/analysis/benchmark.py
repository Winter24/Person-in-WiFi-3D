import argparse
import json
import os
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

# Import project
from opera.models import build_model

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark WiFi Pose Model')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--checkpoint', help='path to checkpoint (optional)',
                        default=None)
    parser.add_argument('--shape', type=int, nargs='+',
                        default=[1, 3, 3, 20, 60],
                        help='input tensor shape (default: 1 3 3 20 60 '
                             '= batch Rx Tx time features)')
    parser.add_argument('--device', default='cuda:0',
                        help='device used for benchmark (cuda:N or cpu)')
    parser.add_argument('--times', type=int, default=100,
                        help='number of forward passes to measure speed')
    parser.add_argument('--warmup', type=int, default=10,
                        help='number of warmup iterations before timing')
    parser.add_argument('--out', type=str, default=None,
                        help='path to write benchmark results as JSON '
                             '(e.g. paper_assets/logs/B0_benchmark.json)')
    return parser.parse_args()


def _is_cuda(device_str):
    """Return True when the target device is CUDA."""
    return device_str.startswith('cuda')


def _normalize_device(device_str):
    """Resolve the requested device to a safe, usable target.

    Rules:
      - If CUDA is requested but unavailable, fail clearly instead of
        silently benchmarking on CPU.
      - If cuda:N is requested but N is outside the visible-device range,
        fall back to cuda:0 when at least one CUDA device is visible.
      - Otherwise keep the requested device string unchanged.
    """
    if not _is_cuda(device_str):
        return device_str

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requested but torch.cuda.is_available() is False. "
            "This process cannot see a GPU. If CUDA_VISIBLE_DEVICES is used, "
            "pass a remapped ordinal such as '--device cuda:0'. "
            "Do not fall back to CPU for Mamba-based models.")

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError(
            "CUDA benchmark requested but no visible CUDA devices were found "
            "for this process.")

    if ':' not in device_str:
        return 'cuda:0'

    try:
        ordinal = int(device_str.split(':', 1)[1])
    except ValueError:
        print(f"WARNING: Unrecognized CUDA device string '{device_str}'. "
              "Falling back to 'cuda:0'.")
        return 'cuda:0'

    if ordinal >= device_count or ordinal < 0:
        print(f"WARNING: Invalid CUDA device ordinal '{device_str}' for the "
              f"current process (visible device count={device_count}). "
              "Falling back to 'cuda:0'.")
        return 'cuda:0'

    return device_str


def main():
    args = parse_args()

    device = _normalize_device(args.device)
    use_cuda = _is_cuda(device)

    # ------------------------------------------------------------------
    # 1. Load Config & Build Model
    # ------------------------------------------------------------------
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)

    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Create Dummy Input (uses --shape properly)
    # ------------------------------------------------------------------
    input_shape = tuple(args.shape)
    dummy_input = torch.randn(*input_shape).to(device)

    dummy_metas = [[{
        'img_shape': (1, 1, 1),
        'scale_factor': 1.0,
        'batch_input_shape': (1, 1),
    }]]

    # ------------------------------------------------------------------
    # 3. Parameter Count
    # ------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)

    params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6

    print("\n" + "=" * 50)
    print("MODEL STATISTICS")
    print("=" * 50)
    print(f"Total Parameters:     {params_m:.2f} M")
    print(f"Trainable Parameters: {trainable_params_m:.2f} M")

    # ------------------------------------------------------------------
    # 4. FLOPs
    # ------------------------------------------------------------------
    flops_g = None
    print("-" * 50)
    if FlopCountAnalysis is not None:
        try:
            class _Wrapper(torch.nn.Module):
                def __init__(self, m, metas):
                    super().__init__()
                    self.m = m
                    self.metas = metas

                def forward(self, x):
                    return self.m.simple_test(x, self.metas, rescale=False)

            wrapper = _Wrapper(model, dummy_metas)
            flops = FlopCountAnalysis(wrapper, dummy_input)
            flops_g = flops.total() / 1e9
            print(f"FLOPs:                {flops_g:.2f} GFLOPs")
        except Exception as e:
            print(f"FLOPs computation failed: {e}")
    else:
        print("Install 'fvcore' for FLOPs calculation (pip install fvcore)")

    # ------------------------------------------------------------------
    # 5. Latency & FPS
    # ------------------------------------------------------------------
    print("-" * 50)
    print(f"Benchmarking inference speed ({args.times} runs, "
          f"warmup={args.warmup}, device={device}) ...")

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            model.simple_test(dummy_input, dummy_metas, rescale=False)

    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    for _ in range(args.times):
        with torch.no_grad():
            model.simple_test(dummy_input, dummy_metas, rescale=False)
    if use_cuda:
        torch.cuda.synchronize(device)
    end_time = time.time()

    avg_time = (end_time - start_time) / args.times
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000.0

    print(f"Latency:              {latency_ms:.2f} ms / sample")
    print(f"FPS:                  {fps:.2f}")

    # ------------------------------------------------------------------
    # 6. Peak Memory (CUDA only)
    # ------------------------------------------------------------------
    peak_memory_allocated_mb = None
    peak_memory_reserved_mb = None

    if use_cuda:
        peak_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print("-" * 50)
        print(f"Peak Memory Allocated: {peak_memory_allocated_mb:.1f} MB")
        print(f"Peak Memory Reserved:  {peak_memory_reserved_mb:.1f} MB")
    else:
        print("-" * 50)
        print("Memory stats: N/A (CPU mode)")

    print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # 7. Export JSON
    # ------------------------------------------------------------------
    report = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        device=device,
        input_shape=list(input_shape),
        times=args.times,
        warmup=args.warmup,
        params_m=round(params_m, 4),
        trainable_params_m=round(trainable_params_m, 4),
        flops_g=round(flops_g, 4) if flops_g is not None else None,
        latency_ms=round(latency_ms, 4),
        fps=round(fps, 4),
        peak_memory_allocated_mb=(round(peak_memory_allocated_mb, 2)
                                  if peak_memory_allocated_mb is not None
                                  else None),
        peak_memory_reserved_mb=(round(peak_memory_reserved_mb, 2)
                                 if peak_memory_reserved_mb is not None
                                 else None),
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Benchmark results exported to: {args.out}")

    return report


if __name__ == '__main__':
    main()
