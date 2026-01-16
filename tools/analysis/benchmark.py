import argparse
import torch
import time
import numpy as np
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.runner import load_checkpoint

# Import dự án
from opera.models import build_model
from opera.datasets import build_dataset

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
except ImportError:
    print("Warning: fvcore not installed. FLOPs calculation might be skipped.")
    FlopCountAnalysis = None

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark WiTiDAR Model')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--checkpoint', help='path to checkpoint (optional)', default=None)
    parser.add_argument('--shape', type=int, nargs='+', default=[1, 3, 3, 20, 2], help='input size')
    parser.add_argument('--device', default='cuda:0', help='device used for benchmark')
    parser.add_argument('--times', type=int, default=100, help='number of times to measure speed')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Config
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 2. Build Model
    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    # 3. Create Dummy Input
    # Input chuẩn của PETR WiFi thường là 5 chiều: (Batch, Rx, Tx, Subcarrier, Time/Feature)
    # Dựa vào SpectralTokenizer(in_channels=60), ta giả lập input sao cho channel cuối = 60
    # Hoặc tổng reshape lại phù hợp.
    # Ở đây ta tạo input giả lập (Batch, 1, 1, 180, 60) để an toàn nhất với code reshape
    dummy_input = torch.randn(1, 1, 1, 180, 60).to(args.device)
    
    # Dummy img_metas (Cần thiết cho forward của Detector)
    dummy_metas = [[{
        'img_shape': (1, 1, 1),
        'scale_factor': 1.0,
        'batch_input_shape': (1, 1)
    }]]

    # ==========================================
    # 1. ĐO SỐ LƯỢNG THAM SỐ (PARAMS)
    # ==========================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "="*40)
    print(f"MODEL STATISTICS")
    print("="*40)
    print(f"Total Parameters:     {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # ==========================================
    # 2. ĐO FLOPs (Tính toán)
    # ==========================================
    print("-" * 40)
    if FlopCountAnalysis:
        try:
            # Wrapper để fvcore hiểu input
            class Wrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    # Gọi simple_test hoặc forward_dummy
                    return self.model.simple_test(x, dummy_metas, rescale=False)

            wrapper = Wrapper(model)
            flops = FlopCountAnalysis(wrapper, dummy_input)
            print(f"FLOPs (G):            {flops.total() / 1e9:.2f} GFLOPs")
        except Exception as e:
            print(f"Lỗi khi tính FLOPs với fvcore: {e}")
            print("Thử phương pháp tính thủ công (ước lượng)...")
    else:
        print("Cài đặt 'fvcore' để tính FLOPs chính xác (pip install fvcore)")

    # ==========================================
    # 3. ĐO TỐC ĐỘ (FPS & LATENCY)
    # ==========================================
    print("-" * 40)
    print(f"Benchmarking inference speed ({args.times} runs)...")
    
    # Warm up (chạy nháp để GPU nóng máy)
    for _ in range(10):
        with torch.no_grad():
            model.simple_test(dummy_input, dummy_metas, rescale=False)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(args.times):
        with torch.no_grad():
            model.simple_test(dummy_input, dummy_metas, rescale=False)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / args.times
    fps = 1.0 / avg_time
    
    print(f"Latency (Độ trễ):     {avg_time * 1000:.2f} ms / sample")
    print(f"FPS (Tốc độ khung hình): {fps:.2f} FPS")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()


# python tools/analysis/benchmark.py configs/wifi/wi_tidir_wifi.py