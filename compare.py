import torch
import torch.nn as nn
import time
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from opera.models import build_model
import prettytable as pt

# Import custom layers
try:
    import custom_layers
    print(">>> Đã đăng ký các layer tùy chỉnh thành công.")
except ImportError:
    print(">>> Cảnh báo: Không tìm thấy custom_layers.py.")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6

def measure_inference_speed(model, input_shape, iterations=200, warmup=50):
    model.eval()
    # Chuyển mô hình sang GPU
    model.cuda()
    
    # Tạo dummy input đúng shape: (Batch, Antenna_Rx, Antenna_Tx, Time_Packets, Subcarriers)
    # Với dữ liệu của bạn, Subcarriers (chiều cuối) phải là 60 (amp + phase) hoặc 30 (chỉ amp)
    dummy_input = torch.randn(*input_shape).cuda()
    
    # Meta data giả lập cho MMDetection
    img_metas = [{
        'img_shape': (256, 256, 3), 
        'scale_factor': np.array([1., 1., 1., 1.]),
        'batch_input_shape': (256, 256)
    }]

    print(f"--- Đang Warm-up {warmup} vòng với input shape {input_shape}... ---")
    with torch.no_grad():
        for _ in range(warmup):
            # Quy trình forward chính xác của PETR Detector
            bs, _, _, _, channel = dummy_input.shape
            x = dummy_input.reshape(bs, -1, channel)
            feat = model.head(x)
            _ = model.bbox_head.simple_test(feat, img_metas, rescale=False)

    print(f"--- Đang đo FPS thực tế trên {iterations} vòng... ---")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for i in range(iterations):
            starter.record()
            
            # Thực thi logic inference
            bs, _, _, _, channel = dummy_input.shape
            x = dummy_input.reshape(bs, -1, channel)
            feat = model.head(x)
            _ = model.bbox_head.simple_test(feat, img_metas, rescale=False)
            
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    avg_time = np.mean(timings)
    fps = 1000.0 / avg_time
    return avg_time, fps

def run_eval(name, config_path, checkpoint_path=None):
    print(f"\n{'='*20} ĐANG ĐÁNH GIÁ: {name} {'='*20}")
    cfg = Config.fromfile(config_path)
    
    # Khởi tạo mô hình
    model = build_model(cfg.model)
    
    # Load checkpoint
    if checkpoint_path:
        try:
            load_checkpoint(model, checkpoint_path, map_location='cpu')
            print(f"-> Đã load thành công checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"-> Không load được checkpoint ({e}), đo trên weights mặc định.")
    
    # Tự động xác định input dimension từ model head
    # model.head.in_features sẽ cho biết là 60 hay 30
    in_features = model.head.in_features
    # Dựa trên dataset: (Batch, Rx, Tx, Time, Subcarriers)
    # Subcarriers = in_features
    input_shape = (1, 3, 3, 20, in_features) 
    
    total_p, train_p = count_parameters(model)
    latency, fps = measure_inference_speed(model, input_shape)
    
    return {
        "Name": name,
        "Params (M)": f"{total_p:.2f}",
        "Trainable (M)": f"{train_p:.2f}",
        "Latency (ms)": f"{latency:.2f}",
        "FPS": f"{fps:.2f}"
    }

if __name__ == "__main__":
    experiments = [
        {
            "name": "Baseline (Standard PETR)",
            "config": "/root/petr_wifi.py", # File config gốc
            "ckpt": "/root/epoch_10.pth"
        },
        {
            "name": "Upgraded (FNet + Graph GCN)",
            "config": "/root/Person-in-WiFi-3D/work_dirs/petr_wifi_fnet_graph/petr_wifi.py", # File chứa WiFiFNetLayer & WiFiGraphLayer
            "ckpt": "/root/Person-in-WiFi-3D/epoch_10.pth"
        }
    ]

    table = pt.PrettyTable()
    table.field_names = ["Model Variant", "Total Params", "Trainable", "Latency (ms)", "FPS"]

    for exp in experiments:
        try:
            res = run_eval(exp['name'], exp['config'], exp['ckpt'])
            table.add_row([res["Name"], res["Params (M)"], res["Trainable (M)"], res["Latency (ms)"], res["FPS"]])
        except Exception as e:
            print(f"!!! Lỗi khi eval {exp['name']}: {e}")

    print("\n" + "="*30 + " KẾT QUẢ SO SÁNH HIỆU NĂNG " + "="*30)
    if torch.cuda.is_available():
        print(f"Thiết bị: {torch.cuda.get_device_name(0)}")
    print(table)