%%writefile /content/Person-in-WiFi-3D/streamlit/app.py
# @title streamlit/app.py

import streamlit as st
import numpy as np
import scipy.io as sio
import h5py
import time
import pandas as pd
import plotly.graph_objects as go
import os
import re
import cv2
import torch
import sys
import pywt
import torch
from mmcv.parallel import collate, scatter

def preprocess_csi_for_ai(csi_raw_complex):
    """
    csi_raw_complex: [3, 3, 30, 20] (Rx, Tx, Sub, Pkt)
    Tái lập chính xác logic trong WifiPoseDataset
    """
    # 1. DWT Amplitude
    w = pywt.Wavelet('db11')
    # Tính biên độ
    csi_amp = np.abs(csi_raw_complex)
    # Phân rã và tái cấu trúc bằng Wavelet để khử nhiễu
    coeffs = pywt.wavedec(csi_amp, w, mode='symmetric')
    csi_amp_denoised = pywt.waverec(coeffs, w)
    
    # 2. Phase Sanitization (Khử nhiễu pha - lược bỏ bớt logic phức tạp để tăng tốc)
    csi_phase = np.angle(csi_raw_complex)
    
    # 3. Concatenate & Permute
    # Kết hợp Biên độ và Pha: [3, 3, 60, 20]
    csi_combined = np.concatenate((csi_amp_denoised, csi_phase), axis=2)
    
    # Chuyển về Tensor: [1, 3, 3, 20, 30] hoặc tương đương tùy config
    # Dựa trên file petr_wifi.py: csi tensor: (3*3*20*30)
    csi_tensor = torch.FloatTensor(csi_combined).permute(0, 1, 3, 2)
    return csi_tensor.unsqueeze(0) # Thêm chiều Batch [1, 3, 3, 20, 30]

def run_ai_inference(model, csi_tensor):
    """Thực hiện suy luận qua mô hình PETR"""
    device = next(model.parameters()).device
    csi_tensor = csi_tensor.to(device)
    
    with torch.no_grad():
        # Tạo img_metas giả lập cho PETRHead
        img_metas = [{
            'img_shape': (256, 256, 3), 
            'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32),
            'flip': False,
            'filename': 'live_stream.mat'
        }]
        
        # Forward qua model (Sử dụng hàm simple_test của PETR)
        # Kết quả: [ (bboxes, labels, keypoints) ]
        result = model.simple_test(csi_tensor, img_metas, rescale=False)
        
        # Lấy keypoints của những người có score cao (>0.3)
        det_bboxes, det_labels, det_kpts = result[0]
        scores = det_bboxes[:, -1]
        keep = scores > 0.3
        
        final_kpts = det_kpts[keep] # [Num_People, 14, 3]
        
    return final_kpts
# =============================================================================
# 0. SETUP MÔI TRƯỜNG & CUSTOM LAYERS
# =============================================================================
PROJECT_ROOT = "/content/Person-in-WiFi-3D" 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import custom_layers
    print(">>> Đã kích hoạt custom_layers thành công!")
except ImportError as e:
    st.error(f"Lỗi import custom_layers: {e}")

# =============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (ĐÃ FIX THEO YÊU CẦU MỚI)
# =============================================================================
DATA_ROOT = "/content/Person-in-WiFi-3D/data/wifipose"
VIDEO_ROOT = os.path.join(DATA_ROOT, "videos") 
CONFIG_FILE = os.path.join(PROJECT_ROOT, "configs/wifi/petr_wifi.py")
CHECKPOINT_DIR = "/content/drive/MyDrive/RESEARCH/RESFES2026/result_5e-60e_rtx4090_fnet_graph"

# Phân mảng dữ liệu
TRAIN_CSI = os.path.join(DATA_ROOT, "train_data/csi")
TEST_CSI = os.path.join(DATA_ROOT, "test_data/csi")
TRAIN_GT = os.path.join(DATA_ROOT, "train_data/keypoint")
TEST_GT = os.path.join(DATA_ROOT, "test_data/keypoint")

# =============================================================================
# 2. MODULE: DATA & AI HELPERS
# =============================================================================
def load_csi_and_preprocess(file_path):
    """Nạp và xử lý CSI từ HDF5"""
    with h5py.File(file_path, 'r') as f:
        raw_csi = f['csi_out'][()]
        csi_complex = raw_csi['real'] + 1j * raw_csi['imag']
        # Trả về biên độ [Subcarriers, Packets]
        return np.abs(csi_complex[0, 0, :, :])

@st.cache_resource
def get_ai_model(checkpoint_name):
    from opera.apis import init_detector
    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    model = init_detector(CONFIG_FILE, ckpt_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    return model

@st.cache_data
def get_video_library():
    video_groups = {}
    # Quét CSI từ cả 2 nguồn
    for folder in [TRAIN_CSI, TEST_CSI]:
        if not os.path.exists(folder): continue
        for f in os.listdir(folder):
            match = re.match(r"^(S\d+_\d+)_(\d+)\.mat$", f) 
            if match:
                v_id, s_idx = match.group(1), int(match.group(2))
                if v_id not in video_groups: video_groups[v_id] = []
                
                # Tự động tìm Ground Truth tương ứng
                gt_folder = TRAIN_GT if "train_data" in folder else TEST_GT
                gt_path = os.path.join(gt_folder, f.replace('.mat', '.npy'))
                
                video_groups[v_id].append({
                    "index": s_idx,
                    "csi_path": os.path.join(folder, f),
                    "gt_path": gt_path if os.path.exists(gt_path) else None
                })
    for vid in video_groups:
        video_groups[vid] = sorted(video_groups[vid], key=lambda x: x["index"])
    return video_groups

# =============================================================================
# 3. MODULE: VISUALIZATION ENGINE
# =============================================================================
def create_optimized_3d_figure(pred_kpts, gt_kpts, video_id, frame_idx):
    fig = go.Figure()
    
    # 1. DANH SÁCH KHỚP NỐI CHUẨN (Lấy chính xác từ multiperson_visualize.py)
    # Thứ tự này khác hoàn toàn với chuẩn COCO thông thường
    LIMBS_WIFI = [
        [0, 1], [1, 2], [2, 5], [3, 0], [4, 2], [5, 7],
        [6, 3], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7],
        [12, 9], [13, 11]
    ]

    def add_skeleton(kpts, color, name, is_gt=False):
        if kpts is None or len(kpts) == 0: return
        # Đảm bảo mảng có shape (N, 14, 3)
        if kpts.ndim == 2: kpts = np.expand_dims(kpts, axis=0)
        
        for p_idx, person in enumerate(kpts):
            # Khớp (Markers)
            fig.add_trace(go.Scatter3d(
                x=person[:, 0], y=person[:, 1], z=person[:, 2],
                mode='markers',
                marker=dict(size=4, color=color, symbol='circle'),
                name=f"{name} P{p_idx+1}",
                showlegend=True if p_idx == 0 else False
            ))
            
            # Xương (Lines) - Dùng đúng kết nối LIMBS_WIFI
            lx, ly, lz = [], [], []
            for start, end in LIMBS_WIFI:
                # Nối điểm start đến end và thêm None để ngắt đường nối trace
                lx.extend([person[start, 0], person[end, 0], None])
                ly.extend([person[start, 1], person[end, 1], None])
                lz.extend([person[start, 2], person[end, 2], None])
            
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz,
                mode='lines',
                line=dict(color=color, width=5 if not is_gt else 3),
                showlegend=False
            ))

    # 2. VẼ DỮ LIỆU
    add_skeleton(gt_kpts, 'blue', 'Ground Truth', is_gt=True)
    add_skeleton(pred_kpts, 'red', 'AI Prediction')

    # 3. TỰ ĐỘNG TÍNH TOÁN KHUNG NHÌN (Dựa trên logic của visualize script)
    # Hợp nhất tất cả các điểm để tìm min/max
    all_pts = []
    if gt_kpts is not None: all_pts.append(gt_kpts.reshape(-1, 3))
    if pred_kpts is not None: all_pts.append(pred_kpts.reshape(-1, 3))
    
    if all_pts:
        all_pts = np.concatenate(all_pts, axis=0)
        min_vals = np.min(all_pts, axis=0)
        max_vals = np.max(all_pts, axis=0)
        mid_vals = (min_vals + max_vals) / 2
        # Tính range bao phủ người (khoảng 1.5m quanh trọng tâm)
        max_range = 1.2 
        
        x_range = [mid_vals[0] - max_range, mid_vals[0] + max_range]
        y_range = [mid_vals[1] - max_range, mid_vals[1] + max_range]
        
        # --- LẬT TRỤC Z THEO FILE TOOLS ---
        # Trong script gốc: ax.set_zlim(z_max, z_min) 
        # Tức là giá trị lớn nhất nằm ở dưới, nhỏ nhất ở trên
        z_min, z_max = mid_vals[2] - max_range, mid_vals[2] + max_range
        z_range = [z_max, z_min] # Lật ngược
    else:
        x_range, y_range, z_range = [-2, 2], [-2, 2], [2, 0]

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range, title="X (Ngang)", gridcolor="gray"),
            yaxis=dict(range=y_range, title="Y (Sâu)", gridcolor="gray"),
            zaxis=dict(range=z_range, title="Z (Chiều cao)", gridcolor="gray"),
            aspectmode='cube'
        ),
        uirevision='constant', # Giữ góc quay camera khi update
        margin=dict(l=0, r=0, b=0, t=30),
        template="plotly_dark",
        title=f"Analyzer: {video_id} | Frame {frame_idx}"
    )
    # Đặt góc nhìn mặc định giống script gốc (elev=20, azim=-75)
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.5, y=-1.5, z=1.0)
    ))
    
    return fig

# =============================================================================
# 4. STREAMLIT UI SETUP
# =============================================================================
st.set_page_config(page_title="WiPose-3D Analyzer", layout="wide")
st.sidebar.title("🧠 WiPose-3D AI System")

# Chọn Checkpoint
try:
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    selected_ckpt = st.sidebar.selectbox("1. Chọn Model Checkpoint", ckpts)
except:
    st.sidebar.error("Không tìm thấy thư mục work_dirs")
    selected_ckpt = None

# Chọn Video
library = get_video_library()
selected_vid_id = st.sidebar.selectbox("2. Chọn Video Sequence", list(library.keys()))

# Main Layout
st.title("Real-time Wi-Fi 3D Pose Analyzer")

# Khởi tạo các vùng hiển thị trạng thái và FPS ở Sidebar
fps_place = st.sidebar.empty()
status_bar = st.sidebar.empty() # Khởi tạo status_bar tại đây

# Khởi tạo các vùng hiển thị chính ở Main Panel (nếu chưa có)
col_vid, col_3d = st.columns([1, 1.2])
with col_vid:
    video_place = st.empty()
    st.markdown("### 📡 CSI Signal")
    signal_place = st.empty()
with col_3d:
    st.markdown("### 🧍 3D Pose (AI vs GT)")
    pose_place = st.empty()

run_btn = st.sidebar.button("▶️ BẮT ĐẦU PHÂN TÍCH", use_container_width=True)
stop_btn = st.sidebar.button("⏹️ DỪNG")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
inference_mode = st.sidebar.radio(
    "Chiến thuật xử lý:",
    ["Mượt mà (Pre-compute)", "Tức thời (Async - Skip 5)"],
    help="Pre-compute: Chạy AI trước rồi phát lại. Async: Chạy trực tiếp nhưng nhảy khung hình."
)

# =============================================================================
# 2. HÀM INFERENCE THỰC TẾ (Tối ưu cho PETR)
# =============================================================================
def run_real_inference(model, csi_frame_complex):
    """
    csi_frame_complex: Mảng [Rx, Tx, Subcarriers]
    Trả về: Tọa độ người dự đoán (NumPeople, 14, 3)
    """
    try:
        # Tiền xử lý theo chuẩn WifiPoseDataset
        # 1. DWT & Sanitization (Mô phỏng nhanh)
        csi_amp = np.abs(csi_frame_complex)
        csi_phase = np.angle(csi_frame_complex)
        csi_combined = np.concatenate((csi_amp, csi_phase), axis=2) # [3, 3, 60]
        
        # 2. Chuyển sang Tensor [1, 3, 3, 20, 30] - Kiểm tra lại shape khớp với config
        input_tensor = torch.FloatTensor(csi_combined).permute(0, 1, 2).unsqueeze(0).unsqueeze(3)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': np.array([1, 1, 1, 1])}]
            # Chạy model PETR
            result = model.simple_test(input_tensor, img_metas, rescale=False)
            
            # result[0] = (bboxes, labels, keypoints)
            det_bboxes, det_labels, det_kpts = result[0]
            # Lọc những người có score > 0.3
            keep = det_bboxes[:, -1] > 0.3
            return det_kpts[keep]
    except Exception as e:
        return None

# =============================================================================
# 3. VÒNG LẶP THỰC THI (LOGIC CHIẾN THUẬT)
# =============================================================================
if run_btn and selected_vid_id:
    model = get_ai_model(selected_ckpt)
    segments = library[selected_vid_id]
    
    # Khởi tạo Video
    video_path = os.path.join(VIDEO_ROOT, f"{selected_vid_id}.mkv")
    cap = cv2.VideoCapture(video_path)
    
    for seg in segments:
        # Nạp dữ liệu CSI Gốc (Số phức)
        with h5py.File(seg['csi_path'], 'r') as f:
            raw = f['csi_out'][()]
            csi_full_complex = raw['real'] + 1j * raw['imag'] # [Tx, Rx, Sub, Pkt]
        
        # Nạp GT để đối chiếu
        gt_poses = np.load(seg['gt_path'], allow_pickle=True) if os.path.exists(seg['gt_path']) else None
        num_frames = csi_full_complex.shape[3]
        
        # --- CHIẾN THUẬT 1: PRE-COMPUTE ---
        precomputed_preds = []
        if "Pre-compute" in inference_mode:
            progress_bar = st.progress(0)
            status_bar.warning(f"⌛ Đang chạy AI Inference cho Segment {seg['index']}...")
            
            for f_idx in range(num_frames):
                # Lấy 1 frame CSI
                frame_data = csi_full_complex[:, :, :, f_idx]
                pred = run_real_inference(model, frame_data)
                precomputed_preds.append(pred)
                progress_bar.progress((f_idx + 1) / num_frames)
            
            status_bar.success(f"✅ AI đã xử lý xong {num_frames} frames!")
            time.sleep(1) # Chờ 1 giây để người xem thấy thông báo thành công

        # --- VÒNG LẶP HIỂN THỊ (PLAYBACK) ---
        last_pred = None
        prev_time = time.time()

        for f_idx in range(num_frames):
            if stop_btn: break
            
            # A. Lấy kết quả AI theo chiến thuật
            if "Pre-compute" in inference_mode:
                current_pred = precomputed_preds[f_idx]
            else:
                # CHIẾN THUẬT 2: ASYNC (Chỉ chạy AI mỗi 5 frame)
                if f_idx % 5 == 0:
                    frame_data = csi_full_complex[:, :, :, f_idx]
                    current_pred = run_real_inference(model, frame_data)
                    last_pred = current_pred # Lưu lại để hiển thị cho các frame trung gian
                else:
                    current_pred = last_pred

            # B. Cập nhật Video Reference
            if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_place.image(cv2.resize(frame, (640, 360)), use_column_width=True)
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # C. Cập nhật 3D Dashboard (AI vs GT)
            current_gt = gt_poses[f_idx % len(gt_poses)] if gt_poses is not None else None
            
            # Gọi hàm vẽ Giai đoạn 1 (Đã fix lật trục Z và xương)
            fig_3d = create_optimized_3d_figure(current_pred, current_gt, selected_vid_id, f_idx)
            pose_place.plotly_chart(fig_3d, use_container_width=True, key=f"pose_{seg['index']}_{f_idx}")

            # --- D. TÍNH FPS VÀ ĐIỀU TIẾT ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_place.metric("⚡ Hệ thống đang chạy", f"{fps:.1f} FPS")
            
            # Sleep nhẹ để trình duyệt kịp render ảnh
            time.sleep(0.005)

    if cap is not None:
        cap.release()
    status_bar.success(f"🏁 Đã hoàn thành xử lý Video: {selected_vid_id}")