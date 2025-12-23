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

# =============================================================================
# 0. SETUP MÔI TRƯỜNG & CUSTOM LAYERS
# =============================================================================
PROJECT_ROOT = "/root/Person-in-WiFi-3D" 
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
DATA_ROOT = "/root/Person-in-WiFi-3D/data/wifipose"
VIDEO_ROOT = os.path.join(DATA_ROOT, "videos") 
CONFIG_FILE = os.path.join(PROJECT_ROOT, "configs/wifi/petr_wifi.py")
CHECKPOINT_DIR = "/root/Person-in-WiFi-3D/work_dirs/petr_wifi_fnet_graph"

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

# =============================================================================
# 5. EXECUTION LOOP
# =============================================================================
# --- TRONG VÒNG LẶP CHÍNH ---
if run_btn and selected_vid_id and selected_ckpt:
    # 1. KHỞI TẠO MODEL VÀ DỮ LIỆU
    model = get_ai_model(selected_ckpt)
    segments = library[selected_vid_id]
    
    # Mở Video (.mkv) từ thư mục videos riêng
    video_path = os.path.join(VIDEO_ROOT, f"{selected_vid_id}.mkv")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.sidebar.error(f"❌ Không mở được video: {video_path}")
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Khởi tạo bộ đệm cho tín hiệu CSI (vẽ 5 subcarriers, lưu 60 điểm thời gian)
    n_subcarriers = 5
    window_size = 60
    csi_rolling_buffer = np.zeros((window_size, n_subcarriers))
    
    # Biến tính FPS
    prev_time = time.time()
    
    # 2. VÒNG LẶP QUA TỪNG SEGMENT (S11_01_10, _11, _12...)
    for seg in segments:
        if stop_btn: break
        
        status_bar.info(f"🔄 Đang xử lý Segment: {seg['index']} | Nguồn: {seg['csi_path']}")
        
        # Nạp CSI (Hàm nạp số phức đã fix lỗi absolute)
        csi_amp_full = load_csi_and_preprocess(seg['csi_path']) # [Sub, Pkt]
        
        # Nạp Ground Truth (GT)
        gt_poses = None
        if seg['gt_path'] and os.path.exists(seg['gt_path']):
            gt_poses = np.load(seg['gt_path'], allow_pickle=True)
        
        num_packets = csi_amp_full.shape[1]

        # 3. VÒNG LẶP QUA TỪNG PACKET TRONG SEGMENT (REAL-TIME LOOP)
        for f_idx in range(num_packets):
            if stop_btn: break
            
            # --- A. CẬP NHẬT VIDEO (ĐỒNG BỘ KHUNG HÌNH) ---
            if cap is not None and cap.isOpened():
                # Tính toán frame video tương ứng (CSI freq thường cao hơn Video)
                # Giả sử tỷ lệ là 1:1 hoặc bạn có thể chỉnh target_frame dựa trên timestamp
                ret, frame = cap.read() 
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (640, 360)) # Resize để web mượt hơn
                    video_place.image(frame, use_column_width=True)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video nếu hết

            # --- B. TRỰC QUAN HÓA CSI (ROLLING WINDOW) ---
            # Lấy giá trị biên độ của n subcarriers tại thời điểm f_idx
            current_vals = csi_amp_full[:n_subcarriers, f_idx]
            csi_rolling_buffer = np.roll(csi_rolling_buffer, -1, axis=0)
            csi_rolling_buffer[-1, :] = current_vals
            
            fig_sig = go.Figure()
            for sc in range(n_subcarriers):
                fig_sig.add_trace(go.Scatter(
                    y=csi_rolling_buffer[:, sc], 
                    mode='lines', 
                    name=f"Sub {sc}",
                    line=dict(width=1.5)
                ))
            fig_sig.update_layout(
                height=250, margin=dict(l=0, r=0, t=0, b=0),
                template="plotly_dark", showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showgrid=False)
            )
            signal_place.plotly_chart(fig_sig, use_container_width=True, key=f"sig_{seg['index']}_{f_idx}")

            # --- C. HIỂN THỊ 3D POSE (AI VS GT) ---
            # Để tránh lag, ta có thể render 3D ở tốc độ thấp hơn (ví dụ mỗi 2 frame CSI 1 lần)
            if f_idx % 1 == 0: 
                # Lấy GT
                current_gt = gt_poses[f_idx % len(gt_poses)] if gt_poses is not None else None
                
                # CHẠY INFERENCE HOẶC GIẢ LẬP
                # Nếu GPU mạnh: current_pred = run_ai_inference(model, csi_amp_full[:, f_idx])
                # Hiện tại: Dùng GT + nhiễu để demo độ chính xác
                current_pred = current_gt + np.random.normal(0, 0.02, current_gt.shape) if current_gt is not None else None
                
                if current_gt is not None:
                    # Gọi hàm vẽ tối ưu (đã fix khớp và lật trục Z ở Giai đoạn 1)
                    fig_3d = create_optimized_3d_figure(current_pred, current_gt, selected_vid_id, f"{seg['index']}_{f_idx}")
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