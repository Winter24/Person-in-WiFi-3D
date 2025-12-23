import streamlit as st
import numpy as np
import scipy.io as sio
import h5py
import time
import pandas as pd
import plotly.graph_objects as go
import os
import re

# =============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THÔNG SỐ (HÃY ĐIỀU CHỈNH TẠI ĐÂY)
# =============================================================================
TRAIN_PATH = "data/wifipose/train_data/csi"
TEST_PATH = "data/wifipose/test_data/csi"
# Thư mục chứa kết quả dự đoán .npy (tên file .npy phải khớp tên file .mat)
POSE_RESULTS_PATH = "data/wifipose/pose_results" 

# Metadata cho 14 khớp (CrowdPose/WiFiPose Standard)
JOINT_NAMES = [
    'L-Sho', 'R-Sho', 'L-Elb', 'R-Elb', 'L-Wri', 'R-Wri', 
    'L-Hip', 'R-Hip', 'L-Kne', 'R-Kne', 'L-Ank', 'R-Ank', 'Head', 'Neck'
]

# Kết nối xương (Limbs)
LIMBS = [
    (13, 12), (13, 0), (13, 1), # Đầu - Cổ - Vai
    (0, 2), (2, 4),             # Tay trái
    (1, 3), (3, 5),             # Tay phải
    (0, 6), (1, 7), (6, 7),     # Thân trên - Hông
    (6, 8), (8, 10),            # Chân trái
    (7, 9), (9, 11)             # Chân phải
]

# =============================================================================
# 2. MODULE: SMART DATA AGGREGATOR (HỢP NHẤT DỮ LIỆU)
# =============================================================================
@st.cache_data
def get_video_library():
    """Quét train/test và gom nhóm file theo Video ID (Sxx_xx)"""
    video_groups = {}
    
    # Quét cả 2 nguồn dữ liệu
    sources = [("Train", TRAIN_PATH), ("Test", TEST_PATH)]
    
    for source_name, folder in sources:
        if not os.path.exists(folder):
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith('.mat')]
        for f in files:
            # Regex tách chuỗi: S11_01_10 -> Group 1: S11_01, Group 2: 10
            match = re.match(r"^(.*)_(\d+)\.mat$", f)
            if match:
                video_id = match.group(1)
                seg_idx = int(match.group(2))
                
                if video_id not in video_groups:
                    video_groups[video_id] = []
                    
                video_groups[video_id].append({
                    "index": seg_idx,
                    "filename": f,
                    "csi_path": os.path.join(folder, f),
                    "source": source_name
                })
    
    # Sắp xếp index cho từng video
    for vid in video_groups:
        video_groups[vid] = sorted(video_groups[vid], key=lambda x: x["index"])
        
    return video_groups

def check_continuity(segments):
    indices = [s["index"] for s in segments]
    gaps = []
    for i in range(len(indices) - 1):
        if indices[i+1] != indices[i] + 1:
            gaps.append(f"Missing between {indices[i]} and {indices[i+1]}")
    return gaps

# =============================================================================
# 3. MODULE: DATA LOADERS
# =============================================================================
def load_csi_amplitude(file_path):
    """Nạp biên độ CSI từ file .mat"""
    try:
        with h5py.File(file_path, 'r') as f:
            csi = f['csi_out'][()]
            csi = csi['real'] + csi['imag'] * 1j
            # Lấy abs và transpose về (Packets, Subcarriers)
            return np.abs(csi[0, 0, :, :]).T 
    except:
        data = sio.loadmat(file_path)
        csi = data['csi_out']
        # Đối với chuẩn scipy thường là (Tx, Rx, Sub, Pkt)
        return np.abs(csi[0, 0, :, :]).T

def load_pose_npy(filename):
    """Nạp file dự đoán .npy tương ứng"""
    pose_file = filename.replace('.mat', '.npy')
    full_path = os.path.join(POSE_RESULTS_PATH, pose_file)
    if os.path.exists(full_path):
        return np.load(full_path, allow_pickle=True)
    return None

# =============================================================================
# 4. MODULE: VISUALIZATION ENGINES
# =============================================================================
def create_3d_pose_fig(person_kpts, frame_info=""):
    """Vẽ bộ xương 3D chuyên nghiệp"""
    # Đảm bảo mảng luôn là 3D (NumPeople, 14, 3)
    if person_kpts.ndim == 2:
        person_kpts = np.expand_dims(person_kpts, axis=0)
    
    fig = go.Figure()
    
    # Vẽ từng người
    colors = ['#FF3B30', '#4CD964', '#007AFF', '#FFCC00']
    
    for p_idx, kpts in enumerate(person_kpts):
        if kpts.shape[0] < 14: continue
        color = colors[p_idx % len(colors)]
        
        # Vẽ Khớp
        fig.add_trace(go.Scatter3d(
            x=kpts[:, 0], y=kpts[:, 1], z=kpts[:, 2],
            mode='markers',
            marker=dict(size=4, color=color),
            name=f'Subject {p_idx+1}'
        ))
        
        # Vẽ Xương
        for start, end in LIMBS:
            fig.add_trace(go.Scatter3d(
                x=[kpts[start, 0], kpts[end, 0]],
                y=[kpts[start, 1], kpts[end, 1]],
                z=[kpts[start, 2], kpts[end, 2]],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False
            ))

    fig.update_layout(
        title=f"3D Pose Estimation - {frame_info}",
        scene=dict(
            xaxis=dict(range=[-2.5, 2.5], title="X (m)"),
            yaxis=dict(range=[-2.5, 2.5], title="Y (m)"),
            zaxis=dict(range=[-0.5, 2.5], title="Z (m)"), # Trục đứng
            aspectmode='cube',
            bgcolor="black"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark"
    )
    return fig

# =============================================================================
# 5. GIAO DIỆN CHÍNH (STREAMLIT UI)
# =============================================================================
st.set_page_config(page_title="WiPose-3D Smart Dashboard", layout="wide")

# Sidebar
st.sidebar.title("🛰️ WiPose-3D v1.0")
st.sidebar.markdown("---")

library = get_video_library()
selected_vid = st.sidebar.selectbox("🎥 Chọn Video Sequence", list(library.keys()))

if selected_vid:
    segs = library[selected_vid]
    gaps = check_continuity(segs)
    
    with st.sidebar.expander("Chi tiết Sequence", expanded=True):
        st.write(f"Số đoạn (Segments): **{len(segs)}**")
        st.write(f"Bắt đầu: `{segs[0]['index']}` | Kết thúc: `{segs[-1]['index']}`")
        if gaps:
            st.error("⚠️ Phát hiện lỗ hổng dữ liệu!")
            for g in gaps: st.caption(g)
        else:
            st.success("✅ Chuỗi dữ liệu liên tục")

run_btn = st.sidebar.button("▶️ CHẠY DEMO HỆ THỐNG", use_container_width=True)
stop_btn = st.sidebar.button("⏹️ DỪNG", use_container_width=True)

# Main UI
st.title("Hệ thống Giám sát Tư thế 3D qua Tín hiệu Wi-Fi")
st.info("Giải pháp bảo vệ quyền riêng tư & Hoạt động trong môi trường không ánh sáng.")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bảng thi", "AI & Blockchain")
m2.metric("Trạng thái", "Inference Mode")
m3.metric("Công nghệ", "PETR + FNet")
m4.metric("Privacy", "🛡️ Encrypted")

# Containers cho Real-time
col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("📡 Sóng Wi-Fi Sensing (CSI)")
    signal_place = st.empty()
with col_right:
    st.subheader("🧍 Kết quả AI Reconstruction")
    pose_place = st.empty()

status_bar = st.empty()

# =============================================================================
# 6. VÒNG LẶP XỬ LÝ CHÍNH
# =============================================================================
if run_btn:
    segments = library[selected_vid]
    
    # Khởi tạo buffer cho biểu đồ sóng
    buffer_size = 60
    sc_to_show = 5 # Hiện 5 subcarriers
    wave_buffer = pd.DataFrame(np.zeros((buffer_size, sc_to_show)))

    for seg in segments:
        status_bar.warning(f"🔄 Đang tải Segment {seg['index']} từ {seg['source']}...")
        
        # Load Data
        amp_data = load_csi_amplitude(seg['csi_path'])
        poses = load_pose_npy(seg['filename'])
        
        num_frames = amp_data.shape[0]
        
        for f in range(num_frames):
            if stop_btn: break
            
            # --- Xử lý Sóng ---
            new_val = amp_data[f, :sc_to_show]
            wave_buffer = pd.concat([wave_buffer.iloc[1:], pd.DataFrame([new_val])], ignore_index=True)
            
            fig_sig = go.Figure()
            for sc in range(sc_to_show):
                fig_sig.add_trace(go.Scatter(y=wave_buffer[sc], mode='lines', line=dict(width=1.5)))
            fig_sig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", 
                                 xaxis=dict(showticklabels=False), yaxis=dict(showgrid=False))
            signal_place.plotly_chart(fig_sig, use_container_width=True, key=f"sig_{seg['index']}_{f}")

            # --- Xử lý Pose ---
            if poses is not None:
                # Đồng bộ frame (giả sử tần số lấy mẫu CSI và Pose khớp nhau)
                curr_pose = poses[f % len(poses)]
                
                # Check té ngã đơn giản (Nếu khớp đầu hạ thấp bất thường)
                # Giả sử trục Z là trục đứng (index 2)
                is_fall = False
                if curr_pose.ndim == 2: # 1 người
                    if curr_pose[12, 2] < 0.5: is_fall = True # Khớp 12 là Head
                
                info_str = f"Seq: {selected_vid} | Seg: {seg['index']} | Frame: {f}"
                if is_fall: info_str += " ⚠️ FALL DETECTED!"
                
                fig_pose = create_3d_pose_fig(curr_pose, frame_info=info_str)
                pose_place.plotly_chart(fig_pose, use_container_width=True, key=f"pose_{seg['index']}_{f}")
            else:
                pose_place.error("Không có dữ liệu Pose cho segment này.")

            # Tốc độ demo (ms)
            time.sleep(0.02)
            
        if stop_btn: 
            st.sidebar.info("Đã dừng giám sát.")
            break

    status_bar.success("🏁 Hoàn thành trình chiếu Sequence.")
else:
    st.info("Vui lòng chọn Video Sequence và nhấn 'Bắt đầu' để trình diễn.")