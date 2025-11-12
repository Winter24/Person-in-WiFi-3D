# # --- Hướng dẫn sử dụng ---
# # 1. Cài đặt các thư viện cần thiết nếu chưa có:
# #    pip install numpy matplotlib h5py scipy tqdm scikit-learn
# #
# # 2. THAY ĐỔI ĐƯỜNG DẪN DƯỚI ĐÂY:
# #    Chỉnh sửa biến `ROOT_DATA_PATH` để trỏ đến thư mục data của bạn.
# #
# # 3. Chạy toàn bộ script.

# import matplotlib.pyplot as plt
# import h5py
# import scipy.io as sio
# import os
# import numpy as np
# from tqdm import tqdm
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# ################################################################################
# ### PHẦN 1: CÁC THIẾT LẬP VÀ HÀM TIỆN ÍCH
# ################################################################################

# # !!! QUAN TRỌNG: Hãy thay đổi đường dẫn này đến thư mục gốc chứa dữ liệu của bạn !!!
# ROOT_DATA_PATH = "/home/winter24/Person-in-WiFi-3D-repo/data/wifipose" 

# TRAIN_DATA_PATH = os.path.join(ROOT_DATA_PATH, "train_data")

# def validate_dataset_integrity_optimized(data_path, list_filename):
#     """Kiểm tra tính toàn vẹn của bộ dữ liệu."""
#     print(f"\n--- Bắt đầu kiểm tra thư mục: {os.path.basename(data_path)} ---")
#     list_file_path = os.path.join(data_path, list_filename)
#     csi_dir = os.path.join(data_path, "csi")
#     keypoint_dir = os.path.join(data_path, "keypoint")
    
#     if not all(os.path.exists(p) for p in [list_file_path, csi_dir, keypoint_dir]):
#         print(f"!!! Lỗi: Không tìm thấy tệp hoặc thư mục cần thiết trong {data_path}")
#         return []

#     csi_files_set = set(os.listdir(csi_dir))
#     keypoint_files_set = set(os.listdir(keypoint_dir))

#     with open(list_file_path, 'r') as f:
#         all_ids = [line.strip() for line in f if line.strip()]
        
#     print(f"Tìm thấy {len(all_ids)} ID. Bắt đầu xác thực...")
#     valid_ids = [
#         sample_id for sample_id in tqdm(all_ids, desc="Xác thực")
#         if f"{sample_id}.mat" in csi_files_set and f"{sample_id}.npy" in keypoint_files_set
#     ]
#     print(f">>> Kiểm tra hoàn tất. Tổng số mẫu hợp lệ: {len(valid_ids)} / {len(all_ids)}")
#     return valid_ids

# def load_csi_sample(data_path, sample_id):
#     """Tải dữ liệu CSI cho một sample_id cụ thể."""
#     csi_file_path = os.path.join(data_path, "csi", f"{sample_id}.mat")
#     try:
#         with h5py.File(csi_file_path, 'r') as f:
#             csi_data_complex = f['csi_out']
#             csi_data = csi_data_complex['real'][:] + 1j * csi_data_complex['imag'][:]
#             csi_data = csi_data.transpose(3, 2, 1, 0)
#     except Exception:
#         mat_contents = sio.loadmat(csi_file_path)
#         csi_data = mat_contents['csi_out']
#     return csi_data

# ################################################################################
# ### PHẦN 2: TẢI DỮ LIỆU VÀ XỬ LÝ PCA
# ################################################################################

# train_ids = validate_dataset_integrity_optimized(TRAIN_DATA_PATH, "train_data_list.txt")

# if not train_ids:
#     print("\n!!! Không tìm thấy mẫu dữ liệu hợp lệ. Vui lòng kiểm tra lại `ROOT_DATA_PATH`.")
# else:
#     sample_id_to_analyze = np.random.choice(train_ids)
#     print(f"\n--- Phân tích mẫu CSI: {sample_id_to_analyze} ---")

#     # Tải dữ liệu CSI thô
#     csi_sample = load_csi_sample(TRAIN_DATA_PATH, sample_id_to_analyze)
    
#     # 1. Lấy biên độ (amplitude) từ dữ liệu CSI phức
#     amplitude_sample = np.abs(csi_sample)
#     num_tx, num_rx, num_subcarriers, num_packets = amplitude_sample.shape

#     # 2. Chuẩn bị dữ liệu cho PCA
#     # PCA yêu cầu dữ liệu 2D dạng (n_samples, n_features)
#     # Ở đây, mỗi packet (mẫu thời gian) là một sample.
#     # Các features là tất cả các luồng CSI (Tx * Rx * Subcarrier) gộp lại.
#     # Shape ban đầu: (Tx, Rx, Sub, Pkt) -> Reshape: (Tx*Rx*Sub, Pkt) -> Transpose: (Pkt, Tx*Rx*Sub)
#     features = amplitude_sample.reshape(-1, num_packets).T
    
#     # 3. Chuẩn hóa dữ liệu (bước quan trọng cho PCA)
#     # Đưa tất cả các feature về cùng một thang đo (trung bình 0, phương sai 1)
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)

#     # 4. Áp dụng PCA
#     # Chúng ta sẽ trích xuất 3 thành phần chính để vẽ "Spectrum"
#     n_components = 3
#     pca = PCA(n_components=n_components)
#     principal_components = pca.fit_transform(features_scaled)
    
#     print(f"Đã áp dụng PCA và trích xuất {principal_components.shape[1]} thành phần chính.")

#     ################################################################################
#     ### PHẦN 3: TRỰC QUAN HÓA SPECTRUM VÀ SIGNAL
#     ################################################################################

#     # Tạo một cửa sổ chứa 2 đồ thị con (1 hàng, 2 cột)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
#     fig.suptitle(f"Phân tích PCA trên tín hiệu CSI - Mẫu: {sample_id_to_analyze}", fontsize=16)

#     # --- ĐỒ THỊ 1: SPECTRUM (Bên trái) ---
#     # Đây là heatmap của các thành phần chính theo thời gian.
#     # Ta cần chuyển vị ma trận principal_components để trục thời gian nằm ngang.
#     spectrum_data = principal_components.T
    
#     # Sử dụng imshow để vẽ heatmap, cmap='jet' để có dải màu giống hình mẫu
#     im = ax1.imshow(spectrum_data, aspect='auto', cmap='jet', origin='lower', 
#                     extent=[0, num_packets, 0.5, n_components + 0.5])
    
#     ax1.set_title("Spectrum")
#     ax1.set_xlabel("Time (Frame Index)")
#     ax1.set_ylabel("Features (Principal Components)")
#     ax1.set_yticks(np.arange(1, n_components + 1)) # Đặt nhãn cho các thành phần
#     fig.colorbar(im, ax=ax1)

#     # --- ĐỒ THỊ 2: SIGNAL (Bên phải) ---
#     # Tín hiệu này chính là Thành phần chính đầu tiên (PC1)
#     signal_pc1 = principal_components
    
#     ax2.plot(signal_pc1)
    
#     # Thêm điểm màu đỏ để mô phỏng "Association" giống trong hình
#     # Bạn có thể thay đổi vị trí điểm này nếu muốn
#     association_frame_index = 100
#     if association_frame_index < len(signal_pc1):
#          ax2.plot(association_frame_index, signal_pc1[association_frame_index], 'ro', markersize=8, label='Association Point')

#     ax2.set_title("Signal")
#     ax2.set_xlabel("Frame Index")
#     ax2.set_ylabel("Amplitude (Component Score)")
#     ax2.grid(True)
#     ax2.legend()
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.93])
#     plt.show()

# --- Hướng dẫn sử dụng ---
# 1. Cài đặt các thư viện cần thiết nếu chưa có:
#    pip install numpy matplotlib h5py scipy tqdm scikit-learn
#
# 2. THAY ĐỔI ĐƯỜNG DẪN DƯỚI ĐÂY:
#    Chỉnh sửa biến `ROOT_DATA_PATH` để trỏ đến thư mục data của bạn.
#
# 3. Chạy toàn bộ script.

import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

################################################################################
### PHẦN 1: CÁC THIẾT LẬP VÀ HÀM TIỆN ÍCH
################################################################################

# !!! QUAN TRỌNG: Hãy thay đổi đường dẫn này đến thư mục gốc chứa dữ liệu của bạn !!!
ROOT_DATA_PATH = "/home/winter24/Person-in-WiFi-3D-repo/data/wifipose" 

TRAIN_DATA_PATH = os.path.join(ROOT_DATA_PATH, "train_data")

def validate_dataset_integrity_optimized(data_path, list_filename):
    """Kiểm tra tính toàn vẹn của bộ dữ liệu."""
    print(f"\n--- Bắt đầu kiểm tra thư mục: {os.path.basename(data_path)} ---")
    list_file_path = os.path.join(data_path, list_filename)
    csi_dir = os.path.join(data_path, "csi")
    keypoint_dir = os.path.join(data_path, "keypoint")
    
    if not all(os.path.exists(p) for p in [list_file_path, csi_dir, keypoint_dir]):
        print(f"!!! Lỗi: Không tìm thấy tệp hoặc thư mục cần thiết trong {data_path}")
        return []

    csi_files_set = set(os.listdir(csi_dir))
    keypoint_files_set = set(os.listdir(keypoint_dir))

    with open(list_file_path, 'r') as f:
        all_ids = [line.strip() for line in f if line.strip()]
        
    print(f"Tìm thấy {len(all_ids)} ID. Bắt đầu xác thực...")
    valid_ids = [
        sample_id for sample_id in tqdm(all_ids, desc="Xác thực")
        if f"{sample_id}.mat" in csi_files_set and f"{sample_id}.npy" in keypoint_files_set
    ]
    print(f">>> Kiểm tra hoàn tất. Tổng số mẫu hợp lệ: {len(valid_ids)} / {len(all_ids)}")
    return valid_ids

def load_csi_sample(data_path, sample_id):
    """Tải dữ liệu CSI cho một sample_id cụ thể."""
    csi_file_path = os.path.join(data_path, "csi", f"{sample_id}.mat")
    try:
        with h5py.File(csi_file_path, 'r') as f:
            csi_data_complex = f['csi_out']
            csi_data = csi_data_complex['real'][:] + 1j * csi_data_complex['imag'][:]
            csi_data = csi_data.transpose(3, 2, 1, 0)
    except Exception:
        mat_contents = sio.loadmat(csi_file_path)
        csi_data = mat_contents['csi_out']
    return csi_data

################################################################################
### PHẦN 2: TẢI DỮ LIỆU VÀ TRỰC QUAN HÓA DỮ LIỆU THÔ
################################################################################

train_ids = validate_dataset_integrity_optimized(TRAIN_DATA_PATH, "train_data_list.txt")

if not train_ids:
    print("\n!!! Không tìm thấy mẫu dữ liệu hợp lệ. Vui lòng kiểm tra lại `ROOT_DATA_PATH`.")
else:
    # Chọn ngẫu nhiên một mẫu để phân tích
    sample_id_to_analyze = np.random.choice(train_ids)
    print(f"\n--- Phân tích mẫu CSI: {sample_id_to_analyze} ---")

    # Tải dữ liệu CSI thô
    csi_sample = load_csi_sample(TRAIN_DATA_PATH, sample_id_to_analyze)
    
    # Lấy biên độ (amplitude) từ dữ liệu CSI phức
    amplitude_sample = np.abs(csi_sample)
    num_tx, num_rx, num_subcarriers, num_packets = amplitude_sample.shape
    
    print(f"\n--- Trực quan hóa toàn bộ {num_tx*num_rx*num_subcarriers} luồng tín hiệu biên độ thô ---")

    # Ghép tất cả các chiều (Tx, Rx, Subcarrier) lại thành một chiều duy nhất.
    all_streams_amplitude = amplitude_sample.reshape(-1, num_packets)

    # 1. Trực quan hóa dữ liệu thô bằng Heatmap
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle(f"Heatmap của {all_streams_amplitude.shape[0]} Luồng Tín hiệu Thô - Mẫu: {sample_id_to_analyze}", fontsize=16)
    im = ax.imshow(all_streams_amplitude, aspect='auto', cmap='jet', origin='lower')
    ax.set_title("Luồng tín hiệu thay đổi theo thời gian")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Đặc trưng")
    fig.colorbar(im, ax=ax, label="Biên độ")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    ################################################################################
    ### PHẦN 3: XỬ LÝ DỮ LIỆU BẰNG PCA
    ################################################################################
    
    print("\n--- Áp dụng Phân tích Thành phần chính (PCA) để trích xuất đặc trưng ---")
    
    # Chuẩn bị dữ liệu cho PCA: (n_samples, n_features) -> (packets, streams)
    features = all_streams_amplitude.T
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Áp dụng PCA để trích xuất 3 thành phần chính
    n_components = 3
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_scaled)
    
    print(f"Đã trích xuất thành công {principal_components.shape[1]} thành phần chính.")

    ################################################################################
    ### PHẦN 4: TRỰC QUAN HÓA KẾT QUẢ SAU KHI XỬ LÝ
    ################################################################################
    
    print("\n--- Trực quan hóa kết quả sau khi xử lý PCA ---")
    
    # Tạo một cửa sổ chứa 2 đồ thị con (1 hàng, 2 cột)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f"Kết quả Phân tích PCA - Mẫu: {sample_id_to_analyze}", fontsize=16)

    # --- ĐỒ THỊ 1: SPECTRUM (Bên trái) ---
    spectrum_data = principal_components.T
    im = ax1.imshow(spectrum_data, aspect='auto', cmap='jet', origin='lower', 
                    extent=[0, num_packets, 0.5, n_components + 0.5])
    ax1.set_title("Spectrum (Các thành phần chính theo thời gian)")
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Principal Components")
    ax1.set_yticks(np.arange(1, n_components + 1))
    fig.colorbar(im, ax=ax1, label="Component Score")

    # --- ĐỒ THỊ 2: SIGNAL (Bên phải) ---
    # Vẽ cả 3 thành phần chính đã được trích xuất
    ax2.plot(principal_components[:, 0], label='PC 1 (Tín hiệu chính)')
    ax2.plot(principal_components[:, 1], label='PC 2')
    ax2.plot(principal_components[:, 2], label='PC 3')
    ax2.set_title("Signal (Các tín hiệu đã được làm sạch)")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Amplitude (Component Score)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()