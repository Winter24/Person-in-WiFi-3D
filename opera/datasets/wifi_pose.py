import json
import os
from scipy import io
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import pywt
from collections import OrderedDict
import scipy.fft as fft
from .builder import DATASETS
from mmdet.datasets.pipelines import Compose
import h5py

@DATASETS.register_module()
class WifiPoseDataset(dataset):
    CLASSES = ('person', )
    def __init__(self, dataset_root, pipeline, mode, limit_samples=None, **kwargs):
        
        self.data_root = dataset_root
        self.pipeline = Compose(pipeline)
        self.filename_list = self.load_file_name_list(os.path.join(self.data_root, mode + '_data_list.txt'))
        if limit_samples is not None and limit_samples > 0:
            print(f"\n!!! CHẾ ĐỘ DEBUG: Chỉ tải {limit_samples} mẫu dữ liệu cho mode='{mode}'.\n")
            self.filename_list = self.filename_list[:limit_samples]
        self._set_group_flag()
        self.JOINT_NAMES = [
            'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow', 
            'R_Hip', 'L_Wrist', 'R_Wrist', 'R_Knee', 'R_Ankle', 
            'L_Knee', 'L_Hip', 'L_Ankle'
        ]

        self.TARGET_BONES = [
            (3, 2),    # LShoulder <-> RShoulder
            (12, 6),   # LHip <-> RHip
            (3, 5),    # LShoulder -> LElbow (Tay Trái)
            (5, 7),    # LElbow -> LWrist
            (2, 4),    # RShoulder -> RElbow (Tay Phải)
            (4, 8),    # RElbow -> RWrist
            (12, 11),  # LHip -> LKnee (Chân Trái)
            (11, 13),  # LKnee -> LAnkle
            (6, 9),    # RHip -> RKnee (Chân Phải)
            (9, 10),   # RKnee -> RAnkle
        ]
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir

    def get_item_single_frame(self,index): 
        data_name = self.filename_list[index]
        csi_path = os.path.join(self.data_root,'csi',(str(data_name)+'.mat'))
        keypoint_path = os.path.join(self.data_root,'keypoint',(str(data_name)+'.npy'))
        
        '''csi =  io.loadmat(csi_path)['csi_out']
        csi = np.array(csi)
        csi = csi.astype(np.complex128)'''
        
        # csi = h5py.File(csi_path)['csi_out'].value
        with h5py.File(csi_path, 'r') as f:
            csi = f['csi_out'][()]   # Change
            
        csi = csi['real'] + csi['imag']*1j
        csi = np.array(csi).transpose(3,2,1,0)
        csi = csi.astype(np.complex128)
        
        '''csi_amp = abs(csi)
        csi_amp = torch.FloatTensor(csi_amp).permute(0,1,3,2) #csi tensor: (3*3*30*20 -> 3*3*20*30)
        
        csi_ph = np.unwrap(np.angle(csi))
        csi_ph = fft.ifft(csi_ph)
        csi_phd = csi_ph[:,:,:,1:20] - csi_ph[:,:,:,0:19]
        csi_phd = torch.FloatTensor(csi_phd).permute(0,1,3,2)'''
        
        #-------------------
        csi_amp = self.dwt_amp(csi)
        csi_ph = self.phase_deno(csi)
        csi_ph = np.angle(csi_ph)
        #csi = np.concatenate((csi_amp, csi_ph), axis=2)
        csi = np.concatenate((csi_amp, csi_ph), axis=2)
        #csi = torch.FloatTensor(csi)
        
        '''csi_amp = self.dwt_amp(csi)
        csi = torch.FloatTensor(csi_amp)'''
        
        #csi = torch.cat((csi_amp, csi_ph), 2)
        csi = torch.FloatTensor(csi).permute(0,1,3,2)
        

        keypoint = np.array(np.load(keypoint_path))
        #keypoint = self.keypoint_process(keypoint)
        keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*14*3)

        numOfPerson = keypoint.shape[0]
        gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)
        gt_bboxes = torch.tensor([])
        gt_areas = torch.tensor([])
        result = dict(img=csi, gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes, gt_areas = gt_areas, img_name = data_name)
        return result
    
    def get_item_single_frame_limit(self,index): 
        data_name = self.filename_list[index]
        csi_path = os.path.join(self.data_root,'csi',(str(data_name)+'.mat'))
        keypoint_path = os.path.join(self.data_root,'keypoint',(str(data_name)+'.npy'))
        
        csi =  io.loadmat(csi_path)['csi_out']
        csi = np.array(csi)
        csi = csi.astype(np.complex128)
        
        '''csi_amp = abs(csi)
        csi_amp = torch.FloatTensor(csi_amp).permute(0,1,3,2) #csi tensor: (3*3*30*20 -> 3*3*20*30)
        
        csi_ph = np.unwrap(np.angle(csi))
        csi_ph = fft.ifft(csi_ph)
        csi_phd = csi_ph[:,:,:,1:20] - csi_ph[:,:,:,0:19]
        csi_phd = torch.FloatTensor(csi_phd).permute(0,1,3,2)'''
        
        
        csi_amp = self.dwt_amp(csi)
        csi_ph = self.phase_deno(csi)
        csi_ph = np.angle(csi_ph)
        #csi = np.concatenate((csi_amp, csi_ph), axis=2)
        #csi = torch.cat((csi_amp, csi_ph), 2)
        csi = torch.FloatTensor(csi).permute(0,1,3,2)
        #csi = np.concatenate((csi_amp, csi_ph), axis=3)
        #csi = torch.FloatTensor(csi)
        

        keypoint = np.array(np.load(keypoint_path))
        #keypoint = self.keypoint_process(keypoint)
        keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*14*3)

        numOfPerson = keypoint.shape[0]
        gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)
        gt_bboxes = torch.tensor([])
        gt_areas = torch.tensor([])
        result = dict(img=csi, gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes, gt_areas = gt_areas )
        return result
    
    def __getitem__(self, index):
        result = self.get_item_single_frame(index)
        return self.pipeline(result)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                file_name_list.append(lines.split()[0])
        return file_name_list

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    def CSI_sanitization(self, csi_rx):
        one_csi = csi_rx[0,:,:]
        two_csi = csi_rx[1,:,:]
        three_csi = csi_rx[2,:,:]
        pi = np.pi
        M = 3  # 天线数量3
        N = 30  # 子载波数目30
        T = one_csi.shape[1]  # 总包数
        fi = 312.5 * 2  # 子载波间隔312.5 * 2
        csi_phase = np.zeros((M, N, T))
        for t in range(T):  # 遍历时间戳上的CSI包，每根天线上都有30个子载波
            csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
            csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
            csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
            ai = np.tile(2 * pi * fi * np.array(range(N)), M)
            bi = np.ones(M * N)
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
            A = np.dot(ai, ai)
            B = np.dot(ai, bi)
            C = np.dot(bi, bi)
            D = np.dot(ai, ci)
            E = np.dot(bi, ci)
            rho_opt = (B * E - C * D) / (A * C - B ** 2)
            beta_opt = (B * D - A * E) / (A * C - B ** 2)
            temp = np.tile(np.array(range(N)), M).reshape(M, N)
            csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
        antennaPair_One = abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
        antennaPair_Two = abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
        antennaPair_Three = abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
        antennaPair = np.concatenate((np.expand_dims(antennaPair_One,axis=0), 
                                      np.expand_dims(antennaPair_Two,axis=0), 
                                      np.expand_dims(antennaPair_Three,axis=0),))
        return antennaPair


    def phase_deno(self, csi):
        #input csi shape (3*3*30*20)
        ph_rx1 = self.CSI_sanitization(csi[0,:,:,:])
        ph_rx2 = self.CSI_sanitization(csi[1,:,:,:])
        ph_rx3 = self.CSI_sanitization(csi[2,:,:,:])
        csi_phde = np.concatenate((np.expand_dims(ph_rx1,axis=0), 
                                   np.expand_dims(ph_rx2,axis=0), 
                                   np.expand_dims(ph_rx3,axis=0),))
        #csi_phde = csi_phde.transpose(0,1,3,2)
        return csi_phde
    
    def dwt_amp(self, csi):
        #csi = csi.transpose(0,1,3,2)
        #cA, cD = pywt.dwt(abs(csi), 'db11')
        #csi_amp = np.concatenate((cA, cD), axis=2)
        #csi_amp = np.concatenate((cA, cD), axis=3)
        w = pywt.Wavelet('dB11')
        list = pywt.wavedec(abs(csi), w,'sym')
        csi_amp = pywt.waverec(list, w)
        return csi_amp
        
    def keypoint_process(self, keypoints):
        next_point = np.array([[0,1], [1,2], [2,5], [3,0], [4,2], [5,7],
                               [6,3], [7,3], [8,4], [9,5], [10,6], [11,7],
                               [12,9], [13,11]])
        keypoints_list = []
        for numofperson in range(keypoints.shape[0]):
            for numofpoint in range(keypoints.shape[1]):
                point_with_next = np.concatenate((keypoints[numofperson,next_point[numofpoint,0],:],
                                                  keypoints[numofperson,next_point[numofpoint,1],:]), axis=0)
                point_class = np.zeros((15))
                keypoints_list.append(point_with_next)
        
        return np.array(keypoints_list)
    
    # def evaluate(self,
    #              results,
    #              metric='keypoints',
    #              logger=None,
    #              jsonfile_prefix=None,
    #              classwise=False,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=None,
    #              metric_items=None):
    #     mpjpe_3d_list = []
    #     mpjpe_h_list = []
    #     mpjpe_v_list = []
    #     mpjpe_d_list = []
    #     for i in range(len(results)):
    #         info = self.get_item_single_frame(i)
    #         gt_keypoints = info['gt_keypoints']
    #         data_name = info['img_name']
    #         det_bboxes, det_keypoints = results[i]
    #         for label in range(len(det_keypoints)):
    #             kpt_pred = det_keypoints[label]
    #             kpt_pred = torch.tensor(kpt_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
    #             #np.save('/home/yankangwei/opera-main/result/pose_o/%s.npy' %data_name, kpt_pred)
    #             mpjpe_3d,mpjpeh,mpjpev,mpjped = self.calc_mpjpe(gt_keypoints, kpt_pred, data_name, root = [5,7])
    #             mpjpe_3d_list.append(mpjpe_3d.numpy())
    #             mpjpe_h_list.append(mpjpeh.numpy())
    #             mpjpe_v_list.append(mpjpev.numpy())
    #             mpjpe_d_list.append(mpjped.numpy())
    #             #mpjpe_3d_list.append(np.array([0]))

    #     mpjpe = np.array(mpjpe_3d_list).mean()   
    #     mpjpeh = np.array(mpjpe_h_list).mean() 
    #     mpjpev = np.array(mpjpe_v_list).mean() 
    #     mpjped = np.array(mpjpe_d_list).mean() 
    #     result = {'mpjpe':mpjpe, 'mpjpeh':mpjpeh, 'mpjpev':mpjpev, 'mpjped':mpjped}
    #     return OrderedDict(result)
    
    def evaluate(self,
                results,
                metric='keypoints',
                logger=None,
                jsonfile_prefix=None,
                classwise=False,
                proposal_nums=(100, 300, 1000),
                iou_thrs=None,
                metric_items=None):
        
        # --- PHẦN KHỞI TẠO CÁC LIST ĐỂ LƯU KẾT QUẢ ---
        all_mpjpe_metrics = [] # MPJPE, PJDLE_h, PJDLE_v, PJDLE_d cho từng mẫu
        all_per_joint_mpjpe = [] # MPJPE cho từng khớp, shape (num_samples, 14)
        all_bone_length_errors = [] # Sai số chiều dài xương, shape (num_samples, num_bones)

        # Tải chiều dài xương ground-truth từ file JSON
        try:
            with open('gt_bone_stats.json', 'r') as f:
                bone_stats = json.load(f)
            gt_bone_lengths_mean = torch.tensor(bone_stats['mean'])
            bones_definition_from_json = bone_stats['bones_definition']
        except FileNotFoundError:
            print("Lỗi: Không tìm thấy file 'gt_bone_stats.json'. Vui lòng chạy script phân tích trước.")
            return {}

        # Tạo mapping từ TARGET_BONES tới chỉ số trong gt_bone_lengths_mean
        json_indices = []
        for target_bone in self.TARGET_BONES:
            for i, full_bone in enumerate(bones_definition_from_json):
                if (target_bone[0] == full_bone[0] and target_bone[1] == full_bone[1]) or \
                (target_bone[0] == full_bone[1] and target_bone[1] == full_bone[0]):
                    json_indices.append(i)
                    break
        reliable_gt_lengths_mean = gt_bone_lengths_mean[json_indices]
        
        # --- VÒNG LẶP XỬ LÝ KẾT QUẢ ---
        for i in range(len(results)):
            info = self.get_item_single_frame(i)
            gt_keypoints = info['gt_keypoints']
            
            # Bỏ qua nếu không có ground-truth
            if gt_keypoints.shape[0] == 0:
                continue
                
            det_bboxes, det_keypoints = results[i]
            
            # Chỉ xử lý lớp 'person' (label 0)
            kpt_pred = det_keypoints[0]
            
            # Bỏ qua nếu không có dự đoán nào
            if kpt_pred.shape[0] == 0:
                # Nếu có ground-truth nhưng không có dự đoán, coi như sai số là rất lớn
                # Hoặc đơn giản là bỏ qua để tính trên các mẫu có dự đoán
                continue
                
            kpt_pred = torch.tensor(kpt_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
            
            # --- TÍNH TOÁN CÁC METRIC ---
            # 1. Khớp cặp và tính toán MPJPE, PJDLE
            matched_results = self.calc_mpjpe_and_match(gt_keypoints, kpt_pred)
            
            if matched_results:
                # Nếu có ít nhất một cặp được khớp
                mpjpe_metrics, per_joint_mpjpe, matched_pred_kpts, matched_gt_kpts = matched_results
                all_mpjpe_metrics.append(mpjpe_metrics)
                all_per_joint_mpjpe.append(per_joint_mpjpe)
                
                # 2. Tính toán sai số chiều dài xương trên các cặp đã khớp
                bone_error = self.calc_bone_length_error(matched_pred_kpts, reliable_gt_lengths_mean)
                all_bone_length_errors.append(bone_error)

        # --- TỔNG HỢP VÀ IN KẾT QUẢ ---
        if not all_mpjpe_metrics:
            print("Không có mẫu nào được đánh giá (có thể do không có dự đoán hoặc ground-truth).")
            return {}

        # Tính trung bình các metric MPJPE/PJDLE
        avg_mpjpe_metrics = np.mean(all_mpjpe_metrics, axis=0)
        
        # Tính trung bình MPJPE trên từng khớp
        avg_per_joint_mpjpe = np.mean(all_per_joint_mpjpe, axis=0)
        
        # Tính trung bình sai số chiều dài xương
        avg_bone_length_error = np.mean(all_bone_length_errors, axis=0)
        
        # Tạo dictionary kết quả
        result_dict = OrderedDict(
            mpjpe=avg_mpjpe_metrics[0],
            mpjpeh=avg_mpjpe_metrics[1],
            mpjpev=avg_mpjpe_metrics[2],
            mpjped=avg_mpjpe_metrics[3]
        )

        # In ra bảng kết quả chi tiết
        print("\n" + "="*60)
        print(" " * 15 + "BÁO CÁO ĐÁNH GIÁ CHI TIẾT")
        print("="*60)
        print(f"MPJPE Tổng thể: {result_dict['mpjpe']:.2f} mm")
        print(f"PJDLE (ngang):   {result_dict['mpjpeh']:.2f} mm")
        print(f"PJDLE (sâu):     {result_dict['mpjpev']:.2f} mm")
        print(f"PJDLE (cao):     {result_dict['mpjped']:.2f} mm")
        print("-"*60)
        print(" " * 15 + "MPJPE TRUNG BÌNH TRÊN TỪNG KHỚP (mm)")
        print("-"*60)
        # Sắp xếp để dễ xem
        sorted_joint_errors = sorted(zip(self.JOINT_NAMES, avg_per_joint_mpjpe), key=lambda item: item[1], reverse=True)
        for joint_name, error in sorted_joint_errors:
            print(f"{joint_name:<15} | {error:.2f}")
        print("-"*60)
        print(" " * 10 + "SAI SỐ CHIỀU DÀI XƯƠNG TRUNG BÌNH (mm)")
        print("-"*60)
        bone_names = [f"{self.JOINT_NAMES[b[0]]}-{self.JOINT_NAMES[b[1]]}" for b in self.TARGET_BONES]
        sorted_bone_errors = sorted(zip(bone_names, avg_bone_length_error), key=lambda item: item[1], reverse=True)
        for bone_name, error in sorted_bone_errors:
            print(f"{bone_name:<25} | {error:.2f}")
        print("="*60)
        
        return result_dict

    # THÊM 2 HÀM HELPER NÀY VÀO BÊN TRONG CLASS WifiPoseDataset
    def calc_bone_length_error(self, pred_kpts, gt_lengths_mean):
        """Tính sai số L1 trung bình của chiều dài xương."""
        if pred_kpts.numel() == 0:
            return []
        
        bones_tensor = torch.tensor(self.TARGET_BONES, dtype=torch.long, device=pred_kpts.device)
        
        p1 = pred_kpts[:, bones_tensor[:, 0], :3]
        p2 = pred_kpts[:, bones_tensor[:, 1], :3]
        
        pred_lengths = torch.norm(p1 - p2, p=2, dim=-1) # shape: (num_matched, num_bones)
        
        target_lengths = gt_lengths_mean.to(pred_kpts.device).expand_as(pred_lengths)
        
        error = torch.abs(pred_lengths - target_lengths) * 1000 # Chuyển sang mm
        
        return error.mean(dim=0).cpu().numpy() # Trả về sai số trung bình cho từng xương

    def calc_mpjpe_and_match(self, gt_kpts, pred_kpts):
        """Khớp cặp GT và Pred, sau đó tính các loại MPJPE."""
        n_gt, n_pred = gt_kpts.shape[0], pred_kpts.shape[0]
        
        # Tạo ma trận chi phí
        cost_matrix = torch.cdist(gt_kpts.view(n_gt, -1), pred_kpts.view(n_pred, -1), p=2)
        cost_matrix = cost_matrix.cpu().numpy()
        
        # Dùng thuật toán Hungary để khớp cặp
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        # Lấy ra các cặp đã được khớp
        matched_gt = gt_kpts[gt_indices]
        matched_pred = pred_kpts[pred_indices]
        
        if matched_gt.numel() == 0:
            return None

        # Tính toán các metric trên các cặp đã khớp
        # Sai số Euclid trên từng khớp
        per_joint_error_3d = torch.norm(matched_gt - matched_pred, p=2, dim=-1) # shape: (num_matched, 14)
        
        # MPJPE tổng thể
        mpjpe = per_joint_error_3d.mean() * 1000 # mm
        
        # PJDLE
        per_joint_error_dim = torch.abs(matched_gt - matched_pred) # shape: (num_matched, 14, 3)
        mpjpeh = per_joint_error_dim[..., 0].mean() * 1000 # mm
        mpjpev = per_joint_error_dim[..., 1].mean() * 1000 # mm (đổi Y và Z để khớp với paper)
        mpjped = per_joint_error_dim[..., 2].mean() * 1000 # mm
        
        mpjpe_metrics = [mpjpe.cpu().numpy(), mpjpeh.cpu().numpy(), mpjpev.cpu().numpy(), mpjped.cpu().numpy()]
        per_joint_mpjpe = per_joint_error_3d.mean(dim=0).cpu().numpy() * 1000 # mm, shape (14,)
        
        return mpjpe_metrics, per_joint_mpjpe, matched_pred, matched_gt
    
    def calc_mpjpe(self, real, pred, no, root=0):
        n = real.shape[0]
        m = pred.shape[0]
        j, c = pred.shape[1:]
        assert j == real.shape[1] and c == real.shape[2]
        if isinstance(root,list):
            real_root = real.unsqueeze(1).expand(n, m, j, c)
            pred_root = pred.unsqueeze(0).expand(n, m, j, c)
            #n*m*j  n*j
            distance_array = torch.ones((n,m), dtype=torch.float) * 2 ** 24  # TODO: magic number!
            for i in range(n):
                for j in range(m):
                    distance_array[i][j] = torch.norm(real[i]-pred[j], p=2, dim=-1).mean()

            # distance_array = torch.norm(real_root-pred_root, p=2, dim=-1)*vis_mask.unsqueeze(1).expand(n, m, j)
            # distance_array = distance_array.sum(-1) / vis_mask.sum(-1).unsqueeze(1)
            # print(torch.min(distance_array))
        else:
            real_root = real[:, root].unsqueeze(0).expand(n, m, c)
            pred_root = pred[:, root].unsqueeze(1).expand(n, m, c)
            distance_array = torch.pow(real_root - pred_root, 2)
        corres = torch.ones(n, dtype=torch.long)*-1
        occupied = torch.zeros(m, dtype=torch.long)

        while torch.min(distance_array) < 50:   # threshold 30.
            min_idx = torch.where(distance_array == torch.min(distance_array))
            
            for i in range(len(min_idx[0])):
                distance_array[min_idx[0][i]][min_idx[1][i]] = 50
                if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
                    continue
                else:
                    corres[min_idx[0][i]] = min_idx[1][i]
                    occupied[min_idx[1][i]] = 1
        new_pred = pred[corres]
        #np.save('/home/yankangwei/opera-main/result/pose_pred/%s.npy' %no, new_pred)
        #np.save('/home/yankangwei/opera-main/result/pose_gt/%s.npy' %no, real)
        mpjpe = torch.sqrt(torch.pow(real - new_pred, 2).sum(-1))
        mpjpeh = torch.sqrt(torch.pow(real[:,:,0] - new_pred[:,:,0], 2))
        mpjpev = torch.sqrt(torch.pow(real[:,:,1] - new_pred[:,:,1], 2))
        mpjped = torch.sqrt(torch.pow(real[:,:,2] - new_pred[:,:,2], 2))
        # mpjpe = torch.norm(real-new_pred, p=2, dim=-1) #n*j
        # mpjpe_mean = (mpjpe*vis_mask.float()).sum(-1)/vis_mask.float().sum(-1) if vis_mask is not None else mpjpe.mean(-1)
        return mpjpe.mean()*1000, mpjpeh.mean()*1000, mpjpev.mean()*1000, mpjped.mean()*1000
# if __name__ == "__main__":
#     path = 'data/'
#     train_ds = Train_Dataset(path)
#     train_dl = DataLoader(train_ds, 1, False, num_workers=1)
