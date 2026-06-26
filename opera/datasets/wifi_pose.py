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
from tools.analysis.per_sample_pose_metrics import build_sample_record, write_jsonl
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
def _json_serializer(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
        img_shape = tuple(csi.shape)
        result = dict(
            img=csi,
            gt_keypoints=keypoint,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            gt_areas=gt_areas,
            img_name=data_name,
            img_shape=img_shape,
            ori_shape=img_shape,
            pad_shape=img_shape)
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
        img_shape = tuple(csi.shape)
        result = dict(
            img=csi,
            gt_keypoints=keypoint,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            gt_areas=gt_areas,
            img_name=data_name,
            img_shape=img_shape,
            ori_shape=img_shape,
            pad_shape=img_shape)
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
                metric_items=None,
                metrics_out=None,
                per_sample_metrics_out=None,
                miss_penalty_mm=500.0,
                match_threshold_mm=500.0):
        """Evaluate pose predictions with GT-person-weighted MPJPE.

        Missed GT persons receive ``miss_penalty_mm`` so empty or partial
        detections are reflected in the reported MPJPE instead of being
        skipped. False positives are not included in MPJPE, but matched/missed
        counts are reported so detection coverage is visible.
        """
        metric_sums = np.zeros(4, dtype=np.float64)
        matched_metric_sums = np.zeros(4, dtype=np.float64)  # matched-only, no miss penalty
        per_joint_sum = np.zeros(len(self.JOINT_NAMES), dtype=np.float64)
        per_joint_mpjdle_sum = np.zeros((len(self.JOINT_NAMES), 3), dtype=np.float64)
        all_bone_length_errors = []

        total_gt_persons = 0
        total_predicted_persons = 0
        total_matched_persons = 0
        total_missed_persons = 0
        total_false_positive_persons = 0

        bucket_metric_sums = {
            1: np.zeros(4, dtype=np.float64),
            2: np.zeros(4, dtype=np.float64),
            3: np.zeros(4, dtype=np.float64),
        }
        bucket_gt_persons = {1: 0, 2: 0, 3: 0}
        bucket_gt_count = {1: 0, 2: 0, 3: 0}
        bucket_matched_frames = {1: 0, 2: 0, 3: 0}
        bucket_missed_persons = {1: 0, 2: 0, 3: 0}
        bucket_false_positive_persons = {1: 0, 2: 0, 3: 0}
        per_sample_records = []

        try:
            bone_stats_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'gt_bone_stats.json')
            with open(bone_stats_path, 'r') as f:
                bone_stats = json.load(f)
            gt_bone_lengths_mean = torch.tensor(bone_stats['mean'])
            bones_definition_from_json = bone_stats['bones_definition']
        except FileNotFoundError:
            print("Warning: 'gt_bone_stats.json' not found. Bone length errors will be skipped.")
            gt_bone_lengths_mean = None
            bones_definition_from_json = None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"Warning: Failed to parse 'gt_bone_stats.json' ({exc}). Bone length errors will be skipped.")
            gt_bone_lengths_mean = None
            bones_definition_from_json = None

        reliable_gt_lengths_mean = None
        if gt_bone_lengths_mean is not None:
            json_indices = []
            for target_bone in self.TARGET_BONES:
                for i, full_bone in enumerate(bones_definition_from_json):
                    if (target_bone[0] == full_bone[0] and target_bone[1] == full_bone[1]) or \
                    (target_bone[0] == full_bone[1] and target_bone[1] == full_bone[0]):
                        json_indices.append(i)
                        break
            reliable_gt_lengths_mean = gt_bone_lengths_mean[json_indices]

        for i in range(len(results)):
            info = self.get_item_single_frame(i)
            gt_keypoints = info['gt_keypoints']

            if gt_keypoints.shape[0] == 0:
                continue

            n_gt_persons = int(gt_keypoints.shape[0])
            total_gt_persons += n_gt_persons

            if n_gt_persons in bucket_gt_count:
                bucket_gt_count[n_gt_persons] += 1
                bucket_gt_persons[n_gt_persons] += n_gt_persons

            det_bboxes, det_keypoints = results[i]
            kpt_pred_np = det_keypoints[0] if len(det_keypoints) else np.zeros(
                (0, gt_keypoints.shape[1], gt_keypoints.shape[2]),
                dtype=np.float32)
            n_pred_persons = int(kpt_pred_np.shape[0])
            total_predicted_persons += n_pred_persons

            matched_results = None
            if n_pred_persons > 0:
                kpt_pred = torch.tensor(
                    kpt_pred_np, dtype=gt_keypoints.dtype,
                    device=gt_keypoints.device)
                matched_results = self.calc_mpjpe_and_match(
                    gt_keypoints, kpt_pred,
                    match_threshold_mm=match_threshold_mm)

            matched_count = 0
            if matched_results:
                mpjpe_metrics, per_joint_mpjpe, per_joint_mpjdle, matched_pred_kpts, matched_gt_kpts = matched_results
                matched_count = int(matched_gt_kpts.shape[0])
                metric_values = np.asarray(
                    [float(np.asarray(value)) for value in mpjpe_metrics],
                    dtype=np.float64)
                per_joint_values = np.asarray(per_joint_mpjpe, dtype=np.float64)
                per_joint_mpjdle_values = np.asarray(per_joint_mpjdle, dtype=np.float64)

                metric_sums += metric_values * matched_count
                matched_metric_sums += metric_values * matched_count
                per_joint_sum += per_joint_values * matched_count
                per_joint_mpjdle_sum += per_joint_mpjdle_values * matched_count
                total_matched_persons += matched_count

                if n_gt_persons in bucket_metric_sums:
                    bucket_metric_sums[n_gt_persons] += metric_values * matched_count
                    bucket_matched_frames[n_gt_persons] += 1

                if reliable_gt_lengths_mean is not None:
                    bone_error = self.calc_bone_length_error(
                        matched_pred_kpts, reliable_gt_lengths_mean)
                    all_bone_length_errors.append(bone_error)

            missed_count = n_gt_persons - matched_count
            if missed_count > 0:
                penalty_values = np.full(4, miss_penalty_mm, dtype=np.float64)
                metric_sums += penalty_values * missed_count
                per_joint_sum += miss_penalty_mm * missed_count
                per_joint_mpjdle_sum += miss_penalty_mm * missed_count
                total_missed_persons += missed_count
                if n_gt_persons in bucket_metric_sums:
                    bucket_metric_sums[n_gt_persons] += penalty_values * missed_count
                    bucket_missed_persons[n_gt_persons] += missed_count

            false_positive_count = max(0, n_pred_persons - matched_count)
            if false_positive_count > 0:
                total_false_positive_persons += false_positive_count
                if n_gt_persons in bucket_false_positive_persons:
                    bucket_false_positive_persons[n_gt_persons] += false_positive_count

            if per_sample_metrics_out:
                per_sample_records.append(build_sample_record(
                    sample_index=i,
                    sample_id=info['img_name'],
                    gt_keypoints=gt_keypoints.detach().cpu().numpy(),
                    pred_keypoints=kpt_pred_np,
                    confidences=self._extract_prediction_confidences(
                        det_bboxes, n_pred_persons),
                    match_threshold_mm=match_threshold_mm,
                    miss_penalty_mm=miss_penalty_mm))

        if total_gt_persons == 0:
            print("No ground-truth samples were available for evaluation.")
            return {}

        avg_mpjpe_metrics = metric_sums / float(total_gt_persons)
        avg_per_joint_mpjpe = per_joint_sum / float(total_gt_persons)
        avg_per_joint_mpjdle = per_joint_mpjdle_sum / float(total_gt_persons)
        if total_matched_persons > 0:
            avg_matched_metrics = matched_metric_sums / float(total_matched_persons)
        else:
            avg_matched_metrics = np.full(4, float('nan'), dtype=np.float64)

        if all_bone_length_errors:
            avg_bone_length_error = np.mean(all_bone_length_errors, axis=0)
        else:
            avg_bone_length_error = np.zeros(len(self.TARGET_BONES))

        mpjpe_1p = (float(bucket_metric_sums[1][0] / bucket_gt_persons[1])
                    if bucket_gt_persons[1] else float('nan'))
        mpjpe_2p = (float(bucket_metric_sums[2][0] / bucket_gt_persons[2])
                    if bucket_gt_persons[2] else float('nan'))
        mpjpe_3p = (float(bucket_metric_sums[3][0] / bucket_gt_persons[3])
                    if bucket_gt_persons[3] else float('nan'))

        count_1p = bucket_gt_count[1]
        count_2p = bucket_gt_count[2]
        count_3p = bucket_gt_count[3]
        matched_1p = bucket_matched_frames[1]
        matched_2p = bucket_matched_frames[2]
        matched_3p = bucket_matched_frames[3]

        result_dict = OrderedDict(
            mpjpe=float(avg_mpjpe_metrics[0]),
            mpjpeh=float(avg_mpjpe_metrics[1]),
            mpjpev=float(avg_mpjpe_metrics[2]),
            mpjped=float(avg_mpjpe_metrics[3]),
            matched_mpjpe=float(avg_matched_metrics[0]),
            matched_mpjpeh=float(avg_matched_metrics[1]),
            matched_mpjpev=float(avg_matched_metrics[2]),
            matched_mpjped=float(avg_matched_metrics[3]),
            mpjpe_1p=mpjpe_1p,
            mpjpe_2p=mpjpe_2p,
            mpjpe_3p=mpjpe_3p,
            count_1p=count_1p,
            count_2p=count_2p,
            count_3p=count_3p,
            matched_1p=matched_1p,
            matched_2p=matched_2p,
            matched_3p=matched_3p,
            total_gt_persons=total_gt_persons,
            predicted_persons=total_predicted_persons,
            matched_persons=total_matched_persons,
            missed_persons=total_missed_persons,
            false_positive_persons=total_false_positive_persons,
            missed_1p=bucket_missed_persons[1],
            missed_2p=bucket_missed_persons[2],
            missed_3p=bucket_missed_persons[3],
            false_positive_1p=bucket_false_positive_persons[1],
            false_positive_2p=bucket_false_positive_persons[2],
            false_positive_3p=bucket_false_positive_persons[3],
            miss_penalty_mm=float(miss_penalty_mm),
            match_threshold_mm=float(match_threshold_mm),
        )

        print("\n" + "="*60)
        print(" " * 15 + "EVALUATION REPORT")
        print("="*60)
        print(f"MPJPE Overall:   {result_dict['mpjpe']:.2f} mm")
        print(f"PJDLE (horiz):   {result_dict['mpjpeh']:.2f} mm")
        print(f"PJDLE (depth):   {result_dict['mpjpev']:.2f} mm")
        print(f"PJDLE (vert):    {result_dict['mpjped']:.2f} mm")
        print("-"*60)
        print(" " * 10 + "MPJPE BY NUMBER OF PERSONS")
        print("-"*60)
        print(f"  1-person:  {mpjpe_1p:.2f} mm  (n={count_1p}, matched={matched_1p})")
        print(f"  2-person:  {mpjpe_2p:.2f} mm  (n={count_2p}, matched={matched_2p})")
        print(f"  3-person:  {mpjpe_3p:.2f} mm  (n={count_3p}, matched={matched_3p})")
        print(f"  matched persons: {total_matched_persons} / {total_gt_persons}")
        print(f"  missed persons:  {total_missed_persons} "
              f"(penalty={miss_penalty_mm:.1f} mm)")
        print(f"  match threshold: {match_threshold_mm:.1f} mm")
        print(f"  false positives: {total_false_positive_persons}")
        print("-"*60)
        print(" " * 10 + "PER-JOINT MPJPE (mm)")
        print("-"*60)
        sorted_joint_errors = sorted(
            zip(self.JOINT_NAMES, avg_per_joint_mpjpe),
            key=lambda item: item[1], reverse=True)
        for joint_name, error in sorted_joint_errors:
            print(f"{joint_name:<15} | {error:.2f}")
        if all_bone_length_errors:
            print("-"*60)
            print(" " * 10 + "BONE LENGTH ERROR (mm)")
            print("-"*60)
            bone_names = [f"{self.JOINT_NAMES[b[0]]}-{self.JOINT_NAMES[b[1]]}" for b in self.TARGET_BONES]
            sorted_bone_errors = sorted(
                zip(bone_names, avg_bone_length_error),
                key=lambda item: item[1], reverse=True)
            for bone_name, error in sorted_bone_errors:
                print(f"{bone_name:<25} | {error:.2f}")
        print("="*60)

        if metrics_out:
            per_joint_dict = {name: float(err) for name, err in zip(self.JOINT_NAMES, avg_per_joint_mpjpe)}
            bone_names = [f"{self.JOINT_NAMES[b[0]]}-{self.JOINT_NAMES[b[1]]}" for b in self.TARGET_BONES]
            bone_dict = {name: float(err) for name, err in zip(bone_names, avg_bone_length_error)}
            per_joint_mpjdle_dict = {
                name: {
                    'h': float(avg_per_joint_mpjdle[i, 0]),
                    'v': float(avg_per_joint_mpjdle[i, 1]),
                    'd': float(avg_per_joint_mpjdle[i, 2]),
                }
                for i, name in enumerate(self.JOINT_NAMES)
            }
            export = OrderedDict(result_dict)
            export['per_joint_mpjpe'] = per_joint_dict
            export['per_joint_mpjdle'] = per_joint_mpjdle_dict
            export['bone_length_error'] = bone_dict
            os.makedirs(os.path.dirname(os.path.abspath(metrics_out)), exist_ok=True)
            with open(metrics_out, 'w') as f:
                json.dump(export, f, indent=2, default=_json_serializer)
            print(f"\nMetrics exported to: {metrics_out}")

        if per_sample_metrics_out:
            write_jsonl(per_sample_records, per_sample_metrics_out)
            print(f"\nPer-sample metrics exported to: {per_sample_metrics_out}")

        return result_dict

    @staticmethod
    def _extract_prediction_confidences(det_bboxes, n_pred_persons):
        if n_pred_persons <= 0 or not det_bboxes:
            return []
        bboxes = np.asarray(det_bboxes[0])
        if bboxes.ndim != 2 or bboxes.shape[0] == 0 or bboxes.shape[1] < 5:
            return []
        return [float(value) for value in bboxes[:n_pred_persons, 4]]

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

    def calc_mpjpe_and_match(self, gt_kpts, pred_kpts,
                             match_threshold_mm=500.0):
        """Match GT and predictions, then compute matched-person MPJPE."""
        n_gt, n_pred = gt_kpts.shape[0], pred_kpts.shape[0]
        if n_gt == 0 or n_pred == 0:
            return None
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')

        cost_matrix = torch.cdist(
            gt_kpts.reshape(n_gt, -1).float(),
            pred_kpts.reshape(n_pred, -1).float(),
            p=2)
        gt_indices, pred_indices = linear_sum_assignment(
            cost_matrix.detach().cpu().numpy())
        gt_indices = torch.as_tensor(gt_indices, dtype=torch.long, device=gt_kpts.device)
        pred_indices = torch.as_tensor(pred_indices, dtype=torch.long, device=pred_kpts.device)

        pair_errors_mm = torch.norm(
            gt_kpts[gt_indices] - pred_kpts[pred_indices],
            p=2, dim=-1).mean(dim=-1) * 1000
        valid_pairs = pair_errors_mm <= match_threshold_mm
        if valid_pairs.sum().item() == 0:
            return None
        gt_indices = gt_indices[valid_pairs]
        pred_indices = pred_indices[valid_pairs]

        matched_gt = gt_kpts[gt_indices]
        matched_pred = pred_kpts[pred_indices]
        if matched_gt.numel() == 0:
            return None

        per_joint_error_3d = torch.norm(matched_gt - matched_pred, p=2, dim=-1)
        mpjpe = per_joint_error_3d.mean() * 1000

        per_joint_error_dim = torch.abs(matched_gt - matched_pred)
        mpjpeh = per_joint_error_dim[..., 0].mean() * 1000
        mpjpev = per_joint_error_dim[..., 1].mean() * 1000
        mpjped = per_joint_error_dim[..., 2].mean() * 1000

        mpjpe_metrics = [
            mpjpe.cpu().numpy(),
            mpjpeh.cpu().numpy(),
            mpjpev.cpu().numpy(),
            mpjped.cpu().numpy(),
        ]
        per_joint_mpjpe = per_joint_error_3d.mean(dim=0).cpu().numpy() * 1000
        per_joint_mpjdle = per_joint_error_dim.mean(dim=0).cpu().numpy() * 1000
        return mpjpe_metrics, per_joint_mpjpe, per_joint_mpjdle, matched_pred, matched_gt

    def calc_mpjpe(self, real, pred, no, root=0, penalty_mm=500.0):
        """Legacy metric helper returning per-frame person-summed errors.

        Unmatched GT persons receive ``penalty_mm``. This avoids the historical
        ``pred[-1]`` silent bug when a GT person has no matched prediction.
        """
        n = real.shape[0]
        m = pred.shape[0]
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0

        j, c = pred.shape[1:]
        assert j == real.shape[1] and c == real.shape[2]

        if m == 0:
            penalty = penalty_mm * n
            return penalty, penalty, penalty, penalty

        if isinstance(root, list):
            distance_array = torch.ones((n, m), dtype=torch.float, device=real.device) * 1e6
            for gt_idx in range(n):
                for pred_idx in range(m):
                    distance_array[gt_idx, pred_idx] = torch.norm(
                        real[gt_idx] - pred[pred_idx], p=2, dim=-1).mean()
        else:
            real_root = real[:, root].unsqueeze(1).expand(n, m, c)
            pred_root = pred[:, root].unsqueeze(0).expand(n, m, c)
            distance_array = torch.norm(real_root - pred_root, p=2, dim=-1)

        corres = torch.ones(n, dtype=torch.long, device=real.device) * -1
        dist_mat = distance_array.clone()
        while dist_mat.numel() > 0 and torch.min(dist_mat) < 50:
            min_idx = torch.where(dist_mat == torch.min(dist_mat))
            gt_idx = int(min_idx[0][0].item())
            pred_idx = int(min_idx[1][0].item())
            corres[gt_idx] = pred_idx
            dist_mat[gt_idx, :] = 1e6
            dist_mat[:, pred_idx] = 1e6

        sum_mpjpe = 0.0
        sum_h = 0.0
        sum_v = 0.0
        sum_d = 0.0
        for gt_idx in range(n):
            pred_idx = int(corres[gt_idx].item())
            if pred_idx < 0:
                sum_mpjpe += penalty_mm
                sum_h += penalty_mm
                sum_v += penalty_mm
                sum_d += penalty_mm
                continue

            matched_real = real[gt_idx]
            matched_pred = pred[pred_idx]
            sum_mpjpe += torch.norm(matched_real - matched_pred, p=2, dim=-1).mean().item() * 1000
            per_dim = torch.abs(matched_real - matched_pred)
            sum_h += per_dim[:, 0].mean().item() * 1000
            sum_v += per_dim[:, 1].mean().item() * 1000
            sum_d += per_dim[:, 2].mean().item() * 1000

        return sum_mpjpe, sum_h, sum_v, sum_d
# if __name__ == "__main__":
#     path = 'data/'
#     train_ds = Train_Dataset(path)
#     train_dl = DataLoader(train_ds, 1, False, num_workers=1)
