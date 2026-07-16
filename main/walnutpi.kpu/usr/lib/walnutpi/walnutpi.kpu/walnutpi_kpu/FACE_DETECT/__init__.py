import numpy as np
import cv2
from typing import List
import os
from walnutpi_kpu import get_nncase, NNCASEVersionType,KPU_BASE
import time


class Landmark:
    """人脸关键点，包含 x 和 y 坐标"""
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Landmark(x={self.x}, y={self.y})"


class FACE_DETECT_RESULT():
    """人脸检测结果，包含 bbox 和 5 个关键点"""
    x: int  # 框左上角的x
    y: int  # 框左上角的y
    w: int  # 框的宽
    h: int  # 框的高
    reliability: float  # 置信度
    left_eye: Landmark    # 左眼关键点
    right_eye: Landmark   # 右眼关键点
    nose: Landmark        # 鼻子关键点
    left_mouth: Landmark  # 左嘴角关键点
    right_mouth: Landmark # 右嘴角关键点


class FACE_DETECT(KPU_BASE):
    """人脸检测类，基于 RetinaFace 架构，支持 bbox + 5 点关键点检测"""
    
    results: List[FACE_DETECT_RESULT] = []
    
    # 支持的模型尺寸
    SUPPORTED_SIZES = (320, 640)
    
    # 默认阈值
    _DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    _DEFAULT_NMS_THRESHOLD = 0.4
    
    # AI2D 配置：人脸检测专用填充色
    ai2d_pad_color = [104, 117, 123]
    
    def __init__(self, 
                 kmodel_path: str = None,
                 anchors_path: str = None,
                 size: int = 320, 
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        初始化人脸检测器
        
        @size: 模型输入尺寸，支持 320 和 640
        @nncase_version: nncase 版本
        @kmodel_path: 模型路径，不传则根据 size 自动查找 face_detection/face_detection_{size}.kmodel
        @anchors_path: anchors 文件路径，不传则根据 size 自动查找 face_detection/prior_data_{size}.bin
        """
        if size not in self.SUPPORTED_SIZES:
            raise ValueError(f"不支持的模型尺寸: {size}，可选: {self.SUPPORTED_SIZES}")
        
        _RES_DIR = os.path.dirname(os.path.abspath(__file__))

        # 自动解析 kmodel 路径
        if kmodel_path is None:
            kmodel_path = os.path.join(_RES_DIR, f"face_detection_{size}.kmodel")
        if not os.path.exists(kmodel_path):
            raise FileNotFoundError(f"模型文件不存在: {kmodel_path}")
        
        # 自动解析 anchors 路径
        if anchors_path is None:
            anchors_path = os.path.join(_RES_DIR, f"prior_data_{size}.bin")
        
        super().__init__(kmodel_path, size, nncase_version)
        self.confidence_threshold = self._DEFAULT_CONFIDENCE_THRESHOLD
        self.nms_threshold = self._DEFAULT_NMS_THRESHOLD
        self.anchors = self._load_anchors(anchors_path)
    
    def _load_anchors(self, anchors_path: str) -> np.ndarray:
        """加载 anchors 文件"""
        if not os.path.exists(anchors_path):
            raise FileNotFoundError(f"Anchors file not found: {anchors_path}")
        
        anchors = np.fromfile(anchors_path, dtype=np.float32)
        anchors = anchors.reshape(-1, 4)
        return anchors
    
    def run(self, img, reliability_threshold=None, nms_threshold=None) -> List[FACE_DETECT_RESULT]:
        """检测图片中的人脸"""
        if reliability_threshold is None:
            reliability_threshold = self.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.nms_threshold
        return super().run(img, reliability_threshold, nms_threshold)
    
    def get_result(self) -> List[FACE_DETECT_RESULT]:
        return super().get_result()
    
    def post_process(self, reliability_threshold, nms_threshold) -> List[FACE_DETECT_RESULT]:
        """
        人脸检测后处理，基于 C++ face_detection.cc 实现
        
        模型有 9 个输出，分三个尺度：
        - Output 0-2: 位置偏移 (1, 8, H, W)  → 8通道 = 2尺度 × 4坐标
        - Output 3-5: 置信度   (1, 4, H, W)  → 4通道 = 2尺度 × 2类别
        - Output 6-8: 关键点   (1, 20, H, W) → 20通道 = 2尺度 × 10坐标(5点×2)
        """
        # 1. 获取所有输出
        outputs = [self.kpu.get_output_tensor(i).to_numpy() for i in range(9)]
        
        # 2. 三个尺度的特征图尺寸（根据模型输入尺寸动态计算）
        # 320: [40, 20, 10] -> 1600+400+100 = 2100 locations * 2 anchors = 4200
        # 640: [80, 40, 20] -> 6400+1600+400 = 8400 locations * 2 anchors = 16800
        scale_factor = self.model_w // 320
        feature_sizes = [s * scale_factor for s in [40, 20, 10]]
        
        all_boxes = []
        all_scores = []
        all_landmarks = []
        
        anchor_offset = 0
        
        for scale_idx, feat_size in enumerate(feature_sizes):
            H = W = feat_size
            num_locations = H * W
            
            loc_out = outputs[scale_idx]        # (1, 8, H, W)
            conf_out = outputs[3 + scale_idx]   # (1, 4, H, W)
            land_out = outputs[6 + scale_idx]   # (1, 20, H, W)
            
            # ===== 重新排列数据 =====
            # 按照 C++ deal_loc/deal_conf 的访问模式：
            # loc[(hh * LOC_SIZE + cc) * size + ww]
            # 等价于 flat[ch, ww]，其中 size = H*W
            
            # Location: (1, 8, H, W) → flat(8, H*W)
            loc_flat = loc_out[0].reshape(8, -1)
            loc_scale1 = loc_flat[:4, :].T    # (H*W, 4)
            loc_scale2 = loc_flat[4:, :].T    # (H*W, 4)
            
            # Confidence: (1, 4, H, W) → flat(4, H*W)
            conf_flat = conf_out[0].reshape(4, -1)
            conf_scale1 = conf_flat[:2, :].T   # (H*W, 2)
            conf_scale2 = conf_flat[2:, :].T   # (H*W, 2)
            
            # Landmarks: (1, 20, H, W) → flat(20, H*W)
            land_flat = land_out[0].reshape(20, -1)
            land_scale1 = land_flat[:10, :].T   # (H*W, 10)
            land_scale2 = land_flat[10:, :].T   # (H*W, 10)
            
            # 交错合并两个尺度：[s1_pos0, s2_pos0, s1_pos1, s2_pos1, ...]
            n = num_locations
            loc_combined = np.empty((n * 2, 4), dtype=np.float32)
            loc_combined[0::2] = loc_scale1
            loc_combined[1::2] = loc_scale2
            
            conf_combined = np.empty((n * 2, 2), dtype=np.float32)
            conf_combined[0::2] = conf_scale1
            conf_combined[1::2] = conf_scale2
            
            land_combined = np.empty((n * 2, 10), dtype=np.float32)
            land_combined[0::2] = land_scale1
            land_combined[1::2] = land_scale2
            
            # Softmax 获取置信度
            scores = self._softmax_2class(conf_combined)
            
            # 获取对应的 anchors
            num_anchors_in_scale = n * 2
            scale_anchors = self.anchors[anchor_offset:anchor_offset + num_anchors_in_scale]
            
            # 解码 bbox
            boxes = self._decode_boxes(loc_combined, scale_anchors)
            
            # 解码 landmarks
            landmarks = self._decode_landmarks(land_combined, scale_anchors)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_landmarks.append(landmarks)
            
            anchor_offset += num_anchors_in_scale
        
        # 3. 合并所有尺度
        all_boxes = np.concatenate(all_boxes, axis=0)       # (4200, 4)
        all_scores = np.concatenate(all_scores, axis=0)     # (4200,)
        all_landmarks = np.concatenate(all_landmarks, axis=0)  # (4200, 10)
        
        # 4. 置信度过滤
        mask = all_scores > reliability_threshold
        filtered_boxes = all_boxes[mask]
        filtered_scores = all_scores[mask]
        filtered_landmarks = all_landmarks[mask]
        
        if len(filtered_boxes) == 0:
            return []
        
        # 5. 坐标还原到原图尺寸
        # C++ 使用 max_src_size 做坐标映射（模型输出是归一化到正方形空间的坐标）
        max_src_size = max(self.img_w, self.img_h)
        filtered_boxes[:, 0] *= max_src_size  # cx
        filtered_boxes[:, 1] *= max_src_size  # cy
        filtered_boxes[:, 2] *= max_src_size  # w
        filtered_boxes[:, 3] *= max_src_size  # h
        
        # 关键点坐标也乘以 max_src_size
        filtered_landmarks *= max_src_size
        
        # 裁剪到图像边界
        filtered_boxes[:, 0] = np.clip(filtered_boxes[:, 0], 0, self.img_w)
        filtered_boxes[:, 1] = np.clip(filtered_boxes[:, 1], 0, self.img_h)
        
        # 6. 转换为 xyxy 格式用于 NMS
        boxes_xyxy = self._convert_to_xyxy(filtered_boxes)
        
        # 7. NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            filtered_scores.tolist(),
            reliability_threshold,
            nms_threshold
        )
        
        # 8. 封装结果
        if len(indices) == 0:
            return []
        
        indices = indices.flatten()
        results = []
        for i in indices:
            result = FACE_DETECT_RESULT()
            box = filtered_boxes[i]
            
            # 从 center-wh 计算左上角坐标和宽高
            x1 = box[0] - box[2] / 2
            y1 = box[1] - box[3] / 2
            w = box[2]
            h = box[3]
            
            result.x = int(x1)
            result.y = int(y1)
            result.w = int(w)
            result.h = int(h)
            result.reliability = float(filtered_scores[i])
            
            # 5 个关键点（RetinaFace标准顺序：左眼、右眼、鼻子、左嘴角、右嘴角）
            lm = filtered_landmarks[i]
            result.left_eye = Landmark(int(lm[0]), int(lm[1]))
            result.right_eye = Landmark(int(lm[2]), int(lm[3]))
            result.nose = Landmark(int(lm[4]), int(lm[5]))
            result.left_mouth = Landmark(int(lm[6]), int(lm[7]))
            result.right_mouth = Landmark(int(lm[8]), int(lm[9]))
            
            results.append(result)
        
        return results
    
    def _softmax_2class(self, logits: np.ndarray) -> np.ndarray:
        """
        对 2 分类 logits 做 softmax，返回正类（人脸）概率
        Input: (N, 2)  Output: (N,)
        """
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[:, 1]
    
    def _decode_boxes(self, predictions: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """
        解码 bbox（来自 C++ get_box 函数）
        cx = anchor_x + tx * 0.1 * anchor_w
        cy = anchor_y + ty * 0.1 * anchor_h
        w  = anchor_w * exp(tw * 0.2)
        h  = anchor_h * exp(th * 0.2)
        """
        tx = predictions[:, 0]
        ty = predictions[:, 1]
        tw = predictions[:, 2]
        th = predictions[:, 3]
        
        cx = anchors[:, 0] + tx * 0.1 * anchors[:, 2]
        cy = anchors[:, 1] + ty * 0.1 * anchors[:, 3]
        w = anchors[:, 2] * np.exp(tw * 0.2)
        h = anchors[:, 3] * np.exp(th * 0.2)
        
        return np.stack([cx, cy, w, h], axis=1)
    
    def _decode_landmarks(self, predictions: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """
        解码 5 个关键点（来自 C++ get_landmark 函数）
        x = anchor_x + pred_x * 0.1 * anchor_w
        y = anchor_y + pred_y * 0.1 * anchor_h
        """
        result = np.zeros_like(predictions)
        for k in range(5):
            result[:, 2*k]   = anchors[:, 0] + predictions[:, 2*k]   * 0.1 * anchors[:, 2]
            result[:, 2*k+1] = anchors[:, 1] + predictions[:, 2*k+1] * 0.1 * anchors[:, 3]
        return result
    
    def _convert_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """center-wh → xyxy"""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return np.stack([x1, y1, x2, y2], axis=1)