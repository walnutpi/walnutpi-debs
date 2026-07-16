"""
人体关键点检测模块 (YOLOv8n-pose)
基于 YOLOv8n-pose 架构，检测人体 17 个关键点
"""
import os
import numpy as np
import cv2
from typing import List
from walnutpi_kpu import KPU_BASE, NNCASEVersionType


class Keypoint:
    """单个关键点"""
    x: int          # x 坐标
    y: int          # y 坐标
    confidence: float  # 置信度 (0~1)

    def __repr__(self):
        return f"Keypoint(x={self.x}, y={self.y}, conf={self.confidence:.3f})"


class PERSON_KEYPOINT_RESULT:
    """人体关键点检测结果"""
    x: int              # bbox 左上角 x
    y: int              # bbox 左上角 y
    w: int              # bbox 宽度
    h: int              # bbox 高度
    reliability: float  # 检测置信度
    keypoints: List[Keypoint]  # 17 个关键点（COCO 格式）

    def __repr__(self):
        kps = ", ".join(
            f"({kp.x},{kp.y})" for kp in self.keypoints
        )
        return (f"PERSON_KEYPOINT_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, rel={self.reliability:.3f}, "
                f"kps=[{kps}])")


class PERSON_KEYPOINT(KPU_BASE):
    """
    人体关键点检测类，基于 YOLOv8n-pose 架构

    用法:
        detector = PERSON_KEYPOINT()
        results = detector.run(img)
    """

    results: List[PERSON_KEYPOINT_RESULT] = []

    # 默认阈值（与原 personPoint.py 一致）
    _DEFAULT_CONFIDENCE_THRESHOLD = 0.2
    _DEFAULT_NMS_THRESHOLD = 0.5

    # AI2D 配置：YOLO 用黑色填充
    ai2d_pad_color = [0, 0, 0]

    # YOLOv8-pose 模型参数
    _NUM_KEYPOINTS = 17
    _NUM_KP_VALUES = _NUM_KEYPOINTS * 3  # 51 (x, y, conf)
    _NUM_BBOX = 4
    _NUM_SCORE = 1
    _NUM_CHANNELS = _NUM_BBOX + _NUM_SCORE + _NUM_KP_VALUES  # 56

    # COCO 17 关键点名称
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    # 关键点颜色 (BGR，来自原 personPoint.py 的 KPS_COLORS)
    KPS_COLORS = [
        (255, 0, 255),    # 0: nose
        (255, 0, 255),    # 1: left_eye
        (255, 0, 255),    # 2: right_eye
        (255, 0, 255),    # 3: left_ear
        (255, 0, 255),    # 4: right_ear
        (128, 255, 255),  # 5: left_shoulder
        (128, 255, 255),  # 6: right_shoulder
        (128, 255, 255),  # 7: left_elbow
        (128, 255, 255),  # 8: right_elbow
        (128, 255, 255),  # 9: left_wrist
        (128, 255, 255),  # 10: right_wrist
        (153, 51, 255),   # 11: left_hip
        (153, 51, 255),   # 12: right_hip
        (153, 51, 255),   # 13: left_knee
        (153, 51, 255),   # 14: right_knee
        (153, 51, 255),   # 15: left_ankle
        (153, 51, 255),   # 16: right_ankle
    ]

    # 骨架颜色 (BGR，来自原 personPoint.py 的 LIMB_COLORS)
    LIMB_COLORS = [
        (153, 51, 255),   # 0: 踝-膝
        (153, 51, 255),   # 1: 踝-膝
        (153, 51, 255),   # 2: 踝-膝
        (153, 51, 255),   # 3: 踝-膝
        (51, 255, 255),   # 4: 髋交叉
        (51, 255, 255),   # 5: 肩-髋
        (51, 255, 255),   # 6: 肩-髋
        (128, 255, 255),  # 7: 肩交叉
        (128, 255, 255),  # 8: 肩-肘
        (128, 255, 255),  # 9: 肩-肘
        (128, 255, 255),  # 10: 肘-腕
        (128, 255, 255),  # 11: 肘-腕
        (255, 0, 255),    # 12: 眼-眼
        (255, 0, 255),    # 13: 鼻-眼
        (255, 0, 255),    # 14: 鼻-眼
        (255, 0, 255),    # 15: 眼-耳
        (255, 0, 255),    # 16: 眼-耳
        (255, 0, 255),    # 17: 耳-肩
        (255, 0, 255),    # 18: 耳-肩
    ]

    def __init__(self,
                 kmodel_path: str = None,
                 size: int = 320,
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        初始化人体关键点检测器

        @size: 模型输入尺寸，默认 320
        @kmodel_path: 模型路径，不传则使用自带的 yolov8n-pose.kmodel
        @nncase_version: nncase 版本
        """
        _RES_DIR = os.path.dirname(os.path.abspath(__file__))

        if kmodel_path is None:
            kmodel_path = os.path.join(_RES_DIR, "yolov8n-pose.kmodel")
        if not os.path.exists(kmodel_path):
            raise FileNotFoundError(f"模型文件不存在: {kmodel_path}")

        super().__init__(kmodel_path, size, nncase_version)
        self.confidence_threshold = self._DEFAULT_CONFIDENCE_THRESHOLD
        self.nms_threshold = self._DEFAULT_NMS_THRESHOLD

    # ============================================================
    # 公开接口
    # ============================================================

    def run(self, img,
            reliability_threshold: float = None,
            nms_threshold: float = None) -> List[PERSON_KEYPOINT_RESULT]:
        """同步检测人体关键点"""
        if reliability_threshold is None:
            reliability_threshold = self.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.nms_threshold
        return super().run(img, reliability_threshold, nms_threshold)

    def get_result(self) -> List[PERSON_KEYPOINT_RESULT]:
        """获取最近一次检测的结果"""
        return super().get_result()

    # ============================================================
    # AI2D 预处理（覆盖父类：使用居中 padding 匹配训练分布）
    # ============================================================

    def ai2d_init(self, model_w: int, model_h: int,
                  img_w: int, img_h: int):
        """
        与 KPU_BASE 的区别：padding 方式从「顶部-左侧对齐」改为「居中」。
        原始 personPoint.py 使用 get_padding_param() 做居中 padding。
        """
        if self.ai2d_2d_w == model_w and self.ai2d_2d_h == model_h:
            return
        self.ai2d_2d_w, self.ai2d_2d_h = model_w, model_h

        self.ratio = min(model_w / img_w, model_h / img_h)
        new_w, new_h = int(img_w * self.ratio), int(img_h * self.ratio)
        dw, dh = (model_w - new_w), (model_h - new_h)

        # 居中 padding
        pad_left = dw // 2
        pad_right = dw - pad_left
        pad_top = dh // 2
        pad_bottom = dh - pad_top

        self.ai2d.set_datatype(
            self.nn.AI2D_FORMAT.NCHW_FMT,
            self.nn.AI2D_FORMAT.NCHW_FMT,
            np.uint8, np.uint8,
        )
        self.ai2d.set_resize_param(
            True,
            self.nn.AI2D_INTERP_METHOD.tf_bilinear,
            self.nn.AI2D_INTERP_MODE.half_pixel,
        )
        self.ai2d.set_pad_param(
            True,
            [0, 0, 0, 0, pad_top, pad_bottom, pad_left, pad_right],
            0,
            self.ai2d_pad_color,
        )
        self.ai2d.build([1, 3, img_h, img_w], [1, 3, model_h, model_w])

    # ============================================================
    # 后处理（YOLOv8-pose 解码 + NMS）
    # ============================================================

    def post_process(self,
                     reliability_threshold: float,
                     nms_threshold: float) -> List[PERSON_KEYPOINT_RESULT]:
        """
        YOLOv8-pose 后处理（参照 aidemo.person_kp_postprocess）：
        1. 从输出 tensor 提取 bbox、score、keypoints
        2. 坐标映射回原图 + NMS
        """

        # -------------------------------------------------------
        # 1. 获取模型输出
        # -------------------------------------------------------
        model_output = self.kpu.get_output_tensor(0).to_numpy()
        predictions = model_output[0].transpose()  # (N, 56)

        boxes = predictions[:, :self._NUM_BBOX].astype(np.float32)
        scores = predictions[:, self._NUM_BBOX].astype(np.float32)
        keypoints = predictions[:, self._NUM_BBOX + self._NUM_SCORE:].astype(np.float32)

        # -------------------------------------------------------
        # 2. 置信度过滤
        # -------------------------------------------------------
        mask = scores > reliability_threshold
        if not mask.any():
            return []

        boxes = boxes[mask]
        scores = scores[mask]
        keypoints = keypoints[mask]

        # -------------------------------------------------------
        # 3. 坐标映射：模型空间 → 原图空间
        #    与 aicube/aidemo 一致：(坐标 - pad) / ratio
        # -------------------------------------------------------
        model_pad_x = (self.model_w - self.img_w * self.ratio) * 0.5
        model_pad_y = (self.model_h - self.img_h * self.ratio) * 0.5
        inv_ratio = 1.0 / self.ratio

        # bbox: [cx, cy, w, h] → 缩放
        boxes_scaled = boxes.copy()
        boxes_scaled[:, 0] = (boxes[:, 0] - model_pad_x) * inv_ratio
        boxes_scaled[:, 1] = (boxes[:, 1] - model_pad_y) * inv_ratio
        boxes_scaled[:, 2] *= inv_ratio
        boxes_scaled[:, 3] *= inv_ratio

        # keypoints: [kx, ky, kconf, ...] × 17 → 缩放
        kps_scaled = keypoints.copy()
        for k in range(self._NUM_KEYPOINTS):
            kps_scaled[:, k * 3] = (keypoints[:, k * 3] - model_pad_x) * inv_ratio
            kps_scaled[:, k * 3 + 1] = (keypoints[:, k * 3 + 1] - model_pad_y) * inv_ratio

        # -------------------------------------------------------
        # 4. NMS
        # -------------------------------------------------------
        # center-wh → xyxy
        x1 = boxes_scaled[:, 0] - boxes_scaled[:, 2] * 0.5
        y1 = boxes_scaled[:, 1] - boxes_scaled[:, 3] * 0.5
        x2 = boxes_scaled[:, 0] + boxes_scaled[:, 2] * 0.5
        y2 = boxes_scaled[:, 1] + boxes_scaled[:, 3] * 0.5
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            reliability_threshold,
            nms_threshold,
        )

        if len(indices) == 0:
            return []

        # -------------------------------------------------------
        # 5. 封装结果
        # -------------------------------------------------------
        indices = indices.flatten()
        results = []
        for i in indices:
            r = PERSON_KEYPOINT_RESULT()
            # bbox 左上角
            r.x = int(boxes_scaled[i, 0] - boxes_scaled[i, 2] * 0.5)
            r.y = int(boxes_scaled[i, 1] - boxes_scaled[i, 3] * 0.5)
            r.w = int(boxes_scaled[i, 2])
            r.h = int(boxes_scaled[i, 3])
            r.reliability = float(scores[i])
            # 17 个关键点
            r.keypoints = []
            for k in range(self._NUM_KEYPOINTS):
                kp = Keypoint()
                kp.x = int(kps_scaled[i, k * 3])
                kp.y = int(kps_scaled[i, k * 3 + 1])
                kp.confidence = float(kps_scaled[i, k * 3 + 2])
                r.keypoints.append(kp)
            results.append(r)

        return results
