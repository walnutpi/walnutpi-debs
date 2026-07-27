'''手势识别，输出 21 个关键点 + 手势标签'''
import numpy as np
from typing import List
from walnutpi_kpu.HAND_KEYPOINT import HAND_KEYPOINT, HAND_KEYPOINT_RESULT, Keypoint


# 手势名称列表（索引与 gesture_id 一一对应）
GESTURE_NAMES = [
    "fist", "five", "gun", "love", "one",
    "six", "three", "thumbUp", "yeah",
]


class HAND_KEYPOINT_CLS_RESULT(HAND_KEYPOINT_RESULT):
    """手势识别结果，继承 HAND_KEYPOINT_RESULT 并增加手势标签

    Attributes:
        label: 手势编号，-1 表示未识别
    """

    label: int = -1

    def __repr__(self):
        return (f"HAND_KEYPOINT_CLS_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, rel={self.reliability:.3f}, "
                f"label={self.label})")


class HAND_KEYPOINT_CLS(HAND_KEYPOINT):
    """手势识别"""

    def run(self, img,
            reliability_threshold: float = 0.2,
            nms_threshold: float = 0.5) -> List[HAND_KEYPOINT_CLS_RESULT]:
        """检测图片中所有手部的关键点并识别手势

        Args:
            img: BGR 图片 (HWC)
            reliability_threshold: 手掌检测置信度阈值
            nms_threshold: 手掌检测 NMS 阈值

        Returns:
            List[HAND_KEYPOINT_CLS_RESULT]: 手势识别结果列表
        """
        dets = self.detector.run(img,
                                  reliability_threshold=reliability_threshold,
                                  nms_threshold=nms_threshold)

        img_h, img_w = img.shape[:2]
        results = []
        for det in dets:
            if not self._is_valid_det(det, img_h, img_w):
                continue

            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.w, det.y + det.h
            kp_raw = self.kp_model.run_crop(img, x1, y1, x2, y2)

            r = self._make_cls_result(det, kp_raw)
            results.append(r)

        return results

    def _make_cls_result(self, det, kp_raw: np.ndarray) -> HAND_KEYPOINT_CLS_RESULT:
        """封装结果并判断手势"""
        r = HAND_KEYPOINT_CLS_RESULT()
        r.x = det.x
        r.y = det.y
        r.w = det.w
        r.h = det.h
        r.reliability = det.reliability
        r.keypoints = [Keypoint(int(kp_raw[i * 2]), int(kp_raw[i * 2 + 1]))
                       for i in range(len(kp_raw) // 2)]
        r.label = self._classify_gesture(r.keypoints)
        return r

    # ============================================================
    # 手势分类逻辑（移植自 HandDetCls.py）
    # ============================================================

    @staticmethod
    def _vector_2d_angle(v1, v2) -> float:
        """计算两个二维向量的夹角（度）"""
        v1_x, v1_y = v1
        v2_x, v2_y = v2
        v1_norm = np.sqrt(v1_x * v1_x + v1_y * v1_y)
        v2_norm = np.sqrt(v2_x * v2_x + v2_y * v2_y)
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        dot_product = v1_x * v2_x + v1_y * v2_y
        cos_angle = dot_product / (v1_norm * v2_norm)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # 数值稳定
        return float(np.arccos(cos_angle) * 180 / np.pi)

    @staticmethod
    def _classify_gesture(keypoints: List[Keypoint]) -> int:
        """根据 21 个关键点判断手势，返回手势编号，-1 表示未识别"""
        if len(keypoints) < 21:
            return -1

        # 转为 42 个值的扁平数组（与原始代码兼容）
        kp_arr = np.zeros(42, dtype=np.int16)
        for i, p in enumerate(keypoints):
            if i < 21:
                kp_arr[i * 2] = p.x
                kp_arr[i * 2 + 1] = p.y

        angle_list = []
        for i in range(5):
            # 与原始 HandDetCls.hk_gesture 完全一致的索引计算
            v1 = (kp_arr[0] - kp_arr[i * 8 + 4], kp_arr[1] - kp_arr[i * 8 + 5])
            v2 = (kp_arr[i * 8 + 6] - kp_arr[i * 8 + 8],
                  kp_arr[i * 8 + 7] - kp_arr[i * 8 + 9])
            angle = HAND_KEYPOINT_CLS._vector_2d_angle(v1, v2)
            angle_list.append(angle)

        thr_angle = 65.0
        thr_angle_thumb = 53.0
        thr_angle_s = 49.0

        if any(a > 65534 for a in angle_list):
            return -1

        thumb, idx, mid, ring, pinky = angle_list

        if (thumb > thr_angle_thumb and idx > thr_angle and
            mid > thr_angle and ring > thr_angle and pinky > thr_angle):
            return 0  # fist
        if (thumb < thr_angle_s and idx < thr_angle_s and
              mid < thr_angle_s and ring < thr_angle_s and pinky < thr_angle_s):
            return 1  # five
        if (thumb < thr_angle_s and idx < thr_angle_s and
              mid > thr_angle and ring > thr_angle and pinky > thr_angle):
            return 2  # gun
        if (thumb < thr_angle_s and idx < thr_angle_s and
              mid > thr_angle and ring > thr_angle and pinky < thr_angle_s):
            return 3  # love
        if (thumb > 5 and idx < thr_angle_s and
              mid > thr_angle and ring > thr_angle and pinky > thr_angle):
            return 4  # one
        if (thumb < thr_angle_s and idx > thr_angle and
              mid > thr_angle and ring > thr_angle and pinky < thr_angle_s):
            return 5  # six
        if (thumb > thr_angle_thumb and idx < thr_angle_s and
              mid < thr_angle_s and ring < thr_angle_s and pinky > thr_angle):
            return 6  # three
        if (thumb < thr_angle_s and idx > thr_angle and
              mid > thr_angle and ring > thr_angle and pinky > thr_angle):
            return 7  # thumbUp
        if (thumb > thr_angle_thumb and idx < thr_angle_s and
              mid < thr_angle_s and ring > thr_angle and pinky > thr_angle):
            return 8  # yeah

        return -1
