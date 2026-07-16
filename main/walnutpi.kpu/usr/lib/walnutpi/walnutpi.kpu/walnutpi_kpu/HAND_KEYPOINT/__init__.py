"""
手掌关键点检测模块
二阶段流水线：YOLOv5 手掌检测 + 关键点回归模型
输出 21 个手部关键点坐标
"""
import os
import numpy as np
import cv2
from typing import List, Tuple
from walnutpi_kpu import get_nncase, NNCASEVersionType
from walnutpi_kpu.HAND_DETECT import HAND_DETECT


class Keypoint:
    """单个关键点"""
    x: int
    y: int

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Keypoint({self.x}, {self.y})"


class HAND_KEYPOINT_RESULT:
    """手掌关键点检测结果"""
    x: int                    # 手掌 bbox 左上角 x
    y: int                    # 手掌 bbox 左上角 y
    w: int                    # 手掌 bbox 宽度
    h: int                    # 手掌 bbox 高度
    reliability: float        # 检测置信度
    keypoints: List[Keypoint]  # 21 个手部关键点

    def __repr__(self):
        return (f"HAND_KEYPOINT_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, rel={self.reliability:.3f})")


class HandKPModel():
    """
    手掌关键点回归模型封装

    与普通检测模型不同，此模型需要先 crop 再推理。
    直接用 nncase Interpreter，不继承 KPU_BASE（避免不必要的线程/AI2D开销）。
    """

    def __init__(self,
                 kmodel_path: str,
                 size: int = 256,
                 nncase_version: NNCASEVersionType = "2.11"):
        self.model_w = size
        self.model_h = size
        self.nn = get_nncase(nncase_version)
        self.kpu = self.nn.Interpreter()
        self.kpu.load_model(kmodel_path)
        # 绑定输入 tensor
        tmp_tensor = self.nn.RuntimeTensor.from_numpy(
            np.ones((1, 3, self.model_h, self.model_w), dtype=np.uint8)
        )
        self.kpu.set_input_tensor(0, tmp_tensor)

    def run_crop(self, img, x1, y1, x2, y2) -> np.ndarray:
        """
        对检测框内的区域进行 crop 后推理，返回关键点坐标

        @param img: 原图 (H, W, 3) BGR
        @param x1, y1, x2, y2: 检测框坐标（原图坐标系）
        @return: 42 个值 (21 个关键点的 x, y 坐标)，在原图坐标系下
        """
        img_h, img_w = img.shape[:2]

        # 计算 crop 区域（扩大检测框 1.26 倍，确保完整手掌）
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        length = max(x2 - x1, y2 - y1) / 2
        ratio_num = 1.26 * length
        crop_x = int(max(0, cx - ratio_num))
        crop_y = int(max(0, cy - ratio_num))
        crop_x2 = int(min(img_w - 1, cx + ratio_num))
        crop_y2 = int(min(img_h - 1, cy + ratio_num))
        crop_w = int(crop_x2 - crop_x)
        crop_h = int(crop_y2 - crop_y)

        # OpenCV 裁剪 + resize → NCHW
        crop_img = img[crop_y:crop_y2, crop_x:crop_x2]
        crop_resized = cv2.resize(crop_img, (self.model_w, self.model_h),
                                  interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        img_nchw = np.array([img_rgb.transpose((2, 0, 1))], dtype=np.uint8)

        # 直接喂入 KPU（跳过 AI2D，因为已用 OpenCV 完成预处理）
        in_tensor = self.nn.RuntimeTensor.from_numpy(img_nchw)
        kpu_in = self.kpu.get_input_tensor(0)
        in_tensor.copy_to(kpu_in)
        self.kpu.run()

        # 后处理
        model_output = self.kpu.get_output_tensor(0).to_numpy()
        raw = model_output.flatten().astype(np.float32)

        # 自动检测输出范围并归一化到 [0,1]
        raw_mean = np.mean(np.abs(raw))
        if raw_mean > 1.0:
            raw = raw / self.model_w
        raw = np.clip(raw, 0.0, 1.0)

        # 映射回原图坐标
        results = np.zeros(raw.shape, dtype=np.int16)
        results[0::2] = raw[0::2] * crop_w + crop_x
        results[1::2] = raw[1::2] * crop_h + crop_y

        return results


class HAND_KEYPOINT:
    """
    手掌关键点检测类

    用法:
        hk = HAND_KEYPOINT()
        results = hk.run(img)
        for r in results:
            print(r.keypoints)  # 21 个关键点
    """

    def __init__(self,
                 hand_det_kmodel: str = None,
                 hand_kp_kmodel: str = None,
                 det_size: int = 512,
                 kp_size: int = 256,
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        初始化手掌关键点检测器

        @param hand_det_kmodel: 手掌检测模型路径，不传则使用默认
        @param hand_kp_kmodel: 关键点回归模型路径，不传则使用默认
        @param det_size: 检测模型输入尺寸
        @param kp_size: 关键点模型输入尺寸
        """
        _res_dir = os.path.dirname(os.path.abspath(__file__))

        if hand_det_kmodel is None:
            hand_det_kmodel = os.path.join(_res_dir, "..", "HAND_DETECT", "hand_det.kmodel")
        if hand_kp_kmodel is None:
            hand_kp_kmodel = os.path.join(_res_dir, "handkp_det.kmodel")

        self.detector = HAND_DETECT(kmodel_path=hand_det_kmodel, size=det_size,
                                    nncase_version=nncase_version)
        self.kp_model = HandKPModel(kmodel_path=hand_kp_kmodel, size=kp_size,
                                    nncase_version=nncase_version)

        self._det_size = det_size
        self._kp_size = kp_size

    @staticmethod
    def _is_valid_det(det, img_h, img_w) -> bool:
        """过滤不合理的检测框"""
        if det.h < (0.1 * img_h):
            return False
        x1, x2 = det.x, det.x + det.w
        w = det.w
        if (w < (0.25 * img_w) and
            (x1 < (0.03 * img_w) or x2 > (0.97 * img_w))):
            return False
        if (w < (0.15 * img_w) and
            (x1 < (0.01 * img_w) or x2 > (0.99 * img_w))):
            return False
        return True

    def run(self, img,
            reliability_threshold: float = 0.2,
            nms_threshold: float = 0.5) -> List[HAND_KEYPOINT_RESULT]:
        """
        检测手掌关键点

        @param img: BGR 图片
        @param reliability_threshold: 检测置信度阈值
        @param nms_threshold: NMS 阈值
        @return: HAND_KEYPOINT_RESULT 列表
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

            r = self._make_result(det, kp_raw)
            results.append(r)

        return results

    def _make_result(self, det, kp_raw: np.ndarray) -> HAND_KEYPOINT_RESULT:
        """将检测结果和关键点原始数据封装为 HAND_KEYPOINT_RESULT"""
        r = HAND_KEYPOINT_RESULT()
        r.x = det.x
        r.y = det.y
        r.w = det.w
        r.h = det.h
        r.reliability = det.reliability
        r.keypoints = [Keypoint(int(kp_raw[i * 2]), int(kp_raw[i * 2 + 1]))
                       for i in range(len(kp_raw) // 2)]
        return r

    @staticmethod
    def draw_keypoints(img, result: HAND_KEYPOINT_RESULT,
                       kp_color=(0, 255, 0), bbox_color=(255, 0, 255)):
        """
        在图片上绘制关键点和手掌框

        @param img: 图片（会被修改）
        @param result: 检测结果
        """
        # 绘制 bbox
        cv2.rectangle(img,
                      (result.x, result.y),
                      (result.x + result.w, result.y + result.h),
                      bbox_color, 2)

        # 绘制关键点
        for kp in result.keypoints:
            cv2.circle(img, (kp.x, kp.y), 3, kp_color, -1)

        # 绘制手指连线（5 根手指，每指 3 个关键点 + 腕部）
        if len(result.keypoints) >= 21:
            # 手部关键点连接索引（按手指分组）
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
                (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
                (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                (0, 13), (13, 14), (14, 15), (15, 16),# 无名指
                (0, 17), (17, 18), (18, 19), (19, 20),# 小指
            ]
            finger_colors = [
                (255, 0, 0),   # 拇指
                (0, 255, 0),   # 食指
                (255, 255, 0), # 中指
                (0, 255, 255), # 无名指
                (255, 0, 255), # 小指
            ]
            kps = result.keypoints
            for idx, (start, end) in enumerate(hand_connections):
                color = finger_colors[idx // 4] if idx < 20 else (128, 128, 128)
                cv2.line(img,
                         (kps[start].x, kps[start].y),
                         (kps[end].x, kps[end].y),
                         color, 2)
