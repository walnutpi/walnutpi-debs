"""
手掌检测模块 (YOLOv5n)
基于 YOLOv5n 架构，检测图片中的手掌
"""
import numpy as np
from typing import List
from walnutpi_kpu.__yolo5 import YOLO5_BASE, YOLO5_RESULT


class HAND_DETECT_RESULT(YOLO5_RESULT):
    """手掌检测结果"""
    def __repr__(self):
        return (f"HAND_RESULT_DET(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, reliability={self.reliability:.3f}, "
                f"label={self.label})")


class HAND_DETECT(YOLO5_BASE):
    """
    手掌检测类，基于 YOLOv5n 架构

    用法:
        detector = HAND_DETECT()
        results = detector.run(img)                    # 同步检测
        detector.run_async(img)                        # 异步检测
        results = detector.get_result()                # 获取异步结果
    """

    results: List[HAND_DETECT_RESULT] = []

    # 手掌检测 anchors（来自 hand_det.kmodel 训练配置）
    _ANCHORS = np.array([
        [[26, 27],  [53, 52],   [75, 71]],      # stride 8
        [[80, 99],  [106, 82],  [99, 134]],     # stride 16
        [[140, 113],[161, 172], [245, 276]],    # stride 32
    ], dtype=np.float32)

    kmodel_name = "hand_det.kmodel"
    result_class = HAND_DETECT_RESULT
    _DEFAULT_CONFIDENCE_THRESHOLD = 0.2
    _DEFAULT_NMS_THRESHOLD = 0.5

    def __init__(self,
                 kmodel_path: str = None,
                 size: int = 512,
                 nncase_version: str = "2.11"):
        """
        初始化手掌检测器

        @param size: 模型输入尺寸，默认 512
        @param kmodel_path: 模型路径，不传则使用自带的 hand_det.kmodel
        @param nncase_version: nncase 版本
        """
        super().__init__(kmodel_path, size, nncase_version)
