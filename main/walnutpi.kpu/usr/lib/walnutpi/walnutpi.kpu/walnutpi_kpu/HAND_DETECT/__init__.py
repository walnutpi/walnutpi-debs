'''手掌检测，输出 bbox + 标签（手掌）'''
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
    """手掌检测"""

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
        Args:
            kmodel_path: kmodel 文件路径
            size: 模型输入尺寸
            nncase_version: nncase 版本，\"2.10\" 或 \"2.11\"
        """
        super().__init__(kmodel_path, size, nncase_version)
