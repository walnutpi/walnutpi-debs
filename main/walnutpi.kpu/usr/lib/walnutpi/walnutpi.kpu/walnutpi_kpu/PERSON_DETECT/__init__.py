"""
人体检测模块 (YOLOv5n)
基于 YOLOv5n 架构，检测图片中的人体
"""
import numpy as np
from typing import List
from walnutpi_kpu.__yolo5 import YOLO5_BASE, YOLO5_RESULT


class PERSON_DETECT_RESULT(YOLO5_RESULT):
    """人体检测结果"""
    def __repr__(self):
        return (f"PERSON_RESULT_DET(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, reliability={self.reliability:.3f}, "
                f"label={self.label})")


class PERSON_DETECT(YOLO5_BASE):
    """
    人体检测类，基于 YOLOv5n 架构

    用法:
        detector = PERSON_DETECT()
        results = detector.run(img)                    # 同步检测
        detector.run_async(img)                        # 异步检测
        results = detector.get_result()                # 获取异步结果
    """

    results: List[PERSON_DETECT_RESULT] = []

    # YOLOv5n anchors: 3 strides × 3 anchors × 2 (w, h)
    _ANCHORS = np.array([
        [[10, 13],  [16, 30],  [33, 23]],       # stride 8  (小目标)
        [[30, 61],  [62, 45],  [59, 119]],      # stride 16 (中目标)
        [[116, 90], [156, 198], [373, 326]],    # stride 32 (大目标)
    ], dtype=np.float32)

    kmodel_name = "person_detect_yolov5n.kmodel"
    result_class = PERSON_DETECT_RESULT

    def __init__(self,
                 kmodel_path: str = None,
                 size: int = 640,
                 nncase_version: str = "2.11"):
        """
        初始化人体检测器

        @param size: 模型输入尺寸，默认 640
        @param kmodel_path: 模型路径，不传则使用自带的 person_detect_yolov5n.kmodel
        @param nncase_version: nncase 版本
        """
        super().__init__(kmodel_path, size, nncase_version)
