"""
跌倒检测模块 (YOLOv5n)
基于 YOLOv5n 架构，检测图片中的跌倒/未跌倒
"""
import numpy as np
from typing import List
from walnutpi_kpu.__yolo5 import YOLO5_BASE, YOLO5_RESULT


class FALL_DETECT_RESULT(YOLO5_RESULT):
    """跌倒检测结果"""
    def __repr__(self):
        return (f"FALL_RESULT_DET(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, reliability={self.reliability:.3f}, "
                f"label={self.label})")


class FALL_DETECT(YOLO5_BASE):
    """
    跌倒检测类，基于 YOLOv5n 架构

    用法:
        detector = FALL_DETECT()
        results = detector.run(img)                    # 同步检测
        detector.run_async(img)                        # 异步检测
        results = detector.get_result()                # 获取异步结果
    """

    results: List[FALL_DETECT_RESULT] = []

    # 跌倒检测 anchors（与 COCO YOLOv5n 相同）
    _ANCHORS = np.array([
        [[10, 13],  [16, 30],  [33, 23]],       # stride 8
        [[30, 61],  [62, 45],  [59, 119]],      # stride 16
        [[116, 90], [156, 198], [373, 326]],    # stride 32
    ], dtype=np.float32)

    # 跌倒检测有 2 个类别 (Fall, NoFall): tx,ty,tw,th,obj,cls0,cls1 = 7
    _NUM_OUTPUTS_PER_ANCHOR = 7

    kmodel_name = "yolov5n-falldown.kmodel"
    result_class = FALL_DETECT_RESULT

    def __init__(self,
                 kmodel_path: str = None,
                 size: int = 640,
                 nncase_version: str = "2.11"):
        """
        初始化跌倒检测器

        @param size: 模型输入尺寸，默认 640
        @param kmodel_path: 模型路径，不传则使用自带的 yolov5n-falldown.kmodel
        @param nncase_version: nncase 版本
        """
        super().__init__(kmodel_path, size, nncase_version)
