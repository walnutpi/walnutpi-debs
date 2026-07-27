'''交通信号灯检测，输出红绿灯的bbox与颜色'''
import numpy as np
from typing import List
from walnutpi_kpu.__yolo5 import YOLO5_BASE, YOLO5_RESULT


class TRAFFIC_LIGHT_DETECT_RESULT(YOLO5_RESULT):
    """交通信号灯检测结果

    Attributes:
        label_names: 类别名列表 [\"red\", \"green\", \"yellow\"]
        label_name: (property) 根据 label 索引返回类别名
    """

    label_names = ["red", "green", "yellow"]

    @property
    def label_name(self) -> str:
        if 0 <= self.label < len(self.label_names):
            return self.label_names[self.label]
        return str(self.label)

    def __repr__(self):
        return (f"TRAFFIC_LIGHT_DETECT_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, reliability={self.reliability:.3f}, "
                f"label={self.label}, name={self.label_name})")


class TRAFFIC_LIGHT_DETECT(YOLO5_BASE):
    """交通信号灯检测"""

    results: List[TRAFFIC_LIGHT_DETECT_RESULT] = []

    # YOLOv5 自定义小目标 anchors: 3 strides × 3 anchors × 2 (w, h)
    # 严格对齐 traffic_light_detect.h 中的 anchors_0/1/2
    _ANCHORS = np.array([
        [[3, 3],   [13, 14],  [34, 41]],       # stride 8  (小目标)
        [[40, 45], [49, 42],  [48, 50]],       # stride 16 (中目标)
        [[58, 52], [68, 59],  [80, 68]],       # stride 32 (大目标)
    ], dtype=np.float32)

    kmodel_name = "traffic.kmodel"
    result_class = TRAFFIC_LIGHT_DETECT_RESULT

    # 灰色 letterbox 填充，对齐 C++ 端 cv::Scalar(114,114,114)
    ai2d_pad_color = [114, 114, 114]

    # 5 + 3 类 = 8 通道/anchor（默认基类为 6，单类）
    _NUM_OUTPUTS_PER_ANCHOR = 8

    _DEFAULT_CONFIDENCE_THRESHOLD = 0.5   # 图片/摄像头模式均用 0.5
    _DEFAULT_NMS_THRESHOLD = 0.45

    _hint_printed: bool = False

    def __init__(self,
                 kmodel_path: str = None,
                 size: int = 640,
                 nncase_version: str = "2.11"):
        """
        Args:
            kmodel_path: traffic.kmodel 文件路径
            size: 模型输入尺寸
            nncase_version: nncase 版本，\"2.10\" 或 \"2.11\"
        """
        super().__init__(kmodel_path, size, nncase_version)
        self._print_init_hint()

    def _print_init_hint(self):
        """首次实例化时打印一次初始化提示信息"""
        if TRAFFIC_LIGHT_DETECT._hint_printed:
            return
        TRAFFIC_LIGHT_DETECT._hint_printed = True
        print("=" * 52)
        print("[TRAFFIC_LIGHT_DETECT] 交通信号灯检测器初始化完成")
        print(f"  模型文件 : {self.kmodel_name}")
        print(f"  输入尺寸 : {self.model_w} x {self.model_h}")
        print(f"  填充颜色 : gray(114,114,114)")
        print(f"  置信阈值 : {self.confidence_threshold}")
        print(f"  NMS 阈值 : {self.nms_threshold}")
        print(f"  检测类别 : red / green / yellow (RGB 输入)")
        print("=" * 52)
