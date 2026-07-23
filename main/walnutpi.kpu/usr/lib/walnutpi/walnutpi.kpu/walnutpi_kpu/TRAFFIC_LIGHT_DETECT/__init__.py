"""
交通信号灯检测模块 (YOLOv5, 3 类: red/green/yellow)
基于 YOLOv5 anchor-based 架构，检测图片中的红/绿/黄交通信号灯。
移植自 Canaan K230 linux_sdk/traffic_light_detect 示例。

默认输入尺寸 640×640，灰色 letterbox 填充(114)，RGB 输入（与 C++ 图片模式一致:
traffic_light_detect.cc 在 pre_process 中显式 cvtColor(BGR2RGB)）。
"""
import numpy as np
from typing import List
from walnutpi_kpu.__yolo5 import YOLO5_BASE, YOLO5_RESULT


class TRAFFIC_LIGHT_DETECT_RESULT(YOLO5_RESULT):
    """交通信号灯检测结果"""
    label_names = ["red", "green", "yellow"]   # index 0/1/2，对齐 C++ labels

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
    """
    交通信号灯检测类，基于 YOLOv5 架构（3 类: red/green/yellow）

    用法:
        detector = TRAFFIC_LIGHT_DETECT()                    # 默认 640 输入
        results = detector.run(img)                          # 同步检测
        detector.run_async(img)                              # 异步检测
        results = detector.get_result()                      # 获取异步结果

    说明:
        - 默认输入尺寸 640×640
        - 预处理填充色为灰色 114（对齐 C++ 端 Utils::padding_resize(114,114,114)）
        - 输入按 RGB 顺序（图片模式下 C++ 显式 BGR→RGB，与基类默认行为一致，故不覆盖 run）
        - 使用小目标自定义锚框（交通灯为远距离小目标）
        - 推理耗时可经 detector.speed.ms_inference / ms_post_process 读取
    """

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
        初始化交通信号灯检测器
        @param size: 模型输入尺寸，默认 640
        @param kmodel_path: 模型路径，不传则使用自带 kmodel_name
        @param nncase_version: nncase 版本
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
