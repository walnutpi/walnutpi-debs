"""
抽烟检测模块 (YOLOv5s, 单类 smoke)
基于 YOLOv5 anchor-based 架构，检测图片中的抽烟行为。
移植自 Canaan K230 linux_sdk/smoke_detect 示例 (smoke_detect.h / smoke_detect.cc)。

默认输入尺寸 640×640，灰色 letterbox 填充(114)，BGR 输入（与 C++ 端保持一致）。
"""
import os
import time
import numpy as np
from typing import List
from walnutpi_kpu.__yolo5 import YOLO5_BASE, YOLO5_RESULT


class SMOKE_DETECT_RESULT(YOLO5_RESULT):
    """抽烟检测结果"""
    label_name: str = "smoke"

    def __repr__(self):
        return (f"SMOKE_DETECT_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, reliability={self.reliability:.3f}, "
                f"label={self.label})")


class SMOKE_DETECT(YOLO5_BASE):
    """
    抽烟检测类，基于 YOLOv5s 架构（单类：smoke）

    用法:
        detector = SMOKE_DETECT()                          # 默认 640 输入
        results = detector.run(img)                        # 同步检测
        detector.run_async(img)                            # 异步检测
        results = detector.get_result()                    # 获取异步结果

    说明:
        - 默认输入尺寸 640×640（yolov5s 标准）
        - 预处理填充色为灰色 114（对齐 C++ 端 Utils::padding_resize(114,114,114)）
        - 输入按 BGR 顺序（C++ 端 hwc_to_chw 未做色彩转换），与训练分布一致
        - 推理耗时可经 detector.speed.ms_inference / ms_post_process 读取
    """

    results: List[SMOKE_DETECT_RESULT] = []

    # YOLOv5 anchors: 3 strides × 3 anchors × 2 (w, h) —— 与 smoke_detect.h 完全一致
    _ANCHORS = np.array([
        [[10, 13],  [16, 30],  [33, 23]],       # stride 8  (小目标)
        [[30, 61],  [62, 45],  [59, 119]],      # stride 16 (中目标)
        [[116, 90], [156, 198], [373, 326]],    # stride 32 (大目标)
    ], dtype=np.float32)

    kmodel_name = "yolov5s_smoke_best.kmodel"
    result_class = SMOKE_DETECT_RESULT

    # 灰色 letterbox 填充，对齐 C++ 端 cv::Scalar(114,114,114)
    ai2d_pad_color = [114, 114, 114]

    _DEFAULT_CONFIDENCE_THRESHOLD = 0.5   # 图片模式 sd_thresh
    _DEFAULT_NMS_THRESHOLD = 0.45

    _hint_printed: bool = False

    def __init__(self,
                 kmodel_path: str = None,
                 size: int = 640,
                 nncase_version: str = "2.11"):
        """
        初始化抽烟检测器
        @param size: 模型输入尺寸，默认 640
        @param kmodel_path: 模型路径，不传则使用自带 kmodel_name
        @param nncase_version: nncase 版本
        """
        super().__init__(kmodel_path, size, nncase_version)
        self._print_init_hint()

    def _print_init_hint(self):
        """首次实例化时打印一次初始化提示信息"""
        if SMOKE_DETECT._hint_printed:
            return
        SMOKE_DETECT._hint_printed = True
        print("=" * 52)
        print("[SMOKE_DETECT] 抽烟检测器初始化完成")
        print(f"  模型文件 : {self.kmodel_name}")
        print(f"  输入尺寸 : {self.model_w} x {self.model_h}")
        print(f"  填充颜色 : gray(114,114,114)")
        print(f"  置信阈值 : {self.confidence_threshold}")
        print(f"  NMS 阈值 : {self.nms_threshold}")
        print(f"  检测类别 : 单类 smoke (BGR 输入)")
        print("=" * 52)

    def run(self, img, reliability_threshold=None, nms_threshold=None):
        """同步检测（覆盖父类以保留 BGR 输入顺序）"""
        if reliability_threshold is None:
            reliability_threshold = self.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.nms_threshold

        self.is_running = True
        self.has_result = False
        time_point = time.time() * 1000
        try:
            self.img_w, self.img_h = img.shape[1], img.shape[0]
            self.ai2d_init(self.model_w, self.model_h, self.img_w, self.img_h)

            img_nchw = np.array([img.transpose((2, 0, 1))])

            ai2d_input_tensor = self.nn.RuntimeTensor.from_numpy(img_nchw)
            kpu_input_tensor = self.kpu.get_input_tensor(0)
            self.ai2d.run(ai2d_input_tensor, kpu_input_tensor)

            self.kpu.run()
            self.speed.ms_inference = time.time() * 1000 - time_point
            time_point = time.time() * 1000

            self.results = self.post_process(reliability_threshold, nms_threshold)
            self.speed.ms_post_process = time.time() * 1000 - time_point
        except Exception as e:
            import traceback
            traceback.print_exc()

        self.has_result = True
        self.is_running = False
        return self.results
