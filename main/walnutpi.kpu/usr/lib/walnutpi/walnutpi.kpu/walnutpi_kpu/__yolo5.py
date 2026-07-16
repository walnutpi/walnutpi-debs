"""
YOLOv5 anchor-based 检测基类

封装 YOLOv5 系列模型的通用预处理和后处理逻辑。
子类只需配置 anchors、模型路径、输入尺寸等参数即可。

"""
import os
import numpy as np
import cv2
from typing import List
from walnutpi_kpu import KPU_BASE, NNCASEVersionType


class YOLO5_RESULT:
    """YOLOv5 检测结果"""
    x: int              # 框左上角的 x 坐标
    y: int              # 框左上角的 y 坐标
    w: int              # 框的宽度
    h: int              # 框的高度
    reliability: float  # 置信度 (0~1)
    label: int = 0      # 类别索引

    def __repr__(self):
        return (f"YOLO5_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, reliability={self.reliability:.3f}, "
                f"label={self.label})")


class YOLO5_BASE(KPU_BASE):
    """
    YOLOv5 anchor-based 检测基类

    子类需覆盖的类属性:
      - _ANCHORS: np.ndarray, shape (3, 3, 2) — 3 strides × 3 anchors × (w, h)
      - kmodel_name: str — 默认模型文件名（放置在子类目录下）

    子类可通过类属性调整的参数:
      - _STRIDES: 下采样倍数，默认 [8, 16, 32]
      - _DEFAULT_CONFIDENCE_THRESHOLD: 默认置信度阈值
      - _DEFAULT_NMS_THRESHOLD: 默认 NMS 阈值
      - result_class: 结果类，默认为 YOLO5_RESULT

    用法:
        class MY_DETECT(YOLO5_BASE):
            _ANCHORS = np.array([...], dtype=np.float32)
            kmodel_name = "my_model.kmodel"
            _DEFAULT_CONFIDENCE_THRESHOLD = 0.3

        detector = MY_DETECT()
        results = detector.run(img)
    """

    results: List[YOLO5_RESULT] = []

    # ---- 子类必须覆盖 ----
    _ANCHORS: np.ndarray = None  # type: ignore  (3, 3, 2)
    kmodel_name: str = ""        # 默认模型文件名

    # ---- 可选的子类覆盖 ----
    _STRIDES = [8, 16, 32]
    _NUM_ANCHORS = 3
    _NUM_OUTPUTS_PER_ANCHOR = 6   # tx, ty, tw, th, obj, cls0
    _DEFAULT_CONFIDENCE_THRESHOLD = 0.2
    _DEFAULT_NMS_THRESHOLD = 0.45
    result_class = YOLO5_RESULT

    # AI2D 配置：YOLO 用黑色填充
    ai2d_pad_color = [0, 0, 0]

    # 模型文件所在目录（子类可覆盖，默认取类名对应的子目录）
    _MODULE_DIR: str = ""

    def __init__(self,
                 kmodel_path: str = None,
                 size: int = 640,
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        初始化 YOLOv5 检测器

        @param kmodel_path: 模型路径，不传则使用子类目录下的 kmodel_name
        @param size: 模型输入尺寸（宽高相同）
        @param nncase_version: nncase 版本
        """
        if kmodel_path is None:
            if not self.kmodel_name:
                raise ValueError(
                    f"{type(self).__name__}: 请设置 kmodel_name 或传入 kmodel_path"
                )
            _res_dir = os.path.dirname(os.path.abspath(__file__))
            # 使用 _MODULE_DIR 或类名作为子目录名
            _sub_dir = self._MODULE_DIR if self._MODULE_DIR else type(self).__name__
            kmodel_path = os.path.join(_res_dir, _sub_dir, self.kmodel_name)

        if not os.path.exists(kmodel_path):
            raise FileNotFoundError(f"模型文件不存在: {kmodel_path}")

        super().__init__(kmodel_path, size, nncase_version)
        self.confidence_threshold = self._DEFAULT_CONFIDENCE_THRESHOLD
        self.nms_threshold = self._DEFAULT_NMS_THRESHOLD

        # 校验 anchors
        if self._ANCHORS is None:
            raise ValueError(f"{type(self).__name__}: 必须设置 _ANCHORS")

        self._NUM_CHANNELS = self._NUM_ANCHORS * self._NUM_OUTPUTS_PER_ANCHOR  # 18

    # ============================================================
    # 公开接口
    # ============================================================

    def run(self, img,
            reliability_threshold: float = None,
            nms_threshold: float = None) -> List:
        """同步检测"""
        if reliability_threshold is None:
            reliability_threshold = self.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.nms_threshold
        return super().run(img, reliability_threshold, nms_threshold)

    def get_result(self) -> List:
        """获取最近一次检测的结果"""
        return super().get_result()

    # ============================================================
    # AI2D 预处理（覆盖父类：使用居中 padding 匹配训练分布）
    # ============================================================

    def ai2d_init(self, model_w: int, model_h: int,
                  img_w: int, img_h: int):
        """
        与 KPU_BASE 的区别：padding 方式从「顶部-左侧对齐」改为「居中」。
        YOLO 模型是用居中 padding 训练的，grid 位置与绝对坐标绑定，
        顶部对齐会破坏这种绑定导致预测坐标偏移。
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
    # 后处理（YOLOv5 anchor-based 解码 + NMS）
    # ============================================================

    def post_process(self,
                     reliability_threshold: float,
                     nms_threshold: float) -> List:
        """
        YOLOv5 通用后处理：
        1. 从 nncase 扁平化输出 (1, 18, N) 重建为 (18, H, W) 空间格式
        2. 逐 anchor 解码 bbox + 置信度
        3. 坐标映射回原图 → NMS
        """
        all_boxes_xyxy = []
        all_scores = []
        all_labels = []

        for scale_idx, stride in enumerate(self._STRIDES):
            # -------------------------------------------------------
            # 1. 获取输出并重建空间格式
            # -------------------------------------------------------
            raw = self.kpu.get_output_tensor(scale_idx).to_numpy()
            output = _reshape_output(raw, stride, self.model_w,
                                     self._NUM_CHANNELS)

            C, H, W = output.shape
            if C != self._NUM_CHANNELS:
                raise ValueError(
                    f"输出通道数异常: 期望 {self._NUM_CHANNELS}, 实际 {C}"
                )

            # Reshape → (3 anchors, N values, H, W)
            output = output.reshape(
                self._NUM_ANCHORS, self._NUM_OUTPUTS_PER_ANCHOR, H, W
            )

            # 计算类别数: 总通道数 = 4(tx,ty,tw,th) + 1(obj) + num_classes
            num_classes = self._NUM_OUTPUTS_PER_ANCHOR - 5

            # 生成网格坐标
            gy, gx = np.meshgrid(
                np.arange(H, dtype=np.float32),
                np.arange(W, dtype=np.float32),
                indexing='ij'
            )

            # -------------------------------------------------------
            # 2. 逐 anchor 解码
            # -------------------------------------------------------
            for a in range(self._NUM_ANCHORS):
                anchor_w, anchor_h = self._ANCHORS[scale_idx, a]

                tx = output[a, 0].astype(np.float32)
                ty = output[a, 1].astype(np.float32)
                tw = output[a, 2].astype(np.float32)
                th = output[a, 3].astype(np.float32)
                obj_raw = output[a, 4].astype(np.float32)
                # 类别分数: 从索引 5 开始取 num_classes 个通道
                cls_raw = output[a, 5:5 + num_classes].astype(np.float32)

                # ---- YOLOv5 解码公式 ----
                # nncase 编译时通常已融合 Sigmoid，输出值已在 [0,1] 范围
                cx = (tx * 2.0 - 0.5 + gx) * stride
                cy = (ty * 2.0 - 0.5 + gy) * stride
                bw = anchor_w * (tw * 2.0) ** 2
                bh = anchor_h * (th * 2.0) ** 2

                # ---- 置信度: obj × max(cls) / 取最大类别 ----
                if num_classes == 1:
                    # 单类: cls_raw shape (1, H, W)
                    cls_scores = cls_raw[0]
                else:
                    # 多类: cls_raw shape (num_classes, H, W), 取最大值
                    cls_scores = np.max(cls_raw, axis=0)

                scores = obj_raw * cls_scores

                # 确定类别标签: 多类时取 argmax
                if num_classes > 1:
                    labels = np.argmax(cls_raw, axis=0)
                else:
                    labels = np.zeros_like(scores, dtype=np.int32)

                # ---- 展平 + 置信度过滤 ----
                cx, cy = cx.ravel(), cy.ravel()
                bw, bh = bw.ravel(), bh.ravel()
                scores = scores.ravel()
                labels = labels.ravel()

                mask = scores > reliability_threshold
                if not mask.any():
                    continue

                cx, cy = cx[mask], cy[mask]
                bw, bh = bw[mask], bh[mask]
                s = scores[mask]
                lbl = labels[mask]

                # ---------------------------------------------------
                # 3. 坐标映射：模型空间 → 原图空间
                # ---------------------------------------------------
                model_pad_x = (self.model_w - self.img_w * self.ratio) * 0.5
                model_pad_y = (self.model_h - self.img_h * self.ratio) * 0.5
                inv_ratio = 1.0 / self.ratio

                x1 = (cx - model_pad_x - bw * 0.5) * inv_ratio
                y1 = (cy - model_pad_y - bh * 0.5) * inv_ratio
                x2 = (cx - model_pad_x + bw * 0.5) * inv_ratio
                y2 = (cy - model_pad_y + bh * 0.5) * inv_ratio

                # 裁剪到图像边界
                np.clip(x1, 0, self.img_w, out=x1)
                np.clip(y1, 0, self.img_h, out=y1)
                np.clip(x2, 0, self.img_w, out=x2)
                np.clip(y2, 0, self.img_h, out=y2)

                xyxy = np.stack([x1, y1, x2, y2], axis=1)
                all_boxes_xyxy.append(xyxy)
                all_scores.append(s)
                all_labels.append(lbl)

        # -----------------------------------------------------------
        # 4. 无检测结果
        # -----------------------------------------------------------
        if not all_boxes_xyxy:
            return []

        # -----------------------------------------------------------
        # 5. 合并所有尺度 → NMS
        # -----------------------------------------------------------
        all_boxes = np.concatenate(all_boxes_xyxy, axis=0).astype(np.float32)
        all_scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0)

        indices = cv2.dnn.NMSBoxes(
            all_boxes.tolist(),
            all_scores.tolist(),
            reliability_threshold,
            nms_threshold
        )

        if len(indices) == 0:
            return []

        # -----------------------------------------------------------
        # 6. 封装结果
        # -----------------------------------------------------------
        indices = indices.flatten()
        results = []
        for i in indices:
            r = self.result_class()
            r.x = int(all_boxes[i, 0])
            r.y = int(all_boxes[i, 1])
            r.w = int(all_boxes[i, 2] - all_boxes[i, 0])
            r.h = int(all_boxes[i, 3] - all_boxes[i, 1])
            r.reliability = float(all_scores[i])
            r.label = int(all_labels[i])
            results.append(r)

        return results


# ================================================================
# 模块级工具函数
# ================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid"""
    result = np.empty_like(x, dtype=np.float32)
    mask = x >= 0
    result[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
    result[~mask] = np.exp(x[~mask]) / (1.0 + np.exp(x[~mask]))
    return result


def _reshape_output(raw: np.ndarray, stride: int,
                    model_size: int, num_channels: int) -> np.ndarray:
    """
    将 nncase 输出标准化为 (C, H, W) 空间格式

    nncase 编译的 YOLO 模型常见输出格式:
      - (1, C, N)    扁平化 NCHW（最常见）
      - (1, C, H, W) 空间 NCHW
      - (1, H, W, C) 空间 NHWC

    H = W = model_size // stride 是已知量
    """
    H = W = model_size // stride
    expected_spatial = H * W

    # --- 去掉 batch 维度 ---
    if raw.ndim == 4:
        raw = np.squeeze(raw, axis=0)

    if raw.ndim == 3:
        if raw.shape[1] == H and raw.shape[2] == W:
            return raw
        if raw.shape[0] == H and raw.shape[1] == W:
            return raw.transpose(2, 0, 1)
        if raw.size == num_channels * expected_spatial:
            if raw.shape[0] == num_channels:
                return raw.reshape(num_channels, H, W)
            elif raw.shape[-1] == num_channels:
                return raw.reshape(H, W, num_channels).transpose(2, 0, 1)

    elif raw.ndim == 2:
        if raw.shape[0] == num_channels:
            return raw.reshape(num_channels, H, W)
        elif raw.shape[1] == num_channels:
            return raw.reshape(H, W, num_channels).transpose(2, 0, 1)

    raise ValueError(
        f"无法解析 nncase 输出形状: {raw.shape}, "
        f"期望 {num_channels} 通道, 空间 {H}×{W}={expected_spatial}"
    )
