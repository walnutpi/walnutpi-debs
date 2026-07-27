import os
import numpy as np
import cv2
from typing import List
from walnutpi_kpu import *
from typing import Literal


class YOLO_RESULT_DET:
    """YOLO 检测结果

    Attributes:
        x: bbox 左上角 x 坐标（像素）
        y: bbox 左上角 y 坐标（像素）
        w: bbox 宽度（像素）
        h: bbox 高度（像素）
        xywh: bbox 中心格式 (cx, cy, w, h)
        label: 类别索引
        reliability: 置信度，范围 [0, 1]
        index_in_all_boxes: 在全部候选框中的索引
    """

    x: int
    y: int
    w: int
    h: int
    xywh: np.ndarray
    label: int
    reliability: float
    index_in_all_boxes: int


class YOLO_RESULT_OBB(YOLO_RESULT_DET):
    """YOLO 旋转检测结果

    Attributes:
        angle: 旋转角度（弧度）
    """

    angle: float

    def _rotate_point(self, cx, cy, x, y, angle):
        """旋转点(x, y)围绕中心点(cx, cy)旋转angle弧度"""
        s, c = np.sin(angle), np.cos(angle)
        x_new = c * (x - cx) - s * (y - cy) + cx
        y_new = s * (x - cx) + c * (y - cy) + cy
        return int(x_new), int(y_new)

    def get_top_left(self):
        """获取旋转后的左上角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x - half_w, self.y - half_h, self.angle
        )

    def get_bottom_left(self):
        """获取旋转后的左下角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x - half_w, self.y + half_h, self.angle
        )

    def get_top_right(self):
        """获取旋转后的右上角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x + half_w, self.y - half_h, self.angle
        )

    def get_bottom_right(self):
        """获取旋转后的右下角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x + half_w, self.y + half_h, self.angle
        )


class YOLO_RESULT_SEG(YOLO_RESULT_DET):
    """YOLO 分割检测结果

    Attributes:
        contours: 边界点坐标列表 [((x1,y1), ...)]
        mask: 分割掩码图，物体区域为 255，背景为 0
    """

    contours: list
    mask: np.ndarray
    _raw_mask: np.ndarray


class _YOLO_KEYPOINT:
    """YOLO 单个关键点（内部使用）

    Attributes:
        xy: 关键点坐标 (x, y)
        visibility: 可见度，范围 [0, 1]
    """

    xy = (0, 0)
    visibility: float


class YOLO_RESULT_POSE(YOLO_RESULT_DET):
    """YOLO 姿态估计结果

    Attributes:
        keypoints: 关键点列表
    """

    keypoints: List[_YOLO_KEYPOINT] = []


class _YOLO_RESULT_CLS_INDEX:
    """YOLO 分类索引-置信度对（内部使用）

    Attributes:
        label: 类别索引
        reliability: 置信度
    """

    label: int
    reliability: float

    def __init__(self, label=0, reliability=0):
        self.label = label
        self.reliability = reliability


class YOLO_RESULT_CLS:
    """YOLO 分类结果

    Attributes:
        top5: Top-5 分类结果列表
        all: 所有类别的置信度数组，all[i] 为类别 i 的置信度
    """

    top5 = [
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
    ]
    all = np.zeros(1)



class YOLO11_DET(KPU_BASE):
    """YOLO11 目标检测"""

    results: List[YOLO_RESULT_DET] = []
    _result_type = YOLO_RESULT_DET

    def get_result(self) -> List[YOLO_RESULT_DET]:
        """获取最近一次检测的结果

        Returns:
            List[YOLO_RESULT_DET]
        """
        return super().get_result()

    def run(
        self, img, reliability_threshold=0.5, nms_threshold=0.5
    ) -> List[YOLO_RESULT_DET]:
        """检测图片中的目标

        Args:
            img: BGR 图片 (HWC)
            reliability_threshold: 置信度阈值
            nms_threshold: NMS 阈值

        Returns:
            List[YOLO_RESULT_DET]: 检测结果列表
        """
        return super().run(img, reliability_threshold, nms_threshold)

    def post_process(self, reliability_threshold, nms_threshold):
        """YOLO11 检测后处理：提取 bbox → 归一化坐标还原 → NMS"""
        # 获取模型输出
        model_output = self.kpu.get_output_tensor(0).to_numpy()
        predictions = model_output[0].transpose()  # (8400, 84)

        boxes = predictions[:, :4]  # [x_center, y_center, w, h]
        class_scores = predictions[:, 4:]  # 各类别置信度

        # 取每个候选框的最大类别得分
        scores = np.max(class_scores, axis=1)
        class_ids = class_scores.argmax(axis=1)

        # 过滤低置信度目标
        mask = scores > reliability_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 将归一化坐标还原到原图尺寸
        boxes_scaled = boxes.copy()
        boxes_scaled[:, 0] /= self.ratio  # x_center
        boxes_scaled[:, 1] /= self.ratio  # y_center
        boxes_scaled[:, 2] /= self.ratio  # width
        boxes_scaled[:, 3] /= self.ratio  # height

        scores = scores.astype(np.float32)

        # -------------------------------
        # 执行 NMS
        # -------------------------------
        boxes_xy = boxes_scaled[:, :2] - boxes_scaled[:, 2:4] / 2  # top-left corner
        boxes_xy2 = (
            boxes_scaled[:, :2] + boxes_scaled[:, 2:4] / 2
        )  # bottom-right corner
        boxes_xyxy = np.concatenate([boxes_xy, boxes_xy2], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), scores.tolist(), reliability_threshold, nms_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            res: List[YOLO_RESULT_DET] = []
            for i in indices:
                # 直接使用缩放后的xywh格式数据，避免重复转换
                box_xywh = boxes_scaled[i]

                score = scores[i]
                class_id = class_ids[i]
                re = self._result_type()
                re.index_in_all_boxes = i

                re.x = int(box_xywh[0] - box_xywh[2] / 2)  # 左上角 x
                re.y = int(box_xywh[1] - box_xywh[3] / 2)  # 左上角 y
                re.w = int(box_xywh[2])  # width
                re.h = int(box_xywh[3])  # height
                re.xywh = box_xywh
                re.reliability = score
                re.label = class_id
                res.append(re)
            return res

        return []
class YOLO11_CLS(KPU_BASE):
    """YOLO11 图像分类"""

    results = YOLO_RESULT_CLS()

    def get_result(self) -> YOLO_RESULT_CLS:
        """获取最近一次分类的结果

        Returns:
            YOLO_RESULT_CLS
        """
        return super().get_result()

    def run(
        self, img, reliability_threshold=0.5, nms_threshold=0.5
    ) -> YOLO_RESULT_CLS:
        """对图片进行分类

        Args:
            img: BGR 图片 (HWC)
            reliability_threshold: 置信度阈值
            nms_threshold: 未使用

        Returns:
            YOLO_RESULT_CLS: 分类结果（含 top5 和 all）
        """
        return super().run(img, reliability_threshold, nms_threshold)

    def post_process(self, reliability_threshold, nms_threshold):
        """YOLO11 分类后处理：取出 Top-5 类别索引和置信度"""
        model_output = self.kpu.get_output_tensor(0).to_numpy()
        tensor = model_output[0].transpose()

        top_5_indices = np.argsort(tensor)[-5:][::-1]
        ret = YOLO_RESULT_CLS()
        ret.all = tensor
        indices = 0
        for cls_index in top_5_indices:
            ret.top5[indices] = _YOLO_RESULT_CLS_INDEX(cls_index, tensor[cls_index])
            indices += 1
        if indices < 5:
            for i in range(5 - indices):
                ret.top5[indices + i] = _YOLO_RESULT_CLS_INDEX(
                    top_5_indices[-1], tensor[top_5_indices[-1]]
                )
        return ret
