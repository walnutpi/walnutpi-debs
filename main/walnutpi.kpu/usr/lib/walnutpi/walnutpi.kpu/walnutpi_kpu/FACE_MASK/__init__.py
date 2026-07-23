"""
口罩检测模块
自动完成人脸检测 + 口罩分类两阶段推理，用户只需调用一次 run()。

内部流程:
    1. FACE_DETECT（RetinaFace）检测人脸 → bbox + 5 关键点
    2. MobileNetV2 对每张人脸做 128×128 对齐 → 二分类（戴口罩/未戴口罩）

用法:
    from walnutpi_kpu.FACE_MASK import FACE_MASK

    detector = FACE_MASK()
    results = detector.run(img)

    for r in results:
        print(f"框: ({r.x},{r.y},{r.w},{r.h}), 戴口罩概率: {r.mask:.2f}")
"""

import numpy as np
import cv2
import os
import time
from walnutpi_kpu import KPU_BASE, NNCASEVersionType
from walnutpi_kpu.FACE_DETECT import FACE_DETECT, Landmark, FACE_DETECT_RESULT


class FACE_MASK_RESULT(FACE_DETECT_RESULT):
    mask: float  # 戴口罩的概率 (0~1)


class FACE_MASK(KPU_BASE):
    """
    口罩检测器，内部串联 人脸检测 + 口罩分类 两个模型。

    用法:
        detector = FACE_MASK()
        results = detector.run(img)
        for r in results:
            color = (0,255,0) if r.mask >= 0.5 else (0,0,255)
            cv2.rectangle(img, (r.x,r.y), (r.x+r.w,r.y+r.h), color, 2)
    """

    results: list = []

    # 来自 C++ umeyama_args_128: 基于 112×112 缩放到 128×128 的标准人脸关键点模板
    # 顺序: 左眼, 右眼, 鼻子, 左嘴角, 右嘴角
    _DST_POINTS = np.array([
        [38.2946 * 128 / 112,  51.6963 * 128 / 112],
        [73.5318 * 128 / 112,  51.5014 * 128 / 112],
        [56.0252 * 128 / 112,  71.7366 * 128 / 112],
        [41.5493 * 128 / 112,  92.3655 * 128 / 112],
        [70.7299 * 128 / 112,  92.2041 * 128 / 112],
    ], dtype=np.float32)

    def __init__(self,
                 det_kmodel_path: str = None,
                 det_anchors_path: str = None,
                 det_size: int = 320,
                 mask_kmodel_path: str = None,
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        初始化口罩检测器

        @det_size:         人脸检测模型尺寸，320 或 640
        @kmodel_path:      口罩分类 kmodel 路径，默认自动查找 FACE_MASK/face_mask.kmodel
        @det_kmodel_path:  人脸检测 kmodel 路径，默认根据 det_size 自动查找
        @det_anchors_path: 人脸检测 anchors 文件路径，默认根据 det_size 自动查找
        @nncase_version:   nncase 版本
        """
        # ---- 加载口罩分类 kmodel (128×128 输入) ----
        if mask_kmodel_path is None:
            _RES_DIR = os.path.dirname(os.path.abspath(__file__))
            mask_kmodel_path = os.path.join(_RES_DIR, "face_mask.kmodel")
        if not os.path.exists(mask_kmodel_path):
            raise FileNotFoundError(f"模型文件不存在: {mask_kmodel_path}")

        super().__init__(mask_kmodel_path, 128, nncase_version)

        self.results = []

        # ---- 内部创建人脸检测器 ----
        self._detector = FACE_DETECT(
            size=det_size,
            kmodel_path=det_kmodel_path,
            anchors_path=det_anchors_path,
            nncase_version=nncase_version,
        )

    def run(self, img: np.ndarray,
            det_thresh: float = 0.6,
            nms_thresh: float = 0.4) -> list:
        """
        检测图片中的所有人脸，并判断每张人脸是否戴口罩

        @img:          BGR 图片 (HWC)
        @det_thresh:   人脸检测置信度阈值
        @nms_thresh:   人脸检测 NMS 阈值
        @return: List[FACE_MASK_RESULT]
        """
        self.is_running = True
        self.has_result = False
        self.results = []

        try:
            self.img_w, self.img_h = img.shape[1], img.shape[0]

            # ==========================================
            # 第一阶段：人脸检测
            # ==========================================
            faces = self._detector.run(img, det_thresh, nms_thresh)

            if len(faces) == 0:
                self.speed.ms_inference = self._detector.speed.ms_inference
                self.speed.ms_post_process = self._detector.speed.ms_post_process
                self.has_result = True
                self.is_running = False
                return []

            # ==========================================
            # 第二阶段：对每张人脸做口罩分类
            # ==========================================
            total_infer = 0
            total_post = 0

            for face in faces:
                # 提取 5 个关键点坐标
                landmarks = [
                    [face.left_eye.x, face.left_eye.y],
                    [face.right_eye.x, face.right_eye.y],
                    [face.nose.x, face.nose.y],
                    [face.left_mouth.x, face.left_mouth.y],
                    [face.right_mouth.x, face.right_mouth.y],
                ]

                mask_prob = self._classify(img, landmarks)
                total_infer += self._mask_infer_time
                total_post += self._mask_post_time

                # ---- 组装结果 ----
                r = FACE_MASK_RESULT()
                r.x = face.x
                r.y = face.y
                r.w = face.w
                r.h = face.h
                r.reliability = face.reliability
                r.left_eye = face.left_eye
                r.right_eye = face.right_eye
                r.nose = face.nose
                r.left_mouth = face.left_mouth
                r.right_mouth = face.right_mouth
                r.mask = mask_prob
                self.results.append(r)

            # ---- 汇总耗时 ----
            self.speed.ms_inference = self._detector.speed.ms_inference + total_infer
            self.speed.ms_post_process = self._detector.speed.ms_post_process + total_post

        except Exception:
            import traceback
            traceback.print_exc()

        self.has_result = True
        self.is_running = False
        return self.results

    # ------------------------------------------------------------------
    # 内部方法：口罩分类
    # ------------------------------------------------------------------

    def _classify(self, img: np.ndarray, landmarks: list) -> float:
        """对单张人脸做口罩分类，返回戴口罩概率"""
        time_point = time.time() * 1000

        # 1. Umeyama 仿射对齐
        src_points = np.array(landmarks, dtype=np.float32).reshape(5, 2)
        M = self._umeyama_128(src_points)
        warped = cv2.warpAffine(img, M, (128, 128), flags=cv2.INTER_LINEAR)

        # 2. BGR → RGB + HWC → CHW
        img_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        img_nchw = np.array([img_rgb.transpose((2, 0, 1))], dtype=np.uint8)

        # 3. 送入 KPU
        input_tensor = self.nn.RuntimeTensor.from_numpy(img_nchw)
        self.kpu.set_input_tensor(0, input_tensor)
        self.kpu.run()

        self._mask_infer_time = time.time() * 1000 - time_point
        time_point = time.time() * 1000

        # 4. Softmax 后处理
        output = self.kpu.get_output_tensor(0).to_numpy()
        logits = output[0]  # (2,)
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        mask_prob = float(probs[1])  # class 1 = 戴口罩的概率

        self._mask_post_time = time.time() * 1000 - time_point
        return mask_prob

    def _umeyama_128(self, src_points: np.ndarray) -> np.ndarray:
        """
        Umeyama 算法：从 src 关键点 → 128×128 标准模板的相似变换矩阵
        @return: (2, 3) 仿射矩阵，可直接用于 cv2.warpAffine
        """
        dst = self._DST_POINTS

        src_mean = np.mean(src_points, axis=0)
        dst_mean = np.mean(dst, axis=0)

        src_demean = src_points - src_mean
        dst_demean = dst - dst_mean

        A = dst_demean.T @ src_demean / 5.0

        U, S, Vt = np.linalg.svd(A)

        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        src_var = np.sum(src_demean ** 2) / 5.0
        scale = np.sum(S) / src_var

        t = dst_mean - scale * (R @ src_mean)

        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = scale * R
        M[:2, 2] = t
        return M

    def get_result(self) -> list:
        self.has_result = False
        return self.results
