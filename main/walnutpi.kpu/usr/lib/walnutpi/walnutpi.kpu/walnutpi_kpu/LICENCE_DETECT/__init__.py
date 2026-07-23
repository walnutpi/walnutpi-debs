"""
车牌检测模块 (Licence Plate Detection)

基于 RetinaFace 风格的单模型检测，直接输出 4 个角点坐标。

模型: licence_det.kmodel, 输入 640x640, 输出 9 个张量:
  loc_0..2    — bbox 偏移 (每尺度 4 值)
  conf_0..2   — 置信度 (每尺度 2 类, softmax)
  landms_0..2 — 角点偏移 (每尺度 8 值, 4 个角点)

用法:
    det = LICENCE_DETECT()
    results = det.run(img)
    LICENCE_DETECT.draw_results(img, results)
"""
import os
import cv2
import re
import numpy as np
from typing import List
from walnutpi_kpu import get_nncase, NNCASEVersionType


# ================================================================
class LICENCE_RESULT:
    """车牌检测+识别结果"""
    corners: list          # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] 整数像素坐标
    reliability: float           # 检测置信度
    text: str = ""         # 识别文本

    def __repr__(self):
        return f"LICENCE_RESULT(text='{self.text}', reliability={self.reliability:.3f})"

# ================================================================
def _load_anchors(anchors_path: str) -> np.ndarray:
    """加载锚框, 若传入 .bin 文件路径则直接加载, 否则从目录内查找"""
    if anchors_path.endswith(".bin"):
        return np.fromfile(anchors_path, dtype=np.float32).reshape(16800, 4)
    # 目录: 优先 .bin
    bin_path = os.path.join(anchors_path, "anchors_640.bin")
    if os.path.exists(bin_path):
        return np.fromfile(bin_path, dtype=np.float32).reshape(16800, 4)
    cpp_path = os.path.join(anchors_path, "anchors_640.cpp")
    if not os.path.exists(cpp_path):
        raise FileNotFoundError(f"锚框文件不存在: {bin_path} 或 {cpp_path}")
    with open(cpp_path, "r") as f:
        text = f.read()
    match = re.search(r"=\s*\{(.+)\}\s*;", text, re.DOTALL)
    if not match:
        raise ValueError(f"无法解析锚框: {cpp_path}")
    body = match.group(1)
    anchors = []
    for m in re.finditer(r"\{([^}]+)\}", body):
        anchors.append([float(x) for x in m.group(1).split(",")])
    return np.array(anchors, dtype=np.float32)

# ================================================================
class LICENCE_DETECTOR:
    """模型1: 车牌检测（输出4个角点）"""

    def __init__(self, kmodel_path: str = None, size: int = 640,
                 anchors_path: str = None,
                 nncase_version: NNCASEVersionType = "2.10"):
        """
        @param kmodel_path: licence_det.kmodel 路径, 默认使用模块自带
        @param anchors_path: anchors_640.bin 路径, 默认使用模块自带
        """
        self.model_w = size
        self.model_h = size
        _mod_dir = os.path.dirname(os.path.abspath(__file__))

        if kmodel_path is None:
            kmodel_path = os.path.join(_mod_dir, "licence_det.kmodel")
        if not os.path.exists(kmodel_path):
            raise FileNotFoundError(f"模型文件不存在: {kmodel_path}")

        self.nn = get_nncase(nncase_version)
        self.kpu = self.nn.Interpreter()
        self.kpu.load_model(kmodel_path)

        tmp = self.nn.RuntimeTensor.from_numpy(
            np.ones((1, 3, self.model_h, self.model_w), dtype=np.uint8))
        self.kpu.set_input_tensor(0, tmp)

        self.anchors = _load_anchors(anchors_path if anchors_path else _mod_dir)

        self.ai2d = self.nn.AI2D()
        self._ai2d_ready = False
        self._last_iw = -1
        self._last_ih = -1

    # ---- AI2D letterbox（保持宽高比，居中填充）----
    def _configure_ai2d(self, iw: int, ih: int):
        if self._ai2d_ready and self._last_iw == iw and self._last_ih == ih:
            return
        self._last_iw = iw; self._last_ih = ih

        ratio = min(self.model_w / iw, self.model_h / ih)
        new_w = int(iw * ratio)
        new_h = int(ih * ratio)
        pad_left = (self.model_w - new_w) // 2
        pad_top = (self.model_h - new_h) // 2
        pad_right = self.model_w - new_w - pad_left
        pad_bottom = self.model_h - new_h - pad_top

        # 保存坐标逆变换参数
        self._pad_left = pad_left
        self._pad_top = pad_top
        self._ratio = ratio

        self.ai2d.set_datatype(
            self.nn.AI2D_FORMAT.NCHW_FMT, self.nn.AI2D_FORMAT.NCHW_FMT,
            np.uint8, np.uint8)
        self.ai2d.set_resize_param(True,
            self.nn.AI2D_INTERP_METHOD.tf_bilinear,
            self.nn.AI2D_INTERP_MODE.half_pixel)
        self.ai2d.set_pad_param(True,
            [0,0,0,0, pad_top, pad_bottom, pad_left, pad_right],
            0, [114,114,114])
        self.ai2d.build([1,3,ih,iw], [1,3,self.model_h,self.model_w])
        self._ai2d_ready = True

    # ---- 预处理 ----
    def pre_process(self, img: np.ndarray):
        ih, iw = img.shape[:2]
        self._configure_ai2d(iw, ih)
        chw = img.transpose((2,0,1))
        nchw = np.array([chw], dtype=np.uint8)
        ai2d_in = self.nn.RuntimeTensor.from_numpy(nchw)
        kpu_in = self.kpu.get_input_tensor(0)
        self.ai2d.run(ai2d_in, kpu_in)

    # ---- 推理 ----
    def inference(self):
        self.kpu.run()

    # ---- 后处理 ----
    @staticmethod
    def _flat(t):
        return t[0].reshape(t.shape[1], -1)

    # ---- 向量化 deal 函数（替代 C++ 的三重 for 循环）----

    def _deal_conf_vec(self, cf: np.ndarray) -> np.ndarray:
        """conf (C,sz) → softmax 后取 class[1], 输出 (sz*2,) hh 在 ww 内层"""
        sz = cf.shape[1]
        r = cf.reshape(2, 2, sz)             # (hh=2, class=2, spatial)
        r = r - r.max(axis=1, keepdims=True)  # softmax 稳定
        e = np.exp(r); s = e.sum(axis=1, keepdims=True)
        r = e / s                              # softmax
        return r[:, 1, :].T.ravel()            # (sz,2)→transpose→flatten ww-first

    def _deal_loc_vec(self, lf: np.ndarray) -> np.ndarray:
        """loc (8,sz) → (sz*2, 4)"""
        return lf.reshape(2, 4, lf.shape[1]).transpose(2, 0, 1).reshape(-1, 4)

    def _deal_landms_vec(self, mf: np.ndarray) -> np.ndarray:
        """landms (16,sz) → (sz*2, 8)"""
        return mf.reshape(2, 8, mf.shape[1]).transpose(2, 0, 1).reshape(-1, 8)

    def _decode_boxes_vec(self, bx: np.ndarray, mask: np.ndarray = None):
        """向量化解码所有 box → (N,4) [cx,cy,w,h]; 可选 mask 只解码部分"""
        if mask is not None:
            a = self.anchors[mask]; v = bx[mask]
        else:
            a = self.anchors; v = bx
        out = np.zeros((a.shape[0], 4), dtype=np.float32)
        out[:, 0] = a[:, 0] + v[:, 0] * 0.1 * a[:, 2]
        out[:, 1] = a[:, 1] + v[:, 1] * 0.1 * a[:, 3]
        out[:, 2] = a[:, 2] * np.exp(v[:, 2] * 0.2)
        out[:, 3] = a[:, 3] * np.exp(v[:, 3] * 0.2)
        return out

    def _decode_landmarks_vec(self, lm: np.ndarray, mask: np.ndarray = None):
        """向量化解码所有 landmark → (N,8)"""
        if mask is not None:
            a = self.anchors[mask]; v = lm[mask]
        else:
            a = self.anchors; v = lm
        out = np.zeros((a.shape[0], 8), dtype=np.float32)
        for ll in range(4):
            out[:, 2 * ll + 0] = a[:, 0] + v[:, 2 * ll + 0] * 0.1 * a[:, 2]
            out[:, 2 * ll + 1] = a[:, 1] + v[:, 2 * ll + 1] * 0.1 * a[:, 3]
        return out

    # ---- 向量化 NMS ----

    def _nms(self, sp, bx, obj_thresh, nms_thresh):
        """向量化 NMS: 返回保留的 obj_index 列表"""
        valid = sp >= obj_thresh
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return []
        order = idx[np.argsort(-sp[idx])]

        keep = []
        suppressed = np.zeros(len(sp), dtype=bool)
        all_boxes = self._decode_boxes_vec(bx)

        for pos, oi in enumerate(order):
            if suppressed[oi]:
                continue
            keep.append(oi)
            ba = all_boxes[oi]

            # 取当前之后、未被抑制的候选框
            if pos + 1 >= len(order):
                break
            rest = order[pos + 1:]
            rest = rest[~suppressed[rest]]
            if len(rest) == 0:
                continue

            # 向量化 IoU
            bb = all_boxes[rest]
            ax1 = ba[0] - ba[2] / 2; ay1 = ba[1] - ba[3] / 2
            ax2 = ba[0] + ba[2] / 2; ay2 = ba[1] + ba[3] / 2
            bx1 = bb[:, 0] - bb[:, 2] / 2; by1 = bb[:, 1] - bb[:, 3] / 2
            bx2 = bb[:, 0] + bb[:, 2] / 2; by2 = bb[:, 1] + bb[:, 3] / 2
            iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
            ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
            inter = iw * ih
            union = ba[2] * ba[3] + bb[:, 2] * bb[:, 3] - inter
            ious = np.where(union > 0, inter / union, 0.0)
            suppressed[rest[ious >= nms_thresh]] = True

        return keep

    # ---- 后处理 ----

    def post_process(self, fw: int, fh: int,
                     obj_thresh: float = 0.5,
                     nms_thresh: float = 0.4) -> List[LICENCE_RESULT]:
        # 获取并展平 9 个输出张量
        lf = [self._flat(self.kpu.get_output_tensor(i).to_numpy()) for i in range(3)]
        cf = [self._flat(self.kpu.get_output_tensor(i+3).to_numpy()) for i in range(3)]
        mf = [self._flat(self.kpu.get_output_tensor(i+6).to_numpy()) for i in range(3)]

        N = self.anchors.shape[0]

        # 向量化 deal_conf（合并三个尺度）
        sp = np.concatenate([self._deal_conf_vec(cf[i]) for i in range(3)])

        # 向量化 deal_loc
        bx = np.concatenate([self._deal_loc_vec(lf[i]) for i in range(3)], axis=0)

        # 向量化 deal_landms
        lm = np.concatenate([self._deal_landms_vec(mf[i]) for i in range(3)], axis=0)

        # NMS（向量化 IoU）
        keep = self._nms(sp, bx, obj_thresh, nms_thresh)

        # 解码 + 坐标映射
        results = []
        if len(keep) > 0:
            keep_arr = np.array(keep, dtype=np.int64)
            decoded_lm = self._decode_landmarks_vec(lm, keep_arr)
            # 批量坐标映射: (norm*model - pad) / ratio
            decoded_lm[:, 0::2] = (decoded_lm[:, 0::2] * self.model_w - self._pad_left) / self._ratio
            decoded_lm[:, 1::2] = (decoded_lm[:, 1::2] * self.model_h - self._pad_top) / self._ratio

            for i, oi in enumerate(keep):
                cn = decoded_lm[i].reshape(4, 2).round().astype(np.int32)
                # 按 y 分上下两组, 组内按 x 排序 → TL,TR,BR,BL 顺时针
                si = np.argsort(cn[:, 1])
                top = cn[si[:2]][np.argsort(cn[si[:2], 0])]
                bot = cn[si[2:]][np.argsort(cn[si[2:], 0])]
                r = LICENCE_RESULT()
                r.corners = [[int(x), int(y)] for x, y in
                             [top[0], top[1], bot[1], bot[0]]]; r.reliability = float(sp[oi])
                results.append(r)

        return results

    def run(self, img: np.ndarray,
            obj_thresh: float = 0.5,
            nms_thresh: float = 0.4) -> List[LICENCE_RESULT]:
        fh, fw = img.shape[:2]
        self.pre_process(img)
        self.inference()
        return self.post_process(fw, fh, obj_thresh, nms_thresh)



# ================================================================
# 透视变换裁剪车牌（复刻 C++ warppersp）
# ================================================================

def warp_plate(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    从原图中透视变换裁剪车牌区域

    @param img: BGR 原图 (H, W, 3)
    @param corners: (4, 2) 四个角点像素坐标 (顺序由检测模型给出)
    @return: 裁剪+校正后的车牌 BGR 图
    """
    # minAreaRect + 排序角点
    rect = cv2.minAreaRect(np.float32(corners))
    vtx = cv2.boxPoints(rect)  # (4, 2)

    # 按 x 排序后区分左右，再按 y 分上下 → tl, tr, br, bl
    idx = np.argsort(vtx[:, 0])
    left = vtx[idx[:2]]; right = vtx[idx[2:]]
    if left[0, 1] < left[1, 1]:
        tl, bl = left[0], left[1]
    else:
        tl, bl = left[1], left[0]
    if right[0, 1] < right[1, 1]:
        tr, br = right[0], right[1]
    else:
        tr, br = right[1], right[0]
    src_pts = np.float32([tl, tr, br, bl])

    # 目标矩形尺寸: w = max(top_w, right_h), h = min(top_w, right_h)
    tw = np.linalg.norm(tr - tl)
    th = np.linalg.norm(br - tr)
    w = int(max(tw, th))
    h = int(min(tw, th))
    dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h))


# ================================================================
# 车牌识别模型（OCR 第二阶段）
# ================================================================

# 车牌字符字典（74 类）
RECO_DICT = [
    "挂","使","领","澳","港","皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏",
    "浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕",
    "甘","青","宁","新","警","学",
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z",
    "_","-",
]


class licence_rec:
    """车牌识别类（OCR 第二阶段）"""

    def __init__(self, kmodel_path: str = None,
                 model_w: int = 220, model_h: int = 32,
                 dict_list: List[str] = None,
                 nncase_version: NNCASEVersionType = "2.10"):
        """
        @param kmodel_path: licence_rec.kmodel 路径, 默认使用模块自带
        @param dict_list: 字符字典列表, 默认使用 RECO_DICT
        """
        _mod_dir = os.path.dirname(os.path.abspath(__file__))
        if kmodel_path is None:
            kmodel_path = os.path.join(_mod_dir, "licence_rec.kmodel")
        if not os.path.exists(kmodel_path):
            raise FileNotFoundError(f"识别模型不存在: {kmodel_path}")

        self._dict = dict_list if dict_list is not None else RECO_DICT
        self._in_w = model_w
        self._in_h = model_h

        self.nn = get_nncase(nncase_version)
        self.kpu = self.nn.Interpreter()
        self.kpu.load_model(kmodel_path)

        # 绑定输入 tensor
        self.kpu.set_input_tensor(0, self.nn.RuntimeTensor.from_numpy(
            np.ones((1, 1, model_h, model_w), dtype=np.uint8)))

    def pre_process(self, plate_img: np.ndarray):
        """
        BGR → Gray → stretch resize → 模型输入

        C++ Utils::resize 对 reco 模型直接拉伸到 220×32，不做 letterbox
        """
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self._in_w, self._in_h),
                             interpolation=cv2.INTER_LINEAR)
        self.kpu.set_input_tensor(0, self.nn.RuntimeTensor.from_numpy(
            np.array([[resized]], dtype=np.uint8)))

    def inference(self):
        self.kpu.run()

    def decode(self) -> str:
        """CTC 贪婪解码 → 车牌文本（复刻 C++ LicenceReco::post_process）"""
        t = self.kpu.get_output_tensor(0).to_numpy()
        t2d = t[:, 0, :]                          # (T,1,74) → (T,74)
        indices = np.argmax(t2d, axis=1).astype(np.int32)
        out = ""
        for i in range(len(indices)):
            ri = int(indices[i])
            if ri != 0 and not (i > 0 and int(indices[i - 1]) == ri):
                out += self._dict[ri - 1]
        return out

    def run(self, plate_img: np.ndarray) -> str:
        """预处理 → 推理 → CTC解码 → 车牌文本"""
        self.pre_process(plate_img)
        self.inference()
        return self.decode()


# ================================================================
# 车牌识别器（检测 + 识别 二合一，用户唯一入口）
# ================================================================

class LICENCE_DETECT:
    """
    车牌识别器 — 检测 + 识别二合一

    用法:
        lpr = LICENCE_DETECT()
        results = lpr.run(img)
        for r in results:
            print(r.text)
        LICENCE_DETECT.draw_results(img, results)
    """

    def __init__(self,
                 det_kmodel: str = None,
                 rec_kmodel: str = None,
                 anchors_path: str = None,
                 dict_list: List[str] = None,
                 det_size: int = 640,
                 rec_size: tuple = (220, 32)):
        """
        @param det_kmodel:   licence_det.kmodel 路径
        @param rec_kmodel:  licence_rec.kmodel 路径
        @param anchors_path: anchors_640.bin 路径
        @param dict_list:    字符字典列表
        @param det_size:     检测模型输入尺寸，默认 640
        @param rec_size:     识别模型输入尺寸 (w, h)，默认 (220, 32)
        """
        self._detector = LICENCE_DETECTOR(
            kmodel_path=det_kmodel, anchors_path=anchors_path, size=det_size)
        self._reco = licence_rec(
            kmodel_path=rec_kmodel, dict_list=dict_list,
            model_w=rec_size[0], model_h=rec_size[1])

    def run(self, img: np.ndarray,
            obj_thresh: float = 0.5,
            nms_thresh: float = 0.4) -> List[LICENCE_RESULT]:
        """
        @param obj_thresh: 检测置信度阈值
        @param nms_thresh: 检测 NMS 阈值
        @return: LICENCE_RESULT 列表（含 text）
        """
        results = []
        dets = self._detector.run(img, obj_thresh, nms_thresh)
        for det in dets:
            plate = warp_plate(img, det.corners)
            text = self._reco.run(plate)
            r = det  # det 已是 LICENCE_RESULT 类型
            r.text = text
            results.append(r)
        return results


