'''OCR，识别图片中的文字'''
import os
import numpy as np
import cv2
import queue
import threading
from typing import List
from walnutpi_kpu import get_nncase, NNCASEVersionType


# ================================================================
# 工具函数
# ================================================================

def _find_rectangle_vertices(points: np.ndarray) -> np.ndarray:
    order = np.argsort(points[:, 0])
    left, right = points[order[:2]], points[order[2:]]
    tl, bl = (left[0], left[1]) if left[0,1] < left[1,1] else (left[1], left[0])
    tr, br = (right[0], right[1]) if right[0,1] < right[1,1] else (right[1], right[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _expand(v, sw, sh, mw, mh):
    tl, tr, br, bl = v
    def _c(x, lo, hi): return max(lo, min(x, hi))
    return np.array([
        [_c(sw*tl[0]-(sw-1)*tr[0],0,mw), _c(sh*tl[1]-(sh-1)*bl[1],0,mh)],
        [_c(sw*tr[0]-(sw-1)*tl[0],0,mw), _c(sh*tr[1]-(sh-1)*br[1],0,mh)],
        [_c(sw*br[0]-(sw-1)*bl[0],0,mw), _c(sh*br[1]-(sh-1)*tr[1],0,mh)],
        [_c(sw*bl[0]-(sw-1)*br[0],0,mw), _c(sh*bl[1]-(sh-1)*tl[1],0,mh)],
    ], dtype=np.float32)


def _unclip(contour):
    a = abs(cv2.contourArea(contour))
    p = cv2.arcLength(contour, True)
    if p < 1e-6: return contour
    d = a * 1.5 / p
    if d < 0.5: return contour
    off = int(p) + 10
    ms = off * 2 + 10
    m = np.zeros((ms, ms), dtype=np.uint8)
    cv2.fillPoly(m, [(contour+off).astype(np.int32)], 255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3,int(d*2+1))|1,)*2)
    co, _ = cv2.findContours(cv2.dilate(m, k, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return co[0][:,0,:].astype(np.float32)-off if co else contour


def _load_dict(path):
    chars = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: chars.append(line)
    return chars


def _ctc_decode(pred, chars, blank_idx):
    """CTC 贪心解码（与 RTOS 一致）"""
    ids = np.argmax(pred, axis=1)
    out = []
    prev = blank_idx
    for idx in ids:
        if idx != blank_idx and idx != prev:
            out.append(chars[idx])
        prev = idx
    return "".join(out)


# ================================================================
# 检测结果类
# ================================================================

class OCR_DET_RESULT:
    """OCR 检测结果

    Attributes:
        x: 文字区域左上角 x 坐标（像素）
        y: 文字区域左上角 y 坐标（像素）
        w: 文字区域宽度（像素）
        h: 文字区域高度（像素）
        reliability: 检测置信度，范围 [0, 1]
        polygon: 四边形顶点列表 [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    """

    x: int
    y: int
    w: int
    h: int
    reliability: float
    polygon: List

    def __init__(self, x=0, y=0, w=0, h=0, reliability=0.0, polygon=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.reliability = reliability
        self.polygon = polygon or []

    def __repr__(self):
        return (f"OCR_DET_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, "
                f"reliability={self.reliability:.3f})")


class OCR_RESULT:
    """OCR 完整结果（检测+识别）

    Attributes:
        x: 文字区域左上角 x 坐标（像素）
        y: 文字区域左上角 y 坐标（像素）
        w: 文字区域宽度（像素）
        h: 文字区域高度（像素）
        text: 识别文本
        reliability: 检测置信度，范围 [0, 1]
        polygon: 四边形顶点列表 [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    """

    x: int
    y: int
    w: int
    h: int
    text: str
    reliability: float
    polygon: List

    def __init__(self, x=0, y=0, w=0, h=0, text="", reliability=0.0, polygon=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.reliability = reliability
        self.polygon = polygon or []

    def __repr__(self):
        return (f"OCR_RESULT(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, text='{self.text}', "
                f"reliability={self.reliability:.3f})")


# ================================================================
# OCR 检测主类（模型1）
# ================================================================

class OCR_DET:
    """文字检测"""

    _DET_SIZE = 640
    _PROC_W = 640
    _PROC_H = 360

    def __init__(self, kmodel_path=None, det_size=None,
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        Args:
            kmodel_path: ocr_det_int16.kmodel 文件路径
            det_size: 模型输入尺寸
            nncase_version: nncase 版本，\"2.10\" 或 \"2.11\"
        """
        if det_size is None:
            det_size = self._DET_SIZE
        _RES_DIR = os.path.dirname(os.path.abspath(__file__))
        if kmodel_path is None:
            kmodel_path = os.path.join(_RES_DIR, "ocr_det_int16.kmodel")
        if not os.path.exists(kmodel_path):
            raise FileNotFoundError(f"模型不存在: {kmodel_path}")

        self.nn = get_nncase(nncase_version)
        self.model_w = self.model_h = det_size
        self.mask_threshold = 0.25
        self.box_threshold = 0.3

        self.kpu = self.nn.Interpreter()
        self.ai2d = self.nn.AI2D()
        self.kpu.load_model(kmodel_path)
        tmp = self.nn.RuntimeTensor.from_numpy(
            np.ones((1,3,self.model_h,self.model_w), dtype=np.uint8))
        self.kpu.set_input_tensor(0, tmp)

        self._cw = self._ch = -1  # ai2d 缓存
        self._pad_top = 140  # AI2D 顶部 padding，由 _ai2d_build 设定

    # ---------------------------------------------------------------
    # 后处理
    # ---------------------------------------------------------------

    def _postprocess(self, seg_map):
        """分割图 → 四边形列表"""
        pt = self._pad_top
        contours, _ = cv2.findContours(
            (seg_map>self.mask_threshold).astype(np.uint8)*255,
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        res = []
        for c in contours:
            if len(c) < 4: continue
            ct = c[:,0,:].astype(np.float32)

            # unclip 后取最小外接矩形
            b2 = cv2.boxPoints(cv2.minAreaRect(_unclip(ct).astype(np.int32)))

            # boxScore：在 640x640 空间中计算分割置信度
            xm = max(0, int(np.floor(np.min(ct[:,0]))))
            xx = min(639, int(np.ceil(np.max(ct[:,0]))))
            ym = max(0, int(np.floor(np.min(ct[:,1]))))
            yx = min(639, int(np.ceil(np.max(ct[:,1]))))
            if yx-ym < 1 or xx-xm < 1: continue
            mp = np.zeros((yx-ym+1, xx-xm+1), dtype=np.uint8)
            cv2.fillPoly(mp, [(ct-[xm,ym]).astype(np.int32)], 1)
            sc = float(cv2.mean(seg_map[ym:yx+1, xm:xx+1], mp)[0])
            if sc < self.box_threshold: continue

            # 640x640 → 640x360：x不变，y减去顶部 padding
            sp = np.empty((4,2), dtype=np.float32)
            for m in range(4):
                sp[m,0] = np.clip(b2[m,0], 0, 639)
                sp[m,1] = np.clip(b2[m,1] - pt, 0, 359)

            fp = cv2.boxPoints(cv2.minAreaRect(sp))
            fs = _find_rectangle_vertices(fp)
            fe = _expand(fs, 1.025, 1.15, 640, 360)

            bx = int(np.clip(np.min(fe[:,0]), 0, 640))
            by = int(np.clip(np.min(fe[:,1]), 0, 360))
            bw = int(np.clip(np.max(fe[:,0]), 0, 640)) - bx
            bh = int(np.clip(np.max(fe[:,1]), 0, 360)) - by
            if bw < 2 or bh < 2: continue
            res.append(OCR_DET_RESULT(
                x=bx, y=by, w=bw, h=bh,
                reliability=sc, polygon=fe.tolist()))
        return res

    # ---------------------------------------------------------------
    # 公开接口
    # ---------------------------------------------------------------

    def run(self, img, mask_threshold=None, box_threshold=None):
        """检测图片中的文字区域

        Args:
            img: BGR 图片 (HWC)
            mask_threshold: 分割掩码阈值
            box_threshold: 文字框置信度阈值

        Returns:
            List[OCR_DET_RESULT]
        """
        if mask_threshold is not None: self.mask_threshold = mask_threshold
        if box_threshold is not None: self.box_threshold = box_threshold

        orig_h, orig_w = img.shape[:2]

        # 1. 缩放到固定尺寸（模拟 RTOS 的 rgb888p_size）
        img_proc = cv2.resize(img, (self._PROC_W, self._PROC_H))
        ph, pw = self._PROC_H, self._PROC_W

        # 2. AI2D: 640x360 → 640x640（居中 padding）
        if self._cw != pw or self._ch != ph:
            self._cw, self._ch = pw, ph
            mw, mh = self.model_w, self.model_h
            r = min(mw/pw, mh/ph)
            nw, nh = int(pw*r), int(ph*r)
            self._pad_top = (mh - nh) // 2
            self._pad_bot = mh - nh - self._pad_top
            pad_l = (mw - nw) // 2
            pad_r = mw - nw - pad_l
            self.ai2d.set_datatype(self.nn.AI2D_FORMAT.NCHW_FMT,
                                   self.nn.AI2D_FORMAT.NCHW_FMT,
                                   np.uint8, np.uint8)
            self.ai2d.set_resize_param(True,
                self.nn.AI2D_INTERP_METHOD.tf_bilinear,
                self.nn.AI2D_INTERP_MODE.half_pixel)
            self.ai2d.set_pad_param(True, [0,0,0,0,self._pad_top,self._pad_bot,pad_l,pad_r], 0, [0,0,0])
            self.ai2d.build([1,3,ph,pw], [1,3,mh,mw])

        # 推理
        img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        ai2d_in = self.nn.RuntimeTensor.from_numpy(
            np.array([img_rgb.transpose((2,0,1))]))
        self.ai2d.run(ai2d_in, self.kpu.get_input_tensor(0))
        del img_rgb, ai2d_in

        self.kpu.run()

        # 3. 获取分割图（DMA→普通内存）
        seg = self.kpu.get_output_tensor(0).to_numpy()[0,:,:,0].copy()

        # 4. 后处理（结果在 PRO 空间）
        results_pro = self._postprocess(seg)

        # 5. 坐标映射回原图
        if results_pro:
            sx = orig_w / self._PROC_W
            sy = orig_h / self._PROC_H
            for r in results_pro:
                r.x = int(r.x * sx)
                r.y = int(r.y * sy)
                r.w = int(r.w * sx)
                r.h = int(r.h * sy)
                r.polygon = [[p[0]*sx, p[1]*sy] for p in r.polygon]

        self.results = results_pro
        return self.results


# ================================================================
# OCR 识别类（模型2：ocr_rec_int16.kmodel）
# ================================================================

class OCR_REC:
    """文字识别"""

    def __init__(self, kmodel_path=None, dict_path=None,
                 rec_size=(512, 32),
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        Args:
            kmodel_path: ocr_rec_int16.kmodel 文件路径
            dict_path: 字典文件路径
            rec_size: 模型输入尺寸 (w, h)
            nncase_version: nncase 版本，\"2.10\" 或 \"2.11\"
        """
        _RES_DIR = os.path.dirname(os.path.abspath(__file__))
        if kmodel_path is None:
            kmodel_path = os.path.join(_RES_DIR, "ocr_rec_int16.kmodel")
        if not os.path.exists(kmodel_path):
            raise FileNotFoundError(f"识别模型不存在: {kmodel_path}")

        if dict_path is None:
            dict_path = os.path.join(_RES_DIR, "dict.txt")
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"字典不存在: {dict_path}")

        self.nn = get_nncase(nncase_version)
        self.rec_w, self.rec_h = rec_size
        self.chars = _load_dict(dict_path)
        self.blank_idx = len(self.chars)  # blank = 最后一个索引（与 RTOS 一致）

        # 加载模型
        self.kpu = self.nn.Interpreter()
        self.ai2d = self.nn.AI2D()
        self.kpu.load_model(kmodel_path)
        tmp = self.nn.RuntimeTensor.from_numpy(
            np.ones((1, 3, self.rec_h, self.rec_w), dtype=np.uint8))
        self.kpu.set_input_tensor(0, tmp)

    # ---------------------------------------------------------------
    # 公开接口
    # ---------------------------------------------------------------

    def run(self, crop_img):
        """识别单张裁剪图片中的文字

        Args:
            crop_img: BGR 裁剪图 (H, W, 3)

        Returns:
            str: 识别的文字
        """
        rh, rw = crop_img.shape[:2]
        tw, th = self.rec_w, self.rec_h

        # 保持宽高比 resize
        scale = min(tw / rw, th / rh)
        nw, nh = int(rw * scale), int(rh * scale)
        crop_r = cv2.resize(crop_img, (nw, nh))

        # 右下对齐 pad（与 RTOS letterbox_pad_param 一致：top=0, left=0）
        pad_b = th - nh
        pad_r = tw - nw
        crop_pad = cv2.copyMakeBorder(
            crop_r, 0, pad_b, 0, pad_r,
            cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # BGR→RGB + HWC→NCHW
        img_rgb = cv2.cvtColor(crop_pad, cv2.COLOR_BGR2RGB)
        img_nchw = np.array([img_rgb.transpose((2, 0, 1))])

        # 直接 set_input_tensor
        inp = self.nn.RuntimeTensor.from_numpy(img_nchw)
        self.kpu.set_input_tensor(0, inp)
        self.kpu.run()

        # CTC 解码（blank = 最后一个索引，与 RTOS 一致）
        out = self.kpu.get_output_tensor(0).to_numpy()
        pred = out[:, 0, :]  # (128, num_classes)
        return _ctc_decode(pred, self.chars, self.blank_idx)


# ================================================================
# OCR 完整类（检测 + 识别）
# ================================================================

class OCR:
    """OCR 识别"""

    results: List[OCR_RESULT] = []
    is_running: bool = False
    has_result: bool = False

    def __init__(self,
                 det_model_path=None,
                 rec_model_path=None,
                 dict_path=None,
                 det_size=640,
                 rec_size=(512, 32),
                 nncase_version: NNCASEVersionType = "2.11"):
        """
        Args:
            det_model_path: 检测模型文件路径
            rec_model_path: 识别模型文件路径
            dict_path: 字典文件路径
            det_size: 检测模型输入尺寸
            rec_size: 识别模型输入尺寸 (w, h)
            nncase_version: nncase 版本，\"2.10\" 或 \"2.11\"
        """
        self.det = OCR_DET(det_model_path, det_size or OCR_DET._DET_SIZE, nncase_version)
        self.rec = OCR_REC(rec_model_path, dict_path, rec_size, nncase_version)

        # 异步任务队列和工作线程
        self._task_queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def run(self, img, confidence=0.5):
        """检测并识别图片中的文字（同步）

        Args:
            img: BGR 图片 (HWC)
            confidence: 置信度阈值，低于此值的检测框被丢弃

        Returns:
            List[OCR_RESULT]: 识别结果列表
        """
        # 先跑检测
        det_results = self.det.run(img)

        if not det_results:
            self.results = []
            return []

        results = []
        for dr in det_results:
            # 检测置信度低于门槛 → 整个丢弃
            if dr.reliability < confidence:
                continue

            # 矩形裁剪
            x1 = max(0, dr.x)
            y1 = max(0, dr.y)
            x2 = min(img.shape[1], dr.x + dr.w)
            y2 = min(img.shape[0], dr.y + dr.h)
            crop = img[y1:y2, x1:x2]

            # 识别
            try:
                text = self.rec.run(crop)
            except Exception:
                text = ""

            # 识别后置信度低于门槛 → 不显示文字
            final_text = text if dr.reliability >= confidence else ""

            results.append(OCR_RESULT(
                x=dr.x, y=dr.y, w=dr.w, h=dr.h,
                text=final_text, reliability=dr.reliability,
                polygon=dr.polygon))

        self.results = results
        return results

    def run_async(self, img, confidence=0.5):
        """检测并识别图片中的文字（异步，立即返回）

        Args:
            img: BGR 图片 (HWC)
            confidence: 置信度阈值
        """
        if not self.is_running:
            self.is_running = True
            self.has_result = False
            self._task_queue.put((img, confidence))
        else:
            print("OCR 正在运行中，请等待当前任务完成")

    def get_result(self):
        """获取最近一次异步推理的结果

        Returns:
            List[OCR_RESULT]: 若无结果则返回空列表
        """
        self.has_result = False
        return self.results

    def _worker_loop(self):
        """工作线程：不断从队列取任务执行"""
        while not self._shutdown_event.is_set():
            try:
                task_data = self._task_queue.get(timeout=0.1)
                if task_data is None:
                    break
                img, confidence = task_data
                self.run(img, confidence)
                self.has_result = True
                self.is_running = False
                self._task_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                if not self._task_queue.empty():
                    self._task_queue.task_done()

    def __del__(self):
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
