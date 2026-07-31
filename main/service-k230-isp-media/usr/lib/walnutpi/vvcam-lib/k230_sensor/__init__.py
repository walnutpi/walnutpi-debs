'''用于从csi摄像头读取图像的库'''
import atexit
import ctypes
import os

import cv2
import numpy as np

from ._SensorSetting import SensorSetting, SensorMode, DEV_ISP

# ── load C library ──────────────────────────────────────────────────
_so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "v4l-cam", "libv4l-cam.so")
_lib = ctypes.CDLL(_so_path)

# v4l_cam_ctx_t* v4l_cam_open(int device, int width, int height);
_lib.v4l_cam_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.v4l_cam_open.restype  = ctypes.c_void_p

# void v4l_cam_close(v4l_cam_ctx_t* ctx);
_lib.v4l_cam_close.argtypes = [ctypes.c_void_p]
_lib.v4l_cam_close.restype  = None

# int v4l_cam_read(ctx, uint8_t** data, int* w, int* h, int* size);
_lib.v4l_cam_read.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.v4l_cam_read.restype = ctypes.c_int

# int v4l_cam_set_hmirror(ctx, int enable);
_lib.v4l_cam_set_hmirror.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.v4l_cam_set_hmirror.restype  = ctypes.c_int

# int v4l_cam_set_vflip(ctx, int enable);
_lib.v4l_cam_set_vflip.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.v4l_cam_set_vflip.restype  = ctypes.c_int

# ── constants ───────────────────────────────────────────────────────

QQVGA = [320, 240]
QVGA = [320, 240]
VGA = [640, 480]
FHD = [1920, 1080]
HD = [1280, 720]
FRAME_SIZE_INVAILD = [-1, -1]


gc2093 = SensorSetting(
    i2c_addr=0x37,
    sensor="gc2093",
    mode=[
        SensorMode(1920, 1080, 60, 1),
        SensorMode(1920, 1080, 30, 0),
        SensorMode(1280, 960, 90, 2),
        SensorMode(1280, 720, 90, 3),
    ]
)

ov5647 = SensorSetting(
    i2c_addr=0x36,
    sensor="ov5647",
    mode=[
        SensorMode(1920, 1080, 30, 0),
        SensorMode(2592, 1944, 10, 1),
        SensorMode(1280, 960, 45, 2),
        SensorMode(1280, 720, 60, 3),
        SensorMode(640, 480, 90, 4),
    ]
)

sensor_list = [gc2093, ov5647]



class Sensor:

    def sensor_init(self):
        self.sensor = self._scan_sensor()
        self.sensor.set_closest_mode(self.width, self.height, self.fps)

    def __init__(self, width: int = 1920, height: int = 1080, id: int = 2) -> None:
        """初始化摄像头传感器。
        
        参数:
            width: 目标图像宽度，默认 1920。
            height: 目标图像高度，默认 1080。
        """
        self.fps = 60
        self.sensor = self._scan_sensor()
        self._ctx = None
        self._hmirror = False
        self._vflip = False
        self.isp_num = id
        self._v4l2_dev = (id + 1) * 3  # e.g. sensor=0 → /dev/video3


        self.mode = self.sensor.get_mode(self.isp_num)
        self.width = width
        self.height = height
        self._v4l2_w, self._v4l2_h = self._calc_v4l2_size(width, height)

        # open V4L2 device
        self._ctx = _lib.v4l_cam_open(self._v4l2_dev, self._v4l2_w, self._v4l2_h)
        if not self._ctx:
            raise RuntimeError(f"v4l_cam_open failed: cannot open /dev/{self._v4l2_dev}")

        # Best-effort cleanup on normal exit.  Ctrl+C (SIGINT)
        # works naturally now: v4l-cam.c no longer retries poll()
        # on EINTR, so KeyboardInterrupt can be delivered to Python
        # even when blocked in read().  The calling code should use
        # try/finally to call release() explicitly.
        atexit.register(self.release)

    # ── public API ───────────────────────────────────────────────

    def read(self):
        """读取一帧图像。
        
        返回:
            (True, img): 读取成功，img 为 numpy 数组格式的 BGR 图像。
            (False, None): 读取失败。
        """
        data_ptr = ctypes.POINTER(ctypes.c_uint8)()
        w = ctypes.c_int()
        h = ctypes.c_int()
        sz = ctypes.c_int()

        ret = _lib.v4l_cam_read(
            self._ctx,
            ctypes.byref(data_ptr),
            ctypes.byref(w), ctypes.byref(h), ctypes.byref(sz),
        )
        if ret != 0:
            return False, None

        # wrap raw BGR buffer as numpy array (must copy — reused next read)
        img = np.ctypeslib.as_array(data_ptr, shape=(h.value, w.value, 3))
        img = img.copy()

        # 将图像从rgb转为bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # crop to user-requested size (preserves aspect ratio)
        if w.value != self.width or h.value != self.height:
            img = img[0:self.height, 0:self.width]
            img = cv2.resize(img, (self.width, self.height))

        return True, img

    def set_framesize(self, width: int = 1920, height: int = 1080):
        """修改read函数返回的图像尺寸。
        
        参数:
            width: 新的目标宽度，默认 1920。
            height: 新的目标高度，默认 1080。
        """
        self.width = width
        self.height = height
        self._v4l2_w, self._v4l2_h = self._calc_v4l2_size(width, height)
        self.mode = self.sensor.get_mode(self.isp_num)

        # reopen device with new resolution
        if self._ctx:
            _lib.v4l_cam_close(self._ctx)
        self._ctx = _lib.v4l_cam_open(self._v4l2_dev, self._v4l2_w, self._v4l2_h)
        if not self._ctx:
            raise RuntimeError("v4l_cam_open failed in set_framesize")

    def _reopen(self):
        """关闭并重新打开 V4L2 设备。"""
        if self._ctx:
            _lib.v4l_cam_close(self._ctx)
            self._ctx = None
        self._ctx = _lib.v4l_cam_open(self._v4l2_dev, self._v4l2_w, self._v4l2_h)
        if not self._ctx:
            raise RuntimeError("v4l_cam_open failed during reopen")

    def set_hmirror(self, enable: bool = True):
        """水平镜像功能
        
        参数:
            enable: True 开启水平镜像，False 关闭，默认 True。
        """
        if not self._ctx:
            return
        enable = bool(enable)
        if enable == self._hmirror:
            return
        self._hmirror = enable
        self._reopen()
        _lib.v4l_cam_set_hmirror(self._ctx, 1 if enable else 0)

    def set_vflip(self, enable: bool = True):
        """垂直翻转

        参数:
            enable: True 开启垂直翻转，False 关闭，默认 True。
        """
        if not self._ctx:
            return
        enable = bool(enable)
        if enable == self._vflip:
            return
        self._vflip = enable
        self._reopen()
        _lib.v4l_cam_set_vflip(self._ctx, 1 if enable else 0)

    def isOpened(self):
        """判断摄像头是否已打开。
        
        返回:
            bool: 摄像头已打开返回 True，否则返回 False。
        """
        return self._ctx is not None

    def release(self):
        """关闭摄像头，释放 V4L2 设备资源。"""
        if self._ctx:
            _lib.v4l_cam_close(self._ctx)
            self._ctx = None

    def __del__(self):
        # Belt-and-suspenders: also try to clean up via GC
        self.release()

    # ── internals ────────────────────────────────────────────────

    def _scan_sensor(self, i2c_bus: int = 0) -> SensorSetting:
        """在指定 I2C 总线上扫描已知的摄像头传感器。
        
        参数:
            i2c_bus: I2C 总线编号，默认 0。
        
        返回:
            SensorSetting: 匹配到的传感器配置对象，若未匹配到则默认返回 gc2093。
        """
        for setting in sensor_list:
            if setting.check_i2c(i2c_bus):
                return setting
        return gc2093  # fallback

    def _calc_v4l2_size(self, width: int, height: int):
        """内部使用，计算保持传感器原始宽高比（16:9）的 V4L2 采集分辨率。
        
        参数:
            width: 目标宽度。
            height: 目标高度。
        
        返回:
            tuple[int, int]: 计算后的 V4L2 采集宽度和高度。
        """
        sensor_w = self.mode.width
        sensor_h = self.mode.height
        ratio = sensor_w / sensor_h  # e.g. 1920/1080 = 1.777...

        # try: fit the user's width into the native aspect ratio
        w_in_h_same = int(height * ratio) + 1
        h_in_w_same = int(width / ratio) + 1

        if h_in_w_same > height:
            # request taller frame, crop top/bottom later
            return width, h_in_w_same
        elif w_in_h_same > width:
            # request wider frame, crop left/right later
            return w_in_h_same, height
        else:
            return sensor_w, sensor_h


