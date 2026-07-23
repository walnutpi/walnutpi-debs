
import ctypes
import os

import cv2
import numpy as np

from ._SensorSetting import SensorSetting, SensorMode

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
        # SensorMode(1920, 1080, 30, 0),
        SensorMode(1920, 1080, 60, 1),
    ]
)

ov5647 = SensorSetting(
    i2c_addr=0x36,
    sensor="ov5647",
    mode=[
        SensorMode(1920, 1080, 30, 0)
    ]
)

sensor_list = [gc2093, ov5647]


class Sensor:
    """K230 camera sensor, backed by libv4l-cam.so for V4L2 capture."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.fps = 60
        self.sensor = self._scan_sensor()
        self._ctx = None
        self._hmirror = False
        self._vflip = False

        # determine the nearest 16:9 sensor mode and the V4L2 resolution
        # that preserves aspect ratio (avoids distortion)
        self.mode = self.sensor.get_mode(width, height, self.fps)
        self.width = width
        self.height = height
        self._v4l2_w, self._v4l2_h = self._calc_v4l2_size(width, height)
        self.sensor.set_mode(self.mode)

        # open V4L2 device
        self._ctx = _lib.v4l_cam_open(1, self._v4l2_w, self._v4l2_h)
        if not self._ctx:
            raise RuntimeError("v4l_cam_open failed: cannot open /dev/video1")

    # ── public API ───────────────────────────────────────────────

    def read(self):
        """Read one BGR frame. Returns (True, img) or (False, None)."""
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
        """Change frame size. Re-opens the V4L2 device with new resolution."""
        self.width = width
        self.height = height
        self.mode = self.sensor.get_mode(width, height, self.fps)
        self._v4l2_w, self._v4l2_h = self._calc_v4l2_size(width, height)
        self.sensor.set_mode(self.mode)

        # reopen device with new resolution
        if self._ctx:
            _lib.v4l_cam_close(self._ctx)
        self._ctx = _lib.v4l_cam_open(1, self._v4l2_w, self._v4l2_h)
        if not self._ctx:
            raise RuntimeError("v4l_cam_open failed in set_framesize")

    def _reopen(self):
        """Close and re-open the V4L2 device from scratch."""
        if self._ctx:
            _lib.v4l_cam_close(self._ctx)
            self._ctx = None
        self.sensor.set_mode(self.mode)
        self._ctx = _lib.v4l_cam_open(1, self._v4l2_w, self._v4l2_h)
        if not self._ctx:
            raise RuntimeError("v4l_cam_open failed during reopen")

    def set_hmirror(self, enable: bool = True):
        """Hardware horizontal mirror (V4L2_CID_HFLIP).
        
        Because most V4L2 drivers do not allow changing this control while
        the stream is active, we close and re-open the device internally.
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
        """Hardware vertical flip (V4L2_CID_VFLIP).
        
        Because most V4L2 drivers do not allow changing this control while
        the stream is active, we close and re-open the device internally.
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
        """Compatibility with cv2.VideoCapture API."""
        return self._ctx is not None

    def release(self):
        """Close the camera."""
        if self._ctx:
            _lib.v4l_cam_close(self._ctx)
            self._ctx = None

    def __del__(self):
        self.release()

    # ── internals ────────────────────────────────────────────────

    def _scan_sensor(self, i2c_bus: int = 0) -> SensorSetting:
        """Scan I2C bus for known sensors."""
        for setting in sensor_list:
            if setting.check_i2c(i2c_bus):
                return setting
        return gc2093  # fallback

    def _calc_v4l2_size(self, width: int, height: int):
        """
        Calculate the V4L2 capture resolution that preserves the sensor's
        native aspect ratio (16:9).  When the user requests e.g. 640x480 (4:3)
        we actually open a 16:9 resolution that covers the requested area,
        then crop + resize in read() to avoid distortion.
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


