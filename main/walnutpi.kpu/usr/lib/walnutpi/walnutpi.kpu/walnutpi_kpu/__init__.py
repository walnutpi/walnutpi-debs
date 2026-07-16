from typing import Literal, Dict, Any
import importlib
import numpy as np
import queue
import threading
import time
import cv2

# 支持多个版本
NNCASEVersionType = Literal["2.10","2.11"]


def get_nncase(nncase_version: NNCASEVersionType) -> Any:
    """
    获取指定版本的nncase库
    """
    module_name_map: Dict[NNCASEVersionType, str] = {
        "2.10": "nncase_2_10",
        "2.11": "nncase_2_11",
    }

    module_name = module_name_map.get(nncase_version)
    if not module_name:
        available_versions = list(module_name_map.keys())
        raise ValueError(
            f"Unsupported nncase version: {nncase_version}. "
            f"Available versions: {available_versions}"
        )

    try:
        nncase_lib = importlib.import_module(
            f".{module_name}", package=__name__.rsplit(".", 1)[0]
        )
        return nncase_lib
    except ImportError as e:
        raise ImportError(f"Failed to import .{module_name} from {__name__}: {e}")


class KPU_BASE:
    has_result = False
    is_running = False
    model_size: int
    nms_threshold = 0.45  # nms阈值
    results:Any
    thread = None

    class _speed:
        ms_post_process: float = 0  # 后处理耗时
        ms_inference: float = 0  # 推理耗时

    speed = _speed()

    def __init__(
        self, kmodel_path: str, size: int, nncase_version: NNCASEVersionType = "2.11"
    ):
        """
        初始化
        @kmodel_path: 模型路径
        @size: 模型输入尺寸
        @nncase_version: nncase版本
        """
        self.model_h = size
        self.model_w = size

        self.nn = get_nncase(nncase_version)
        self.kpu = self.nn.Interpreter()
        self.ai2d = self.nn.AI2D()

        self.kpu.load_model(kmodel_path)
        # 创建一个临时输入张量用于绑定输入
        tmp_tensor = self.nn.RuntimeTensor.from_numpy(
            np.ones((1, 3, self.model_h, self.model_w), dtype=np.uint8)
        )

        self.kpu.set_input_tensor(0, tmp_tensor)
        # 创建任务队列和工作线程
        self._task_queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    # AI2D 配置参数(子类可覆盖)
    ai2d_pad_color = [114, 114, 114]  # 填充颜色
    
    ai2d_2d_w = -1
    ai2d_2d_h = -1

    def ai2d_init(self, model_w, model_h, img_w, img_h):
        """
        初始化AI2D
        @model_w: 模型输入尺寸,宽度
        @model_h: 模型输入尺寸,高度
        @img_w: 图片宽度
        @img_h: 图片高度
        """

        if self.ai2d_2d_w == model_w and self.ai2d_2d_h == model_h:
            return
        self.ai2d_2d_w, self.ai2d_2d_h = model_w, model_h

        # 计算输入图像缩放比例，保持纵横比
        self.ratio = min(model_w / img_w, model_h / img_h)
        new_w, new_h = int(img_w * self.ratio), int(img_h * self.ratio)
        dw, dh = (model_w - new_w), (model_h - new_h)
        pad_left, pad_right = 0, int(dw)
        pad_top, pad_bottom = 0, int(dh)

        # ===============================
        # 配置 AI2D 预处理流水线
        # ===============================
        self.ai2d.set_datatype(
            self.nn.AI2D_FORMAT.NCHW_FMT,  # 输入格式
            self.nn.AI2D_FORMAT.NCHW_FMT,  # 输出格式
            np.uint8,
            np.uint8,  # 输入输出数据类型
        )
        # 设置 resize 参数（使用 tf_bilinear 双线性插值）
        self.ai2d.set_resize_param(
            True,
            self.nn.AI2D_INTERP_METHOD.tf_bilinear,
            self.nn.AI2D_INTERP_MODE.half_pixel,
        )
        # 设置 padding 参数（补边）- 使用可配置的填充颜色
        self.ai2d.set_pad_param(
            True,
            [0, 0, 0, 0, pad_top, pad_bottom, pad_left, pad_right],
            0,
            self.ai2d_pad_color,  # 使用类属性配置填充色
        )
        # 构建 AI2D pipeline（输入、输出 shape）
        self.ai2d.build([1, 3, img_h, img_w], [1, 3, model_h, model_w])

    def run(self, img, reliability_threshold=0.5, nms_threshold=0.5):
        """
        检测图片，阻塞直到检测完成，返回检测结果
        @img: 图片
        @reliability_threshold: 置信度阈值
        @nms_threshold: nms阈值
        """
        self.is_running = True
        self.has_result = False

        time_point = time.time() * 1000

        try:
            self.img_w, self.img_h = img.shape[1], img.shape[0]
            self.ai2d_init(self.model_w, self.model_h, self.img_w, self.img_h)

            self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_nchw = np.array([self.img_rgb.transpose((2, 0, 1))])  # 转换为 NCHW

            # -------------------------------
            # 执行 AI2D 预处理（resize + pad）
            # -------------------------------
            ai2d_input_tensor = self.nn.RuntimeTensor.from_numpy(img_nchw)
            kpu_input_tensor = self.kpu.get_input_tensor(0)
            self.ai2d.run(ai2d_input_tensor, kpu_input_tensor)

            # -------------------------------
            # 模型推理
            # -------------------------------
            self.kpu.run()

            self.speed.ms_inference = time.time() * 1000 - time_point
            time_point = time.time() * 1000

            self.results = self.post_process(reliability_threshold, nms_threshold)
            self.speed.ms_post_process = time.time() * 1000 - time_point
            time_point = time.time() * 1000
        except Exception as e:
            import traceback
            traceback.print_exc()

        self.has_result = True
        self.is_running = False
        return self.results

    def run_async(self, img, reliability_threshold=0.5, nms_threshold=0.5):
        """
        检测图片，立即返回，不阻塞
        @img: 图片路径或图像数据
        @reliability_threshold: 置信度阈值
        @nms_threshold: NMS阈值
        """
        if not self.is_running:
            self.nms_threshold = nms_threshold
            self.is_running = True
            # 将任务放入队列
            self._task_queue.put((img, reliability_threshold, nms_threshold))
        else:
            print("模型正在运行中，请等待当前任务完成")

    def _worker_loop(self):
        """工作线程,检测信号随时启动异步任务"""
        while not self._shutdown_event.is_set():
            try:
                # 等待任务，超时后检查是否需要退出
                task_data = self._task_queue.get(timeout=0.1)
                if task_data is None:  # 停止信号
                    break

                img, reliability_threshold, nms_threshold = task_data
                self.thread_async_run(img, reliability_threshold, nms_threshold)
                self._task_queue.task_done()
            except queue.Empty:
                continue  # 超时继续检查退出信号
            except Exception:
                if not self._task_queue.empty():
                    self._task_queue.task_done()

    def thread_async_run(self, img, reliability_threshold, nms_threshold):
        """线程异步任务"""
        try:
            self.run(img, reliability_threshold, nms_threshold)
        except:
            pass

    def get_result(self):
        self.has_result = False
        return self.results

    def __del__(self):
        """析构函数，清理线程"""
        # 使用 getattr 安全地访问属性，如果属性不存在则返回 None
        shutdown_event = getattr(self, "_shutdown_event", None)
        worker_thread = getattr(self, "_worker_thread", None)

        if shutdown_event:
            shutdown_event.set()

        # 等待工作线程结束
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=1.0)  # 设置超时避免无限等待
