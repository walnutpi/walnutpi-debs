"""
跌倒检测测试脚本
使用本地图片测试 FALL_DETECT 类
"""
import cv2
from walnutpi_kpu.FALL_DETECT import FALL_DETECT

# 可修改为你的测试图片路径
IMG_PATH = "./test.jpg"
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45

# 读取图片
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"无法读取图片: {IMG_PATH}")
    exit(1)

print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化跌倒检测器
detector = FALL_DETECT()
# detector = FALL_DETECT(kmodel_path="./yolov5n-falldown.kmodel")

# 执行检测
results = detector.run(img, reliability_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD)

print(f"\n推理耗时: {detector.speed.ms_inference:.1f}ms")
print(f"后处理耗时: {detector.speed.ms_post_process:.1f}ms")
print(f"\n检测到 {len(results)} 个目标")
for result in results:
    print(f"  {result}")

# 绘制结果
LABELS = ["Fall", "NoFall"]
COLORS = [(0, 0, 255), (0, 255, 0)]

for result in results:
    color = COLORS[result.label] if result.label < len(COLORS) else (255, 0, 255)

    cv2.rectangle(img,
                  (result.x, result.y),
                  (result.x + result.w, result.y + result.h),
                  color, 2)

    label_text = f"{LABELS[result.label]} {result.reliability:.2f}"
    cv2.putText(img, label_text, (result.x, result.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 保存结果图片
OUTPUT_PATH = "./.result.jpg"
cv2.imwrite(OUTPUT_PATH, img)
print(f"\n结果已保存至: {OUTPUT_PATH}")
