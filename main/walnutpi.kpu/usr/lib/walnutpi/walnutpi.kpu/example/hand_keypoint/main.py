"""
手掌关键点检测测试脚本
使用本地图片测试 HAND_KEYPOINT 类
"""
import cv2
from walnutpi_kpu.HAND_KEYPOINT import HAND_KEYPOINT

# 可修改为你的测试图片路径
IMG_PATH = "./test.jpg"
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.5

# 读取图片
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"无法读取图片: {IMG_PATH}")
    exit(1)

print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化手掌关键点检测器
hk = HAND_KEYPOINT()

# 执行检测
results = hk.run(img, reliability_threshold=CONFIDENCE_THRESHOLD,
                  nms_threshold=NMS_THRESHOLD)

print(f"\n检测到 {len(results)} 个手掌")
for result in results:
    print(f"  {result}")
    print(f"    关键点数: {len(result.keypoints)}")

# 绘制结果
img_draw = img.copy()
for result in results:
    HAND_KEYPOINT.draw_keypoints(img_draw, result)

# 保存结果
OUTPUT_PATH = "./.result.jpg"
cv2.imwrite(OUTPUT_PATH, img_draw)
print(f"\n结果已保存至: {OUTPUT_PATH}")
