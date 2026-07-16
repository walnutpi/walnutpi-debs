"""
手势识别测试脚本
使用本地图片测试 HAND_KEYPOINT_CLS 类
"""
import cv2
from walnutpi_kpu.HAND_KEYPOINT_CLS import HAND_KEYPOINT_CLS, GESTURE_NAMES

# 可修改为你的测试图片路径
IMG_PATH = "./test.jpg"
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.5

# 手势颜色映射（基于编号）
GESTURE_COLORS = [
    (0, 0, 255),    # 0: fist
    (0, 255, 0),    # 1: five
    (255, 0, 0),    # 2: gun
    (255, 0, 255),  # 3: love
    (255, 255, 0),  # 4: one
    (0, 255, 255),  # 5: six
    (128, 128, 0),  # 6: three
    (128, 0, 128),  # 7: thumbUp
    (0, 128, 128),  # 8: yeah
]

# 读取图片
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"无法读取图片: {IMG_PATH}")
    exit(1)

print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化手势识别器
hkc = HAND_KEYPOINT_CLS()

# 执行检测
results = hkc.run(img, reliability_threshold=CONFIDENCE_THRESHOLD,
                   nms_threshold=NMS_THRESHOLD)

print(f"\n检测到 {len(results)} 个手掌")
for result in results:
    print(f"  {result}")

# 绘制结果
img_draw = img.copy()
for result in results:
    HAND_KEYPOINT_CLS.draw_keypoints(img_draw, result)
    gid = result.label
    color = GESTURE_COLORS[gid] if gid >= 0 else (255, 255, 255)
    name = GESTURE_NAMES[gid] if gid >= 0 else "unknown"
    cv2.putText(img_draw, name, (result.x, result.y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# 保存结果
OUTPUT_PATH = "./.result.jpg"
cv2.imwrite(OUTPUT_PATH, img_draw)
print(f"\n结果已保存至: {OUTPUT_PATH}")
