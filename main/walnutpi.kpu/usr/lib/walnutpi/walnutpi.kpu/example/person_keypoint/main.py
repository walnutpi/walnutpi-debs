"""
人体关键点检测测试脚本
使用本地图片测试 PERSON_KEYPOINT 类
"""
import sys
import os
import numpy as np

# 可修改为你的测试图片路径
img_path = "./test.jpg"

import cv2
from walnutpi_kpu.PERSON_KEYPOINT import PERSON_KEYPOINT

# 读取图片
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()
print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化人体关键点检测器
detector = PERSON_KEYPOINT()          # 默认使用自带的320模型
# detector = PERSON_KEYPOINT(kmodel_path="./my_model.kmodel", size=320) # 自定义模型

# 执行检测
results = detector.run(img, reliability_threshold=0.3, nms_threshold=0.5)

print(f"\n推理耗时: {detector.speed.ms_inference:.1f}ms")
print(f"后处理耗时: {detector.speed.ms_post_process:.1f}ms")
print(f"\n检测到 {len(results)} 个人")

# 绘制结果
for result in results:
    print(f"  {result}")

    # 绘制 bbox
    cv2.rectangle(img,
                  (result.x, result.y),
                  (result.x + result.w, result.y + result.h),
                  (0, 255, 0), 2)

    # 绘制骨架连线（带颜色）
    SKELETON = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11),  (6, 12),  (5, 6),
        (5, 7),   (6, 8),   (7, 9),   (8, 10),
        (1, 2),   (0, 1),   (0, 2),   (1, 3),   (2, 4),
        (3, 5),   (4, 6),
    ]
    for limb_idx, (a, b) in enumerate(SKELETON):
        kp_a = result.keypoints[a]
        kp_b = result.keypoints[b]
        if kp_a.confidence > 0.5 and kp_b.confidence > 0.5:
            color = detector.LIMB_COLORS[limb_idx]
            cv2.line(img, (kp_a.x, kp_a.y), (kp_b.x, kp_b.y),
                     color, 2)

    # 绘制 17 个关键点（不同颜色）
    for k, kp in enumerate(result.keypoints):
        if kp.confidence > 0.5:
            color = detector.KPS_COLORS[k]
            cv2.circle(img, (kp.x, kp.y), 4, color, -1)

# 保存结果图片
output_path = "./.result.jpg"
cv2.imwrite(output_path, img)
print(f"\n结果已保存到: {output_path}")
