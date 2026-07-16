"""
手掌检测测试脚本
使用本地图片测试 HAND_DETECT 类
"""
import sys
import os
import numpy as np

# 可修改为你的测试图片路径
img_path = "./test.jpg"

import cv2
from walnutpi_kpu.HAND_DETECT import HAND_DETECT

# 读取图片
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()
print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化手掌检测器
detector = HAND_DETECT()  # 默认使用自带的512模型
# detector = HAND_DETECT(kmodel_path="./hand_det.kmodel", size=512) # 自定义模型

# 执行检测
results = detector.run(img, reliability_threshold=0.2, nms_threshold=0.5)

print(f"\n推理耗时: {detector.speed.ms_inference:.1f}ms")
print(f"后处理耗时: {detector.speed.ms_post_process:.1f}ms")
print(f"\n检测到 {len(results)} 个手掌")

# 打印并绘制结果
for result in results:
    print(f"  {result}")

    # 绘制手掌检测框
    cv2.rectangle(img,
                  (result.x, result.y),
                  (result.x + result.w, result.y + result.h),
                  (255, 0, 255), 2)

    # 绘制置信度标签
    label = f"hand {result.reliability:.2f}"
    cv2.putText(img, label, (result.x, result.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# 保存结果图片
output_path = "./.result.jpg"
cv2.imwrite(output_path, img)
print(f"\n结果已保存至: {output_path}")
