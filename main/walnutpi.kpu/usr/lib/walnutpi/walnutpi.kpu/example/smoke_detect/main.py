"""
抽烟检测测试脚本
使用本地图片测试 SMOKE_DETECT 类
"""
import sys
import os
import numpy as np

img_path = "./test.jpg"

import cv2
from walnutpi_kpu.SMOKE_DETECT import SMOKE_DETECT

# 读取图片
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()
print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化抽烟检测器
detector = SMOKE_DETECT(kmodel_path="./my_model.kmodel", size=640) # 自定义模型

# 执行检测
results = detector.run(img, reliability_threshold=0.5, nms_threshold=0.45)

print(f"\n推理耗时: {detector.speed.ms_inference:.1f}ms")
print(f"后处理耗时: {detector.speed.ms_post_process:.1f}ms")
print(f"\n检测到 {len(results)} 个抽烟目标")

# 打印并绘制结果
for result in results:
    # 绘制抽烟检测框
    cv2.rectangle(img, (result.x, result.y),
                  (result.x + result.w, result.y + result.h),
                  (0, 0, 255), 2)

    # 绘制置信度标签
    label = f"smoke {result.reliability:.2f}"
    cv2.putText(img, label, (result.x, result.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# 保存结果图片
output_path = "./.result.jpg"
cv2.imwrite(output_path, img)
print(f"\n结果已保存到: {output_path}")
