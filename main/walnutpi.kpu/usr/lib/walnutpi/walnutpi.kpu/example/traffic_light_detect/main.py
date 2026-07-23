"""
交通信号灯检测测试脚本
使用本地图片测试 TRAFFIC_LIGHT_DETECT 类
"""
import sys
import os
import numpy as np

# img_path = "./traffic.jpg"
img_path = "./test.jpg"

import cv2
from walnutpi_kpu.TRAFFIC_LIGHT_DETECT import TRAFFIC_LIGHT_DETECT

# 读取图片
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()
print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化交通信号灯检测器（构造时会打印一次初始化提示信息）
detector = TRAFFIC_LIGHT_DETECT()          # 默认使用自带的640模型
# detector = TRAFFIC_LIGHT_DETECT(size=640) # 指定模型尺寸
# detector = TRAFFIC_LIGHT_DETECT(kmodel_path="./my_model.kmodel", size=640) # 自定义模型

# 执行检测
results = detector.run(img, reliability_threshold=0.5, nms_threshold=0.45)

print(f"\n推理耗时: {detector.speed.ms_inference:.1f}ms")
print(f"后处理耗时: {detector.speed.ms_post_process:.1f}ms")
print(f"\n检测到 {len(results)} 个交通信号灯")

# 类别颜色（BGR）
label_colors = {
    0: (0, 0, 255),    # red    红
    1: (0, 255, 0),    # green  绿
    2: (0, 255, 255),  # yellow 黄
}

# 打印并绘制结果
for result in results:
    color = label_colors.get(result.label, (255, 255, 255))
    # 绘制 检测框
    cv2.rectangle(img, (result.x, result.y),
                  (result.x + result.w, result.y + result.h),
                  color, 2)

    # 绘制 类别 + 置信度
    label = f"{result.label_name} {result.reliability:.2f}"
    cv2.putText(img, label, (result.x, result.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 保存结果图片
output_path = "./.result.jpg"
cv2.imwrite(output_path, img)
print(f"\n结果已保存到: {output_path}")
