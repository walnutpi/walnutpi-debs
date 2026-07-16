"""
人脸检测测试脚本
使用本地图片测试 FACE_DETECT 类
"""
import sys
import os
import numpy as np

# img_path = "./fd.png"
img_path = "./test.jpg"

import cv2
from walnutpi_kpu.FACE_DETECT import FACE_DETECT

# 读取图片
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()
print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化人脸检测器
detector = FACE_DETECT() #默认不加使用320的模型
# detector = FACE_DETECT(size=640) #指定使用640的模型

# 执行检测
results = detector.run(img, reliability_threshold=0.6, nms_threshold=0.9)

print(f"\n推理耗时: {detector.speed.ms_inference:.1f}ms")
print(f"后处理耗时: {detector.speed.ms_post_process:.1f}ms")
print(f"\n检测到 {len(results)} 张人脸")

# 打印并绘制结果
for result in results:

    # 绘制 人脸识别框
    cv2.rectangle(img, (result.x, result.y), 
                    (result.x + result.w, result.y + result.h), 
                    (0, 255, 0), 2)
    
    # 绘制置信度
    label = f"{result.reliability:.2f}"
    cv2.putText(img, label, (result.x, result.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制 5 个关键点
    cv2.circle(img, (result.left_eye.x, result.left_eye.y), 4, (0, 0, 255), -1) # 左眼
    cv2.circle(img, (result.right_eye.x, result.right_eye.y), 4, (0, 255, 255), -1) # 右眼
    cv2.circle(img, (result.nose.x, result.nose.y), 4, (255, 0, 255), -1) # 鼻子
    cv2.circle(img, (result.left_mouth.x, result.left_mouth.y), 4, (0, 255, 0), -1) # 左嘴角
    cv2.circle(img, (result.right_mouth.x, result.right_mouth.y), 4, (255, 0, 0), -1) # 右嘴角

# 保存结果图片
output_path = "./.result.jpg"
cv2.imwrite(output_path, img)
print(f"\n结果已保存到: {output_path}")


