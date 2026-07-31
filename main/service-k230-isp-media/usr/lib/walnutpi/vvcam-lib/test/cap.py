'''
捕获图像
'''
count=20

import cv2,time
import k230_sensor

# 打开摄像头
cap = k230_sensor.Sensor(640, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

for i in range(count):
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 保存起来
    filename = f"image_{i}.jpg"
    cv2.imwrite(filename, img)
    print(f"Saved {filename}")

cap .release() # 关闭摄像头
