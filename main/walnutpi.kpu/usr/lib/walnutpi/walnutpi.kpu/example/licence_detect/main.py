"""
车牌检测+识别测试脚本
使用本地图片测试 LICENCE_DETECT 类
"""
import cv2
from walnutpi_kpu.LICENCE_DETECT import LICENCE_DETECT

# 可修改为你的测试图片路径
img_path = "./test.jpg"

# 读取图片
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()
print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化车牌识别器
lpr = LICENCE_DETECT()                              # 默认使用自带模型
# lpr = LICENCE_DETECT(det_kmodel="./my_det.kmodel",
#                       rec_kmodel="./my_rec.kmodel") # 自定义模型

# 执行识别（检测 + OCR）
results = lpr.run(img, obj_thresh=0.5, nms_thresh=0.4)

print(f"\n检测到 {len(results)} 个车牌:")
for r in results:
    print(f"{r.reliability:.2f}: {r.text}  {r.corners}")

    # 绘制四边形边框
    pts = r.corners
    cv2.line(img, pts[0], pts[1], (0, 0, 255), 2)
    cv2.line(img, pts[1], pts[2], (0, 0, 255), 2)
    cv2.line(img, pts[2], pts[3], (0, 0, 255), 2)
    cv2.line(img, pts[3], pts[0], (0, 0, 255), 2)

    # 绘制角点
    for k in range(4):
        cv2.circle(img, tuple(pts[k]), 3, (0, 255, 0), -1)

    # 绘制识别文本
    cv2.putText(img, r.text, (pts[0][0], max(pts[0][1]- 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 保存结果图片
output_path = "./.result.jpg"
cv2.imwrite(output_path, img)
print(f"\n结果已保存到: {output_path}")
