"""
OCR 检测+识别测试脚本
检测文字区域并识别文字内容
"""
import cv2
from walnutpi_kpu.OCR import OCR

IMG_PATH = "./test.jpg"
MASK_THRESHOLD = 0.25
BOX_THRESHOLD = 0.3

img = cv2.imread(IMG_PATH)
if img is None:
    print(f"无法读取图片: {IMG_PATH}")
    exit(1)

print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化 OCR（检测+识别）
ocr = OCR()

# 执行检测识别
results = ocr.run(img, 0.7)

print(f"\n检测到 {len(results)} 个文字区域")
for r in results:
    print(f"  {r}")

# 绘制结果
img_draw = img.copy()
for r in results:
    poly = r.polygon
    # 绘制四边形
    for j in range(4):
        x1, y1 = poly[j]
        x2, y2 = poly[(j+1)%4]
        cv2.line(img_draw, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 255, 0), 2)
    # 外接矩形 + 识别文字
    cv2.rectangle(img_draw, (r.x, r.y), (r.x+r.w, r.y+r.h),
                  (0, 0, 255), 1)
    cv2.putText(img_draw, f"'{r.text}'", (r.x, max(0, r.y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

OUTPUT_PATH = "./.result.jpg"
cv2.imwrite(OUTPUT_PATH, img_draw)
print(f"\n结果已保存至: {OUTPUT_PATH}")
