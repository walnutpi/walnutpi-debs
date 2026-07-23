"""
口罩检测，检测人脸上是否有口罩

用法:
    python main.py                          # 使用默认测试图片
    python main.py /path/to/image.jpg       # 指定图片路径
"""
import sys
import os
import cv2
from walnutpi_kpu.FACE_MASK import FACE_MASK


def draw_results(img, results):
    """在图片上绘制人脸框和口罩检测结果"""
    for r in results:
        # 戴口罩概率 >= 0.5 → 绿色框，否则红色框
        is_mask = r.mask >= 0.5
        color = (0, 255, 0) if is_mask else (0, 0, 255)
        label = f"mask: {r.mask:.2f}" if is_mask else f"no mask: {r.mask:.2f}"

        # ---- 人脸框 ----
        cv2.rectangle(img, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 2)

        # ---- 标签 ----
        cv2.putText(img, label,
                    (r.x, max(r.y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- 5 个关键点 ----
        keypoint_colors = [
            (0, 0, 255),    # 左眼 - 红色
            (0, 255, 255),  # 右眼 - 青色
            (255, 0, 255),  # 鼻子 - 品红
            (0, 255, 0),    # 左嘴角 - 绿色
            (255, 0, 0),    # 右嘴角 - 蓝色
        ]
        keypoints = [
            (r.left_eye.x, r.left_eye.y),
            (r.right_eye.x, r.right_eye.y),
            (r.nose.x, r.nose.y),
            (r.left_mouth.x, r.left_mouth.y),
            (r.right_mouth.x, r.right_mouth.y),
        ]
        for (px, py), kc in zip(keypoints, keypoint_colors):
            cv2.circle(img, (px, py), 3, kc, -1)


def main():
    # ---- 图片路径 ----
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = os.path.join(os.path.dirname(__file__), "test.jpg")

    if not os.path.exists(img_path):
        print(f"图片不存在: {img_path}")
        print("请放置一张测试图片 test.jpg 到当前目录，或指定图片路径作为参数")
        return

    # ---- 读取图片 ----
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    print(f"图片尺寸: {img.shape[1]}×{img.shape[0]}")

    # ---- 初始化口罩检测器（内部自动串联人脸检测 + 口罩分类） ----
    detector = FACE_MASK(det_size=320)

    # ---- 一次调用完成两阶段推理 ----
    results = detector.run(img, mask_thresh=0.5, det_thresh=0.6, nms_thresh=0.4)

    print(f"\n检测到 {len(results)} 张人脸")

    for i, r in enumerate(results):
        status = "戴口罩" if r.mask >= 0.5 else "未戴口罩"
        print(f"  人脸{i + 1}: {status}, mask={r.mask:.3f}, "
              f"框=({r.x},{r.y},{r.w},{r.h}), "
              f"人脸置信度={r.reliability:.2f}")

    # ---- 绘制结果 ----
    draw_results(img, results)

    # ---- 保存结果 ----
    output_path = os.path.join(os.path.dirname(__file__), ".result.jpg")
    cv2.imwrite(output_path, img)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
