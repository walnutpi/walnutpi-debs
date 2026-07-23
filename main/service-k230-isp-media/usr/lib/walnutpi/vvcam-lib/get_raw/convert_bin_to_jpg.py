#!/usr/bin/env python3
"""
将 NV12 格式的 .bin 文件转换为 JPG 图片
"""

import os
import glob
import numpy as np
import cv2


def nv12_to_bgr(nv12_data, width, height):
    """
    将 NV12 格式的原始数据转换为 BGR 图像
    
    参数:
        nv12_data: NV12 格式的字节数据
        width: 图像宽度
        height: 图像高度
    
    返回:
        BGR 格式的 numpy 数组
    """
    # NV12 格式: Y 平面 + UV 交错平面
    # Y 平面大小: width * height
    # UV 平面大小: width * height / 2 (每个像素占1字节，但只有宽高的一半)
    
    y_size = width * height
    uv_size = y_size // 2
    
    # 提取 Y 和 UV 分量
    y_plane = np.frombuffer(nv12_data[:y_size], dtype=np.uint8).reshape((height, width))
    uv_plane = np.frombuffer(nv12_data[y_size:y_size+uv_size], dtype=np.uint8).reshape((height//2, width//2, 2))
    
    # 合并为 NV12 格式（OpenCV 需要这种格式）
    nv12_image = np.zeros((height + height//2, width), dtype=np.uint8)
    nv12_image[:height, :] = y_plane
    nv12_image[height:, :] = uv_plane.reshape((height//2, width))
    
    # 使用 OpenCV 转换 NV12 到 BGR
    bgr_image = cv2.cvtColor(nv12_image, cv2.COLOR_YUV2BGR_NV12)
    
    return bgr_image


def convert_bin_to_jpg(bin_file_path, output_dir=None):
    """
    将单个 .bin 文件转换为 JPG
    
    参数:
        bin_file_path: .bin 文件路径
        output_dir: 输出目录，默认为 None（与输入文件同目录）
    
    返回:
        输出 JPG 文件路径
    """
    if output_dir is None:
        output_dir = os.path.dirname(bin_file_path) or '.'
    
    # 从文件名解析宽度和高度
    # 文件名格式: frame_0_1280x960.bin
    basename = os.path.basename(bin_file_path)
    
    # 尝试从文件名中提取宽度和高度
    try:
        # 查找 "WxH" 格式
        import re
        match = re.search(r'(\d+)x(\d+)', basename)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
        else:
            print(f"警告: 无法从文件名 {basename} 中解析分辨率，使用默认值 1280x960")
            width = 1280
            height = 960
    except Exception as e:
        print(f"警告: 解析文件名失败: {e}，使用默认值 1280x960")
        width = 1280
        height = 960
    
    # 读取 .bin 文件
    with open(bin_file_path, 'rb') as f:
        nv12_data = f.read()
    
    # 验证数据大小
    expected_size = width * height * 3 // 2  # NV12 格式: Y + UV = 1.5倍像素数
    if len(nv12_data) != expected_size:
        print(f"警告: 文件大小 {len(nv12_data)} 与预期 {expected_size} 不符")
        print(f"      文件: {bin_file_path}")
        # 尝试继续处理
    
    # 转换为 BGR
    try:
        bgr_image = nv12_to_bgr(nv12_data, width, height)
        
        # 生成输出文件名
        jpg_filename = os.path.splitext(basename)[0] + '.jpg'
        jpg_path = os.path.join(output_dir, jpg_filename)
        
        # 保存为 JPG
        cv2.imwrite(jpg_path, bgr_image)
        print(f"已转换: {basename} -> {jpg_filename}")
        
        return jpg_path
        
    except Exception as e:
        print(f"错误: 转换文件 {basename} 时出错: {e}")
        return None


def main():
    """主函数：转换当前目录下所有 .bin 文件"""
    
    # 获取当前目录下所有 .bin 文件
    bin_files = glob.glob('*.bin')
    
    if not bin_files:
        print("当前目录下没有找到 .bin 文件")
        return
    
    print(f"找到 {len(bin_files)} 个 .bin 文件")
    print("=" * 60)
    
    # 创建输出目录（可选）
    output_dir = '.'
    
    success_count = 0
    fail_count = 0
    
    for bin_file in sorted(bin_files):
        result = convert_bin_to_jpg(bin_file, output_dir)
        if result:
            success_count += 1
        else:
            fail_count += 1
    
    print("=" * 60)
    print(f"转换完成: 成功 {success_count} 个, 失败 {fail_count} 个")


if __name__ == '__main__':
    main()
