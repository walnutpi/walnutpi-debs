#!/usr/bin/env python3
"""
批量将原始帧文件转为 JPG，根据后缀名自动识别格式。

支持的格式:
  .nv12  — YUV420 semi-planar (8-bit)
  .nv21  — YUV420 semi-planar (8-bit, swapped UV)
  .nv16  — YUV422 semi-planar (8-bit)
  .bgr   — BGR888 packed (8-bit per channel)
  .rgb   — RGB888 packed (8-bit per channel)
  .ppm   — P6 binary (legacy)

分辨率从父目录名 capture_*_WxH 自动解析。

Usage:
    python3 bin2jpg.py                   # 自动发现并转换所有 capture_* 目录
    python3 bin2jpg.py <directory>       # 转换指定目录
    python3 bin2jpg.py -q 90             # JPEG 质量 (default: 95)
    python3 bin2jpg.py --dry-run         # 预览
"""

import argparse
import os
import re
import sys

import numpy as np

try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) is not installed.", file=sys.stderr)
    print("Install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)


OUTPUT_ROOT = "captures_jpg"


# ── 分辨率解析 ────────────────────────────────────────────────────────

def parse_resolution(dir_name):
    """从 '20260728_160000_480x320' 中解析 (480, 320)。"""
    m = re.search(r"_(\d+)x(\d+)$", dir_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# ── 各格式 → BGR (OpenCV 原生格式) ────────────────────────────────────

def nv12_to_bgr(data, w, h):
    """NV12 → BGR，BT.601 full-range。"""
    yuv = np.frombuffer(data, dtype=np.uint8).reshape(h + h // 2, w)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def nv21_to_bgr(data, w, h):
    """NV21 → BGR。"""
    yuv = np.frombuffer(data, dtype=np.uint8).reshape(h + h // 2, w)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def nv16_to_bgr(data, w, h):
    """NV16 → BGR。"""
    yuv = np.frombuffer(data, dtype=np.uint8).reshape(h * 2, w)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV16)


def bgr_to_bgr(data, w, h):
    """BGR888 raw → BGR（直接 reshape，V4L2 BGR24 字节序与 OpenCV 一致）。"""
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)


def rgb_to_bgr(data, w, h):
    """RGB888 raw → BGR。"""
    rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── 格式分发表 ────────────────────────────────────────────────────────

CONVERTERS = {
    "nv12": nv12_to_bgr,
    "nv21": nv21_to_bgr,
    "nv16": nv16_to_bgr,
    "bgr":  bgr_to_bgr,
    "rgb":  rgb_to_bgr,
}

RAW_EXTENSIONS = set(CONVERTERS.keys())


def convert_to_jpg(src_path, dst_path, w, h, quality):
    """单个文件 → JPG，所有路径统一输出 BGR，直接交 imwrite。"""
    ext = os.path.splitext(src_path)[1].lstrip(".").lower()

    if ext == "ppm":
        bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return False, 0
    elif ext in CONVERTERS:
        with open(src_path, "rb") as f:
            data = f.read()
        try:
            bgr = CONVERTERS[ext](data, w, h)
        except Exception as e:
            print(f"      Error converting {os.path.basename(src_path)}: {e}", file=sys.stderr)
            return False, 0
    else:
        print(f"      Unknown format: .{ext}", file=sys.stderr)
        return False, 0

    ok = cv2.imwrite(dst_path, bgr,
                     [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False, 0
    return True, os.path.getsize(dst_path) / 1024


# ── 目录扫描 / 处理 ──────────────────────────────────────────────────

def find_capture_dirs(target_dir):
    """在 target_dir 下找到所有采集子目录（支持 captures/ 及旧格式 capture_*）。"""
    if not os.path.isdir(target_dir):
        return []

    # captures/ 父目录 → 返回其子目录
    captures_dir = os.path.join(target_dir, "captures")
    if os.path.isdir(captures_dir):
        subs = sorted(
            os.path.join(captures_dir, d)
            for d in os.listdir(captures_dir)
            if os.path.isdir(os.path.join(captures_dir, d))
        )
        if subs:
            return subs

    # 兼容旧格式: 当前目录下的 capture_* 目录
    return sorted(
        os.path.join(target_dir, d)
        for d in os.listdir(target_dir)
        if d.startswith("capture_") and os.path.isdir(os.path.join(target_dir, d))
    )


def list_raw_files(dir_path):
    """列出目录下所有可转换文件（按后缀过滤），按文件名排序。"""
    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lstrip(".").lower() in RAW_EXTENSIONS | {"ppm"}
    )


def convert_capture_dir(capture_dir, out_root, quality, dry_run):
    """转换一个 capture_* 目录。"""
    dir_name = os.path.basename(capture_dir)
    resolution = parse_resolution(dir_name)
    if resolution is None:
        print(f"  Cannot parse WxH from '{dir_name}', skip.", file=sys.stderr)
        return 0, 0, 0
    w, h = resolution

    files = list_raw_files(capture_dir)
    if not files:
        print(f"  (no raw frames found, skip)")
        return 0, 0, 0

    out_dir = os.path.join(out_root, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    converted = 0
    skipped = 0
    failed = 0

    for src_path in files:
        filename = os.path.basename(src_path)
        jpg_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".jpg")

        if os.path.exists(jpg_path):
            skipped += 1
            if not dry_run:
                print(f"    [SKIP] {filename}")
            continue

        if dry_run:
            print(f"    [DRY]  {filename}  →  {os.path.basename(jpg_path)}")
            converted += 1
            continue

        ok, size_kb = convert_to_jpg(src_path, jpg_path, w, h, quality)
        if ok:
            converted += 1
            print(f"    [{converted:3d}] {filename}  →  {os.path.basename(jpg_path)}  ({size_kb:.0f} KB)")
        else:
            failed += 1
            print(f"    [FAIL] {filename}", file=sys.stderr)

    return converted, skipped, failed


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch convert raw frames (.nv12/.bgr/.ppm ...) to JPG."
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Target directory or capture_* dir (omit to auto-discover)"
    )
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="JPEG quality 0–100 (default: 95)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only")
    args = parser.parse_args()

    if args.target is None:
        capture_dirs = find_capture_dirs(".")
        if not capture_dirs:
            print("No captures/ (or capture_*) directory found.")
            print("First run v4l2-cap on the dev board, then copy captures/ here.")
            sys.exit(0)
        print(f"Found {len(capture_dirs)} capture director{'y' if len(capture_dirs) == 1 else 'ies'}:")
        for d in capture_dirs:
            print(f"  - {d}")
    elif os.path.isdir(args.target) and re.search(r"_\d+x\d+$", os.path.basename(args.target)):
        # 直接指定某个具体的采集子目录
        capture_dirs = [args.target]
    elif os.path.isdir(args.target):
        capture_dirs = find_capture_dirs(args.target)
        if not capture_dirs:
            print(f"No captures/ (or capture_*) directory found in '{args.target}'.")
            sys.exit(0)
    else:
        print(f"Error: '{args.target}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    total_c = total_s = total_f = 0
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for cap_dir in capture_dirs:
        name = os.path.basename(cap_dir)
        print(f"\n{name}/")
        c, s, f = convert_capture_dir(cap_dir, OUTPUT_ROOT, args.quality, args.dry_run)
        total_c += c
        total_s += s
        total_f += f

    print(f"\n{'='*50}")
    print(f"Output: {os.path.abspath(OUTPUT_ROOT)}/")
    print(f"Converted: {total_c},  Skipped: {total_s},  Failed: {total_f}")
    if args.dry_run:
        print("(dry-run mode, no files written)")


if __name__ == "__main__":
    main()
