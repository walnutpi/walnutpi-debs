/*
 * save.h — 原始帧保存模块
 */
#ifndef SAVE_H
#define SAVE_H

#include <stddef.h>
#include <stdint.h>

/** pixelformat → 文件扩展名 (不含点)，如 "nv12"、"bgr" */
const char *format_ext(uint32_t pixelformat);

/**
 * 将原始帧数据写入文件。
 * 路径: dir/frame_NNNN.ext
 * 返回 0 成功，-1 失败。
 */
int save_raw_frame(const char *dir, int index,
                   const uint8_t *data, size_t bytesused,
                   uint32_t pixelformat);

#endif /* SAVE_H */
