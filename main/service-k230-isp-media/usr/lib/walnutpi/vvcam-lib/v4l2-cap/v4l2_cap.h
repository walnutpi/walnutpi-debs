/*
 * v4l2_cap.h — V4L2 采集模块类型与 API
 */
#ifndef V4L2_CAP_H
#define V4L2_CAP_H

#include <stddef.h>
#include <stdint.h>

#define V4L2_CAP_MAX_BUFS 6

/* ── 采集器状态 ─────────────────────────────────────────────────── */

typedef struct {
    int      fd;                 /* /dev/videoX */
    unsigned width;              /* 实际分辨率 */
    unsigned height;
    uint32_t pixelformat;        /* 实际 pixelformat */
    unsigned buf_count;          /* 已映射的 buffer 个数 */
    struct {
        void   *mmap;
        size_t  length;
        int     dma_fd;
    } bufs[V4L2_CAP_MAX_BUFS];
} v4l2_cap_t;

/* ── API ────────────────────────────────────────────────────────── */

/**
 * 完整初始化一条 V4L2 pipeline:
 *   open → QUERYCAP → G_FMT(probe) → S_CTRL(flip) → S_FMT →
 *   REQBUFS(MMAP) → mmap+QBUF → STREAMON
 *
 * hflip/vflip < 0 表示不设置（保留默认值）。
 * 成功返回 malloc 的 v4l2_cap_t*，失败返回 NULL。
 */
v4l2_cap_t *v4l2_cap_init(unsigned device, uint32_t want_fmt,
                           unsigned want_w, unsigned want_h,
                           int hflip, int vflip);

/**
 * 等待一帧，返回内存指针和字节数。
 * 成功返回 buffer 索引，data 指向 mmap 内存，bytesused 为帧大小。
 * 超时/出错返回 -1。
 */
int v4l2_cap_dequeue(v4l2_cap_t *c, int timeout_ms,
                     uint8_t **data, size_t *bytesused);

/** 归还 buffer */
int v4l2_cap_queue(v4l2_cap_t *c, int index);

/** 释放全部资源（STREAMOFF → 复位 flip → munmap → close） */
void v4l2_cap_close(v4l2_cap_t *c);

#endif /* V4L2_CAP_H */
