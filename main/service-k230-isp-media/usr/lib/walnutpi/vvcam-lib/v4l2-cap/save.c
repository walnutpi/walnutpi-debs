/*
 * save.c — 原始帧保存实现
 */
#include "save.h"

#include <errno.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <string.h>

#define PR(fmt, ...) fprintf(stderr, "[v4l2-cap] " fmt "\n", ##__VA_ARGS__)

/* ── pixelformat → 扩展名 ──────────────────────────────────────── */

const char *format_ext(uint32_t pixelformat)
{
    switch (pixelformat) {
    case V4L2_PIX_FMT_NV12:  return "nv12";
    case V4L2_PIX_FMT_NV16:  return "nv16";
    case V4L2_PIX_FMT_NV21:  return "nv21";
    case V4L2_PIX_FMT_BGR24: return "bgr";
    case V4L2_PIX_FMT_RGB24: return "rgb";
    default:                 return "raw";
    }
}

/* ── 保存 ───────────────────────────────────────────────────────── */

int save_raw_frame(const char *dir, int index,
                   const uint8_t *data, size_t bytesused,
                   uint32_t pixelformat)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/frame_%04d.%s",
             dir, index, format_ext(pixelformat));

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        PR("fopen %s: %s", path, strerror(errno));
        return -1;
    }

    /* 将 stdio 缓冲区设为帧大小，确保一次 write() 系统调用写出全部数据，
     * 避免默认 8KB 缓冲导致 ~760 次 write/fclose-sync 拖慢嵌入板存储。 */
    setvbuf(fp, NULL, _IOFBF, bytesused);
    fwrite(data, 1, bytesused, fp);
    fclose(fp);

    fprintf(stderr, "  [%04d] %s  (%zu bytes)\n", index, path, bytesused);
    return 0;
}
