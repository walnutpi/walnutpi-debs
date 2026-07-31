/*
 * v4l2-cap — 从 V4L2 设备捕获帧并保存原始数据
 *
 * 用法: ./v4l2-cap [-d device] [-w width] [-h height] [-n count] [-f format]
 *       默认: -d 1 -w 1920 -h 1080 -n 6 -f NV12
 */
#include "v4l2_cap.h"
#include "save.h"

#include <errno.h>
#include <getopt.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>

#define PR(fmt, ...) fprintf(stderr, "[v4l2-cap] " fmt "\n", ##__VA_ARGS__)

/* ── 默认参数 ───────────────────────────────────────────────────── */

static unsigned g_device = 1;
static unsigned g_width  = 1920;
static unsigned g_height = 1080;
static unsigned g_count  = 6;
static uint32_t g_format = V4L2_PIX_FMT_NV12;
static int      g_hflip  = -1;
static int      g_vflip  = -1;

/* ── 命令行解析 ─────────────────────────────────────────────────── */

static uint32_t parse_fourcc(const char *s)
{
    if (!strcasecmp(s, "NV12")) return V4L2_PIX_FMT_NV12;
    if (!strcasecmp(s, "NV16")) return V4L2_PIX_FMT_NV16;
    if (!strcasecmp(s, "NV21")) return V4L2_PIX_FMT_NV21;
    if (!strcasecmp(s, "BGR24") || !strcasecmp(s, "BGR")) return V4L2_PIX_FMT_BGR24;
    if (!strcasecmp(s, "RGB24") || !strcasecmp(s, "RGB")) return V4L2_PIX_FMT_RGB24;
    PR("unknown format '%s', fallback to NV12", s);
    return V4L2_PIX_FMT_NV12;
}

static void help(const char *arg0)
{
    printf("Usage: %s [options]\n", arg0);
    printf("Options:\n"
           "  -d <num>   Video device       (default: 1)\n"
           "  -w <px>    Width              (default: 1920)\n"
           "  -h <px>    Height             (default: 1080)\n"
           "  -n <num>   Frame count        (default: 6)\n"
           "  -f <fmt>   Format: NV12|BGR24  (default: NV12)\n"
           "  --hflip N  Horizontal mirror 0/1\n"
           "  --vflip N  Vertical flip 0/1\n");
}

static int parse_cmd(int argc, char *argv[])
{
    struct option longopts[] = {
        { "hflip", required_argument, NULL, 'm' },
        { "vflip", required_argument, NULL, 'v' },
        { 0, 0, 0, 0 }
    };
    int ch;
    while ((ch = getopt_long(argc, argv, "d:w:h:n:f:m:v:", longopts, NULL)) != -1) {
        switch (ch) {
        case 'd': g_device = (unsigned)atoi(optarg); break;
        case 'w': g_width  = (unsigned)atoi(optarg); break;
        case 'h': g_height = (unsigned)atoi(optarg); break;
        case 'n': g_count  = (unsigned)atoi(optarg); break;
        case 'f': g_format = parse_fourcc(optarg);   break;
        case 'm': g_hflip  = atoi(optarg) ? 1 : 0;  break;
        case 'v': g_vflip  = atoi(optarg) ? 1 : 0;  break;
        default:
            help(argv[0]);
            return -1;
        }
    }
    return 0;
}

/* ── 创建输出目录 ───────────────────────────────────────────────── */

static int make_output_dir(char *dir, size_t dir_sz, unsigned w, unsigned h)
{
    mkdir("captures", 0755); /* 忽略已存在 */

    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    snprintf(dir, dir_sz, "captures/%04d%02d%02d_%02d%02d%02d_%ux%u",
             tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
             tm->tm_hour, tm->tm_min, tm->tm_sec, w, h);
    if (mkdir(dir, 0755) < 0) {
        PR("mkdir %s: %s", dir, strerror(errno));
        return -1;
    }
    PR("output: %s/", dir);
    return 0;
}

/* ── 主流程 ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    if (parse_cmd(argc, argv) < 0)
        return 1;

    v4l2_cap_t *cap = v4l2_cap_init(g_device, g_format,
                                     g_width, g_height,
                                     g_hflip, g_vflip);
    if (!cap) return 1;

    char dir[128];
    if (make_output_dir(dir, sizeof(dir), cap->width, cap->height) < 0) {
        v4l2_cap_close(cap);
        return 1;
    }

    unsigned captured = 0;
    for (unsigned i = 0; i < g_count; i++) {
        uint8_t *data;
        size_t   bytes;
        int idx = v4l2_cap_dequeue(cap, 3000, &data, &bytes);
        if (idx < 0) {
            PR("grab failed, stopped at frame %u/%u", i + 1, g_count);
            break;
        }
        if (save_raw_frame(dir, i, data, bytes, cap->pixelformat) == 0)
            captured++;
        v4l2_cap_queue(cap, idx);
    }

    PR("done: %u/%u frames → %s/", captured, g_count, dir);
    v4l2_cap_close(cap);
    return 0;
}
