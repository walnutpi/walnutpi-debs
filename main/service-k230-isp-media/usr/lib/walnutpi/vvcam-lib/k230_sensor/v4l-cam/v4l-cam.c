#include "v4l-cam.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <unistd.h>

#define MAX_BUFS 10

struct v4l_cam_ctx {
    int      fd;
    int      width;
    int      height;
    int      buf_count;
    bool     is_bgr;         /* true = BGR24 from ISP, false = NV12 */
    bool     setup_done;     /* REQBUFS + mmap + QBUF completed */
    bool     streaming;      /* VIDIOC_STREAMON has been called */
    struct {
        void  *mmap;
        size_t length;
    } bufs[MAX_BUFS];
    uint8_t *output;         /* BGR output buffer, size = width * height * 3 */
};

/* ─── helpers ─────────────────────────────────────────────────────── */

static int xioctl(int fd, unsigned long request, void *arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static int v4l_cam_set_ctrl(int fd, uint32_t id, int value)
{
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id    = id;
    ctrl.value = value;
    return xioctl(fd, VIDIOC_S_CTRL, &ctrl);
}

/* ─── NV12 → BGR  full-range conversion ──────────────────────────── */

/* BT.601 full-range coefficients × 256 (fixed-point 16.8) */
#define VR  359    /*  1.402   * 256 */
#define UG   88    /*  0.34414 * 256 */
#define VG  183    /*  0.71414 * 256 */
#define UB  454    /*  1.772   * 256 */

static inline uint8_t clamp(int v)
{
    if (v < 0)  return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static void nv12_to_bgr_full(const uint8_t *y_plane,
                             const uint8_t *uv_plane,
                             int width, int height,
                             uint8_t *bgr)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int Y = y_plane[y * width + x];
            int uv_off = (y / 2) * width + (x & ~1);
            int U = uv_plane[uv_off];
            int V = uv_plane[uv_off + 1];

            int r = (Y * 256 + VR * (V - 128)) >> 8;
            int g = (Y * 256 - UG * (U - 128) - VG * (V - 128)) >> 8;
            int b = (Y * 256 + UB * (U - 128)) >> 8;

            *bgr++ = clamp(b);
            *bgr++ = clamp(g);
            *bgr++ = clamp(r);
        }
    }
}

/* ─── lazy setup (called on first read, after controls are set) ──── */

static int lazy_setup(v4l_cam_ctx_t *ctx)
{
    struct v4l2_requestbuffers req;
    struct v4l2_buffer buf;

    /* 1. REQBUFS — release any previously allocated buffers first,
     *    then request fresh ones.  The REQBUFS(count=0) step gives
     *    the ISP driver a clean slate so dequeue won't see stale
     *    state after a reopen cycle. */
    memset(&req, 0, sizeof(req));
    req.count  = 0;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(ctx->fd, VIDIOC_REQBUFS, &req);  /* ignore failure — best-effort */

    memset(&req, 0, sizeof(req));
    req.count  = MAX_BUFS;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(ctx->fd, VIDIOC_REQBUFS, &req) < 0) {
        fprintf(stderr, "[v4l-cam] VIDIOC_REQBUFS(%d) failed: %s\n",
                MAX_BUFS, strerror(errno));
        return -1;
    }
    ctx->buf_count = req.count;

    /* 2. allocate output buffer */
    ctx->output = malloc(ctx->width * ctx->height * 3);
    if (!ctx->output) {
        fprintf(stderr, "[v4l-cam] malloc output buffer failed: %s\n",
                strerror(errno));
        return -1;
    }

    /* 3. mmap + QBUF for each buffer */
    for (int i = 0; i < ctx->buf_count; i++) {
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (xioctl(ctx->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "[v4l-cam] VIDIOC_QUERYBUF(%d) failed: %s\n",
                    i, strerror(errno));
            return -1;
        }

        ctx->bufs[i].mmap = mmap(NULL, buf.length,
                                 PROT_READ | PROT_WRITE, MAP_SHARED,
                                 ctx->fd, buf.m.offset);
        if (ctx->bufs[i].mmap == MAP_FAILED) {
            fprintf(stderr, "[v4l-cam] mmap(%d) failed: %s\n",
                    i, strerror(errno));
            ctx->bufs[i].mmap = NULL;
            return -1;
        }
        ctx->bufs[i].length = buf.length;

        if (xioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
            fprintf(stderr, "[v4l-cam] VIDIOC_QBUF(%d) failed: %s\n",
                    i, strerror(errno));
            return -1;
        }
    }

    /* 4. STREAMON */
    {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(ctx->fd, VIDIOC_STREAMON, &type) < 0) {
            fprintf(stderr, "[v4l-cam] VIDIOC_STREAMON failed: %s\n",
                    strerror(errno));
            return -1;
        }
    }
    ctx->streaming = true;

    ctx->setup_done = true;
    return 0;
}

/* ─── public API ──────────────────────────────────────────────────── */

v4l_cam_ctx_t* v4l_cam_open(int device, int width, int height)
{
    v4l_cam_ctx_t *ctx;
    struct v4l2_format fmt;
    char dev_path[32];

    /* 1. allocate context */
    ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "[v4l-cam] calloc ctx failed: %s\n", strerror(errno));
        return NULL;
    }
    ctx->fd = -1;

    /* 2. open device */
    snprintf(dev_path, sizeof(dev_path), "/dev/video%d", device);
    ctx->fd = open(dev_path, O_RDWR | O_NONBLOCK);
    if (ctx->fd < 0) {
        fprintf(stderr, "[v4l-cam] open %s failed: %s\n", dev_path, strerror(errno));
        goto fail_free_ctx;
    }

    /* 3. Try BGR24 first (ISP hardware output, zero-copy to Python),
     *    fall back to NV12 + software conversion */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) < 0) {
        fprintf(stderr, "[v4l-cam] VIDIOC_G_FMT failed: %s\n", strerror(errno));
        goto fail_close;
    }

    /* try BGR24 */
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
    fmt.fmt.pix.width       = width;
    fmt.fmt.pix.height      = height;
    if (xioctl(ctx->fd, VIDIOC_S_FMT, &fmt) == 0) {
        ctx->is_bgr = true;
        ctx->width  = fmt.fmt.pix.width;
        ctx->height = fmt.fmt.pix.height;
        fprintf(stderr, "[v4l-cam] %s opened, BGR24 %dx%d (setup deferred)\n",
                dev_path, ctx->width, ctx->height);
    } else {
        /* fallback: NV12 */
        fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
        fmt.fmt.pix.width       = width;
        fmt.fmt.pix.height      = height;
        if (xioctl(ctx->fd, VIDIOC_S_FMT, &fmt) < 0) {
            fprintf(stderr, "[v4l-cam] VIDIOC_S_FMT(NV12,%dx%d) failed: %s\n",
                    width, height, strerror(errno));
            goto fail_close;
        }
        ctx->is_bgr = false;
        ctx->width  = fmt.fmt.pix.width;
        ctx->height = fmt.fmt.pix.height;
        fprintf(stderr, "[v4l-cam] %s opened, NV12 %dx%d (setup deferred)\n",
                dev_path, ctx->width, ctx->height);
    }

    /* 4. REQBUFS + QBUF + STREAMON deferred to first read(),
     *    so that V4L2 controls (hflip/vflip) can be set
     *    between open() and the first read(). */
    return ctx;

fail_close:
    close(ctx->fd);
    ctx->fd = -1;
fail_free_ctx:
    free(ctx);
    return NULL;
}

void v4l_cam_close(v4l_cam_ctx_t* ctx)
{
    if (!ctx) return;

    if (ctx->fd >= 0) {
        /* Drain: wait briefly for any in-flight frame, then DQBUF
         * and immediately re-queue.  This lets the driver finish
         * its current transfer before STREAMOFF without starving
         * the hardware of buffers.  We drain at most buf_count
         * buffers to avoid an infinite loop. */
        if (ctx->streaming && ctx->buf_count > 0) {
            struct pollfd pf;
            pf.fd      = ctx->fd;
            pf.events  = POLLIN | POLLPRI;
            pf.revents = 0;

            for (int i = 0; i < ctx->buf_count; i++) {
                if (poll(&pf, 1, 100) <= 0)
                    break;  /* no more frames pending */
                struct v4l2_buffer buf;
                memset(&buf, 0, sizeof(buf));
                buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;
                if (xioctl(ctx->fd, VIDIOC_DQBUF, &buf) < 0)
                    break;
                /* re-queue immediately — keep the buffer pool
                 * intact so the ISP driver stays in a clean state */
                xioctl(ctx->fd, VIDIOC_QBUF, &buf);
            }
        }

        if (ctx->streaming) {
            int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(ctx->fd, VIDIOC_STREAMOFF, &type);
        }

        for (int i = 0; i < ctx->buf_count; i++) {
            if (ctx->bufs[i].mmap) {
                munmap(ctx->bufs[i].mmap, ctx->bufs[i].length);
                ctx->bufs[i].mmap = NULL;
            }
        }
        close(ctx->fd);
        ctx->fd = -1;
    }

    free(ctx->output);
    ctx->output = NULL;
    free(ctx);
}

int v4l_cam_read(v4l_cam_ctx_t* ctx,
                 uint8_t** data, int* w, int* h, int* size)
{
    struct pollfd pf;

    if (!ctx || !data || !w || !h || !size) return -1;

    /* lazy setup on first read (after controls can be set) */
    if (!ctx->setup_done) {
        if (lazy_setup(ctx) < 0) {
            return -1;
        }
    }

    /* wait for a frame to arrive (blocking), give up after 3 timeouts */
    pf.fd      = ctx->fd;
    pf.events  = POLLIN | POLLPRI;
    pf.revents = 0;

    {
        int timeouts = 0;
        while (1) {
            int ret = poll(&pf, 1, 1000);
            if (ret < 0 && errno == EINTR)
                continue;
            if (ret < 0) {
                fprintf(stderr, "[v4l-cam] poll error: %s\n", strerror(errno));
                return -1;
            }
            if (ret > 0) break;
            fprintf(stderr, "[v4l-cam] poll timeout (%d/3)\n", ++timeouts);
            if (timeouts >= 3) {
                fprintf(stderr, "[v4l-cam] too many timeouts, giving up\n");
                return -1;
            }
        }
    }

    /* ── Drain all currently-queued buffers to reach the LATEST frame ──
     * With MAX_BUFS buffers and a slow reader, the driver fills every
     * buffer with consecutive frames. A single DQBUF would return the
     * OLDEST queued frame, leaving the reader N frames behind. We instead
     * dequeue and immediately re-queue every filled buffer, copying each
     * one into output; the last buffer dequeued (the most recently
     * filled) is what we keep. This leaves buffer_count unchanged (so the
     * ISP bug workaround still applies) while guaranteeing the returned
     * frame is the newest available at read time. */
    int pending_index = -1;

    for (;;) {
        struct v4l2_buffer dbuf;

        memset(&dbuf, 0, sizeof(dbuf));
        dbuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        dbuf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(ctx->fd, VIDIOC_DQBUF, &dbuf) < 0)
            break;  /* no more frames available right now */

        /* the previously held buffer is now older than this one:
         * hand it back to the driver so it can be refilled. */
        if (pending_index >= 0) {
            struct v4l2_buffer qbuf;
            memset(&qbuf, 0, sizeof(qbuf));
            qbuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            qbuf.memory = V4L2_MEMORY_MMAP;
            qbuf.index  = pending_index;
            if (xioctl(ctx->fd, VIDIOC_QBUF, &qbuf) < 0) {
                fprintf(stderr, "[v4l-cam] VIDIOC_QBUF failed: %s\n",
                        strerror(errno));
                return -1;
            }
        }

        pending_index = (int)dbuf.index;

        /* copy this (newest-so-far) frame into output. It gets
         * overwritten on the next loop iteration if even newer frames
         * are waiting, otherwise it's the one we return. */
        if (ctx->is_bgr) {
            memcpy(ctx->output, ctx->bufs[dbuf.index].mmap,
                   ctx->width * ctx->height * 3);
        } else {
            const uint8_t *y_plane  = ctx->bufs[dbuf.index].mmap;
            const uint8_t *uv_plane = y_plane + ctx->width * ctx->height;
            nv12_to_bgr_full(y_plane, uv_plane,
                             ctx->width, ctx->height, ctx->output);
        }

        /* if another frame is already queued, keep draining; otherwise
         * the current one is the latest and we stop. */
        struct pollfd npf;
        npf.fd      = ctx->fd;
        npf.events  = POLLIN | POLLPRI;
        npf.revents = 0;
        if (poll(&npf, 1, 0) <= 0)
            break;
    }

    if (pending_index < 0) {
        fprintf(stderr, "[v4l-cam] no frame available after poll\n");
        return -1;
    }

    /* hand the newest buffer back to the driver */
    {
        struct v4l2_buffer qbuf;
        memset(&qbuf, 0, sizeof(qbuf));
        qbuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        qbuf.memory = V4L2_MEMORY_MMAP;
        qbuf.index  = pending_index;
        if (xioctl(ctx->fd, VIDIOC_QBUF, &qbuf) < 0) {
            fprintf(stderr, "[v4l-cam] VIDIOC_QBUF failed: %s\n",
                    strerror(errno));
            return -1;
        }
    }

    *data = ctx->output;
    *w    = ctx->width;
    *h    = ctx->height;
    *size = ctx->width * ctx->height * 3;

    return 0;
}

int v4l_cam_set_hmirror(v4l_cam_ctx_t* ctx, int enable)
{
    if (!ctx) return -1;
    if (ctx->setup_done) {
        fprintf(stderr, "[v4l-cam] set_hmirror: must be called before first read()\n");
        return -1;
    }
    return v4l_cam_set_ctrl(ctx->fd, V4L2_CID_HFLIP, enable ? 1 : 0);
}

int v4l_cam_set_vflip(v4l_cam_ctx_t* ctx, int enable)
{
    if (!ctx) return -1;
    if (ctx->setup_done) {
        fprintf(stderr, "[v4l-cam] set_vflip: must be called before first read()\n");
        return -1;
    }
    return v4l_cam_set_ctrl(ctx->fd, V4L2_CID_VFLIP, enable ? 1 : 0);
}
