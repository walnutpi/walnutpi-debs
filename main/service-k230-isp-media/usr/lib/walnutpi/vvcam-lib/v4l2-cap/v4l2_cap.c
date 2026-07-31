/*
 * v4l2_cap.c — V4L2 采集实现
 */
#include "v4l2_cap.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <unistd.h>

#define PR(fmt, ...) fprintf(stderr, "[v4l2-cap] " fmt "\n", ##__VA_ARGS__)

/* ── 内部辅助 ───────────────────────────────────────────────────── */

static int xioctl(int fd, unsigned long req, void *arg)
{
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

static int set_ctrl(int fd, uint32_t id, int value)
{
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id    = id;
    ctrl.value = value;
    return xioctl(fd, VIDIOC_S_CTRL, &ctrl);
}

static void pixfmt_str(char *out, size_t n, uint32_t f)
{
    snprintf(out, n, "%c%c%c%c",
             (f >>  0) & 0xff, (f >>  8) & 0xff,
             (f >> 16) & 0xff, (f >> 24) & 0xff);
}

/* ── 公共 API ───────────────────────────────────────────────────── */

v4l2_cap_t *v4l2_cap_init(unsigned device, uint32_t want_fmt,
                           unsigned want_w, unsigned want_h,
                           int hflip, int vflip)
{
    v4l2_cap_t *c = NULL;

    /* 1. open */
    char dev[32];
    snprintf(dev, sizeof(dev), "/dev/video%u", device);
    int fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) { PR("open %s: %s", dev, strerror(errno)); return NULL; }
    PR("opened %s", dev);

    /* alloc */
    c = calloc(1, sizeof(v4l2_cap_t));
    if (!c) { PR("oom"); close(fd); return NULL; }
    c->fd = fd;

    /* 2. QUERYCAP */
    {
        struct v4l2_capability cap;
        memset(&cap, 0, sizeof(cap));
        if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
            PR("QUERYCAP: %s", strerror(errno)); goto fail;
        }
        PR("driver: %s, card: %s", cap.driver, cap.card);
    }

    /* 3. G_FMT (probe) */
    {
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
            PR("G_FMT probe: %s", strerror(errno)); goto fail;
        }
        char s[5]; pixfmt_str(s, sizeof(s), fmt.fmt.pix.pixelformat);
        PR("probe: %s %ux%u", s, fmt.fmt.pix.width, fmt.fmt.pix.height);
    }

    /* 4. S_CTRL(flip) — 必须在 S_FMT 前 */
    {
        int h = (hflip >= 0) ? hflip : 0;
        int v = (vflip >= 0) ? vflip : 0;
        if (set_ctrl(fd, V4L2_CID_HFLIP, h) < 0)
            PR("S_CTRL(HFLIP=%d): %s (non-fatal)", h, strerror(errno));
        else
            PR("S_CTRL(HFLIP=%d) ok", h);
        if (set_ctrl(fd, V4L2_CID_VFLIP, v) < 0)
            PR("S_CTRL(VFLIP=%d): %s (non-fatal)", v, strerror(errno));
        else
            PR("S_CTRL(VFLIP=%d) ok", v);
    }

    /* 5. S_FMT */
    {
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.pixelformat = want_fmt;
        fmt.fmt.pix.width       = want_w;
        fmt.fmt.pix.height      = want_h;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;

        if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            PR("S_FMT %ux%u: %s", want_w, want_h, strerror(errno)); goto fail;
        }

        c->pixelformat = fmt.fmt.pix.pixelformat;
        c->width       = fmt.fmt.pix.width;
        c->height      = fmt.fmt.pix.height;

        char s[5]; pixfmt_str(s, sizeof(s), c->pixelformat);
        PR("S_FMT: %s %ux%u (bytesperline=%u, sizeimage=%u)",
           s, c->width, c->height,
           fmt.fmt.pix.bytesperline, fmt.fmt.pix.sizeimage);
    }

    /* 6. REQBUFS */
    {
        struct v4l2_requestbuffers req;
        /* release old */
        memset(&req, 0, sizeof(req));
        req.count  = 0;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(fd, VIDIOC_REQBUFS, &req);

        /* request new */
        memset(&req, 0, sizeof(req));
        req.count  = V4L2_CAP_MAX_BUFS;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
            PR("REQBUFS: %s", strerror(errno)); goto fail;
        }
        c->buf_count = req.count;
        PR("REQBUFS: %u buffers", c->buf_count);
    }

    /* 7. mmap + QBUF */
    for (unsigned i = 0; i < c->buf_count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            PR("QUERYBUF(%u): %s", i, strerror(errno)); goto fail;
        }

        c->bufs[i].length = buf.length;
        c->bufs[i].mmap = mmap(NULL, buf.length,
                               PROT_READ | PROT_WRITE, MAP_SHARED,
                               fd, buf.m.offset);
        if (c->bufs[i].mmap == MAP_FAILED) {
            PR("mmap(%u): %s", i, strerror(errno));
            c->bufs[i].mmap = NULL;
            goto fail;
        }

        struct v4l2_exportbuffer expbuf;
        memset(&expbuf, 0, sizeof(expbuf));
        expbuf.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        expbuf.index = i;
        c->bufs[i].dma_fd = (xioctl(fd, VIDIOC_EXPBUF, &expbuf) == 0)
                          ? expbuf.fd : -1;

        if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            PR("QBUF(%u): %s", i, strerror(errno)); goto fail;
        }
    }
    PR("mmap+QBUF ok (%u buffers)", c->buf_count);

    /* 8. STREAMON */
    {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            PR("STREAMON: %s", strerror(errno)); goto fail;
        }
        PR("STREAMON ok");
    }

    return c;

fail:
    v4l2_cap_close(c);
    return NULL;
}

int v4l2_cap_dequeue(v4l2_cap_t *c, int timeout_ms,
                     uint8_t **data, size_t *bytesused)
{
    struct pollfd pf = { .fd = c->fd, .events = POLLIN | POLLPRI };
    int r = poll(&pf, 1, timeout_ms);
    if (r < 0) { PR("poll: %s", strerror(errno)); return -1; }
    if (r == 0) { PR("poll timeout"); return -1; }

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(c->fd, VIDIOC_DQBUF, &buf) < 0) {
        PR("DQBUF: %s", strerror(errno)); return -1;
    }

    *data       = c->bufs[buf.index].mmap;
    *bytesused  = buf.bytesused;
    return (int)buf.index;
}

int v4l2_cap_queue(v4l2_cap_t *c, int index)
{
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = (unsigned)index;
    return xioctl(c->fd, VIDIOC_QBUF, &buf);
}

void v4l2_cap_close(v4l2_cap_t *c)
{
    if (!c) return;

    int fd = c->fd;
    if (fd >= 0) {
        /* STREAMOFF */
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd, VIDIOC_STREAMOFF, &type);
        PR("STREAMOFF ok");

        /* 复位 sticky flip */
        set_ctrl(fd, V4L2_CID_HFLIP, 0);
        set_ctrl(fd, V4L2_CID_VFLIP, 0);
    }

    /* unmap + close dma_fd */
    for (unsigned i = 0; i < c->buf_count; i++) {
        if (c->bufs[i].mmap && c->bufs[i].mmap != MAP_FAILED)
            munmap(c->bufs[i].mmap, c->bufs[i].length);
        if (c->bufs[i].dma_fd >= 0)
            close(c->bufs[i].dma_fd);
    }

    if (fd >= 0) close(fd);
    free(c);
}
