/**
 * 在k230上，以NV12格式读取摄像头接口（isp开启）的图像，然后保存为.bin文件
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <poll.h>

#include <asm/types.h>

#include "get_frame.h"

#define CAP_COUNT 1  // 捕获图像数量
// 摄像头节点
#define CAMERA_DEV_NAME "/dev/video1"
// 节点内可以有多个摄像头，指定哪个摄像头
#define CAMERA_DEV_Index 0
#define CAMERA_Width 1280
#define CAMERA_Height 960
#define CAMEAR_PixelFormat V4L2_PIX_FMT_NV12
#define fps 30
#define wdr_mode 0
enum v4l2_buf_type Camera_buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 单平面缓冲区类型

// 保存图像到文件
void save_image(void *buf, int index, int length)
{
    char path_name[100];
    FILE *file_fd = NULL;
    sprintf(path_name, "frame_%d_%dx%d.bin", index, CAMERA_Width, CAMERA_Height);
    file_fd = fopen(path_name, "wb");
    if (file_fd == NULL) {
        printf("无法创建文件: %s\n", path_name);
        return;
    }
    fwrite(buf, length, 1, file_fd);
    fclose(file_fd);
    printf("已保存帧数据到: %s (大小: %d 字节)\n", path_name, length);
}

int get_frame()
{
    // 对于单平面格式，平面数固定为1
    unsigned int nplanes = 1;

    // 1.打开设备节点
    int fd = open(CAMERA_DEV_NAME, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0)
    {
        printf("open failed\n");
        return -1;
    }

    // 2. 选择使用节点内的哪个摄像头
    struct v4l2_input inp;
    inp.index = CAMERA_DEV_Index;
    if (-1 == ioctl(fd, VIDIOC_S_INPUT, &inp))
    {
        printf("VIDIOC_S_INPUT %d error!\n", CAMERA_DEV_Index);
        return -1;
    }

    // 3.设置摄像头参数
    struct v4l2_streamparm parms;
    memset(&parms, 0, sizeof(parms));
    parms.type = Camera_buf_type;
    parms.parm.capture.timeperframe.numerator = 1;
    parms.parm.capture.timeperframe.denominator = fps;
    parms.parm.capture.capturemode = V4L2_MODE_VIDEO;
    parms.parm.capture.reserved[0] = 0;
    parms.parm.capture.reserved[1] = wdr_mode;
    if (-1 == ioctl(fd, VIDIOC_S_PARM, &parms))
    {
        printf("VIDIOC_S_PARM error (this might be normal on some devices)\n");
        // 注意：有些设备可能不支持此操作，可以忽略错误
    }

    // 4.设置像素格式
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = Camera_buf_type;
    fmt.fmt.pix.width = CAMERA_Width;
    fmt.fmt.pix.height = CAMERA_Height;
    fmt.fmt.pix.pixelformat = CAMEAR_PixelFormat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    
    if (-1 == ioctl(fd, VIDIOC_S_FMT, &fmt))
    {
        printf("VIDIOC_S_FMT error!\n");
        return -1;
    }

    if (-1 == ioctl(fd, VIDIOC_G_FMT, &fmt))
    {
        printf("VIDIOC_G_FMT error!\n");
        return -1;
    }
    else
    {
        // 对于单平面格式，直接使用获取到的尺寸信息
        printf("resolution got from sensor = %d*%d pixel_format = %c%c%c%c\n",
               fmt.fmt.pix.width, fmt.fmt.pix.height,
               (fmt.fmt.pix.pixelformat >> 0) & 0xFF,
               (fmt.fmt.pix.pixelformat >> 8) & 0xFF,
               (fmt.fmt.pix.pixelformat >> 16) & 0xFF,
               (fmt.fmt.pix.pixelformat >> 24) & 0xFF);
    }

    // 5. 向v4l2申请多个缓冲区用于存放图像数据
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 5;
    req.type = Camera_buf_type;  // 使用单平面类型
    req.memory = V4L2_MEMORY_MMAP;
    if (-1 == ioctl(fd, VIDIOC_REQBUFS, &req))
    {
        printf("VIDIOC_REQBUFS error\n");
        return -1;
    }

    // 6. 将申请到的缓冲区放入队列中（单平面格式不需要使用planes数组）
    for (int i = 0; i < req.count; i++)
    {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = Camera_buf_type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (-1 == ioctl(fd, VIDIOC_QBUF, &buf))
        {
            printf("VIDIOC_QBUF failed\n");
            return -1;
        }
    }

    // 7. 映射申请到的缓冲区
    struct buffer
    {
        void *start;
        size_t length;
    };
    struct buffer *buffers;
    buffers = calloc(req.count, sizeof(*buffers));
    
    for (int i = 0; i < req.count; i++)
    {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = Camera_buf_type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (-1 == ioctl(fd, VIDIOC_QUERYBUF, &buf))
        {
            printf("VIDIOC_QUERYBUF error\n");
            return -1;
        }
        
        // 单平面格式直接使用length和offset
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, buf.m.offset);

        if (buffers[i].start == MAP_FAILED)
        {
            printf("mmap failed\n");
            return -1;
        }
    }
    
    // printf("申请到%d个buffer \n", req.count);
    // printf("单平面格式，平面数 = %d \n", nplanes);

    // 8. 启动摄像头
    if (-1 == ioctl(fd, VIDIOC_STREAMON, &Camera_buf_type))
    {
        printf("VIDIOC_STREAMON failed\n");
        return -1;
    }
    else
        printf("VIDIOC_STREAMON ok\n");

    // 9. 读取图像数据（始终返回最新的一帧）
    // 驱动以 FIFO 顺序填充 buffer：读取频率远低于采集频率时，
    // 仅 DQBUF 一次会拿到队列中最旧的帧。因此这里把当前所有已就绪的
    // 缓冲区全部取出，只保留“最后取到的那一个”（即最新帧），
    // 其余较旧的缓冲区立即重新入队交还给驱动继续填充。
    struct v4l2_buffer latest_buf;
    bool has_frame = false;
    struct v4l2_buffer prev_buf;
    bool has_prev = false;

    for (int count = 0; count < CAP_COUNT; count++)
    {
        for (;;)
        {
            fd_set fds;
            struct timeval tv;
            int r;

            FD_ZERO(&fds);
            FD_SET(fd, &fds);

            tv.tv_sec = 2;
            tv.tv_usec = 0;

            r = select(fd + 1, &fds, NULL, NULL, &tv);
            if (-1 == r)
            {
                if (errno == EINTR)
                    continue;
                printf("select err\n");
            }

            if (r == 0)
            {
                fprintf(stderr, "select timeout\n");
                continue;
            }

            // 持续 DQBUF，直到没有更多已填充的缓冲区（EAGAIN）
            for (;;)
            {
                struct v4l2_buffer buf;
                memset(&buf, 0, sizeof(buf));
                buf.type = Camera_buf_type;
                buf.memory = V4L2_MEMORY_MMAP;

                int dq = ioctl(fd, VIDIOC_DQBUF, &buf);
                if (dq == -1)
                {
                    if (errno == EAGAIN)
                        break;          // 已无更多就绪缓冲区，prev_buf 即为最新帧
                    if (errno == EINTR)
                        continue;
                    printf("VIDIOC_DQBUF failed\n");
                    break;
                }

                // 若之前已取到一帧，把那一帧（更旧的）重新入队
                if (has_prev)
                {
                    if (-1 == ioctl(fd, VIDIOC_QBUF, &prev_buf))
                    {
                        printf("VIDIOC_QBUF (older) failed\n");
                    }
                }
                prev_buf = buf;
                has_prev = true;
            }

            if (has_prev)
            {
                latest_buf = prev_buf;
                has_frame = true;
                break;
            }
        }

        if (!has_frame)
            continue;

        char *p = buffers[latest_buf.index].start;
        // printf("通道前3个数据 %d %d %d \n", p[0], p[1], p[2]);

        // 保存单平面数据
        save_image(p, count, buffers[latest_buf.index].length);

        // 将最新帧所在的缓冲区也重新加入队列
        if (-1 == ioctl(fd, VIDIOC_QBUF, &latest_buf))
        {
            printf("VIDIOC_QBUF (latest) failed\n");
            return -1;
        }
    }

    // 10. 停止摄像头捕获
    if (-1 == ioctl(fd, VIDIOC_STREAMOFF, &Camera_buf_type))
    {
        printf("VIDIOC_STREAMOFF failed\n");
        return -1;
    }
    else
        printf("VIDIOC_STREAMOFF ok\n");

    // 清理资源
    for (int i = 0; i < req.count; i++) {
        munmap(buffers[i].start, buffers[i].length);
    }
    free(buffers);
    close(fd);
    
    return 0;
}