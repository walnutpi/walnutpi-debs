#ifndef V4L_CAM_H
#define V4L_CAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct v4l_cam_ctx v4l_cam_ctx_t;

/**
 * Open a V4L2 capture device, configure NV12 format at given resolution,
 * allocate MMAP buffers, and start streaming.
 *
 * @param device  Video device number (e.g. 1 for /dev/video1)
 * @param width   Desired frame width (driver may adjust)
 * @param height  Desired frame height (driver may adjust)
 * @return        Opaque context pointer, or NULL on failure
 */
v4l_cam_ctx_t* v4l_cam_open(int device, int width, int height);

/**
 * Stop streaming, unmap buffers, free resources, and close the device.
 */
void v4l_cam_close(v4l_cam_ctx_t* ctx);

/**
 * Read one frame from the camera. Blocks until a frame is available.
 * The returned BGR data pointer is valid until the next call to v4l_cam_read
 * or v4l_cam_close. The caller should copy the data if persistence is needed.
 *
 * @param ctx   Context from v4l_cam_open
 * @param data  Output: pointer to BGR888 pixel data (internal buffer)
 * @param w     Output: actual frame width
 * @param h     Output: actual frame height
 * @param size  Output: total bytes (w * h * 3)
 * @return      0 on success, -1 on error
 */
int v4l_cam_read(v4l_cam_ctx_t* ctx,
                 uint8_t** data, int* w, int* h, int* size);

/**
 * Enable/disable horizontal mirror (hardware-level via V4L2 control).
 * Must be called after v4l_cam_open and before the first v4l_cam_read.
 *
 * @param enable  0 = disable, non-zero = enable
 * @return        0 on success, -1 on error
 */
int v4l_cam_set_hmirror(v4l_cam_ctx_t* ctx, int enable);

/**
 * Enable/disable vertical flip (hardware-level via V4L2 control).
 * Must be called after v4l_cam_open and before the first v4l_cam_read.
 *
 * @param enable  0 = disable, non-zero = enable
 * @return        0 on success, -1 on error
 */
int v4l_cam_set_vflip(v4l_cam_ctx_t* ctx, int enable);

#ifdef __cplusplus
}
#endif

#endif /* V4L_CAM_H */
