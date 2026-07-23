#ifndef __GET_FRAME_H__
#define __GET_FRAME_H__

#include <linux/types.h>
#include <linux/videodev2.h>

/*  Flags for 'capability' and 'capturemode' fields */
#define V4L2_MODE_HIGHQUALITY		0x0001
#define V4L2_MODE_VIDEO			0x0002
#define V4L2_MODE_IMAGE			0x0003
#define V4L2_MODE_PREVIEW		0x0004


int get_frame();

#endif