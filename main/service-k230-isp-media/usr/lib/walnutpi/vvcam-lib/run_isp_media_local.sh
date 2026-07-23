#!/bin/bash

# 运行lsmod查看当前加载的内核模块，判断是否有vvcam_isp模块

modprobe vvcam_isp
modprobe vvcam_mipi
modprobe vvcam_vb
modprobe vvcam_isp_subdev
modprobe vvcam_video

chmod 777 /proc/vsi/isp_subdev0
ISP_MEDIA_SENSOR_DRIVER=./libvvcam.so ./isp_media_server_debian