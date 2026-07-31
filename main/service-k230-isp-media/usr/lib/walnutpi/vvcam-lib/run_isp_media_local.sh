#!/bin/bash

# 运行lsmod查看当前加载的内核模块，判断是否有vvcam_isp模块

modprobe vvcam_isp
modprobe vvcam_mipi
modprobe vvcam_vb
modprobe vvcam_isp_subdev
modprobe vvcam_video mcm_mask=1

chmod 777 /proc/vsi/isp_subdev0
if [ -w /proc/sys/kernel/sched_rt_runtime_us ]; then
	echo -1 > /proc/sys/kernel/sched_rt_runtime_us
fi


{
    sleep 1
    echo 0 mode=1 > /proc/vsi/isp_subdev0
    echo 1 mode=1 > /proc/vsi/isp_subdev0
    echo 2 mode=1 > /proc/vsi/isp_subdev0
    # python3 -c "import k230_sensor; cap = k230_sensor.Sensor(1920, 1080).sensor_init()"
    # python3 -c "import k230_sensor; cap = k230_sensor.Sensor(1920, 1080, 1).sensor_init()"
} & 
ISP_MEDIA_SENSOR_DRIVER=/usr/lib/libvvcam.so "./isp_media_server"
