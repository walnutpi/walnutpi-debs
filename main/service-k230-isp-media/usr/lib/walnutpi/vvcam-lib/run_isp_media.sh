#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

modprobe vvcam_isp
modprobe vvcam_mipi
modprobe vvcam_vb
modprobe vvcam_isp_subdev
modprobe vvcam_video mcm_mask=1

chmod 777 /proc/vsi/isp_subdev0
if [ -w /proc/sys/kernel/sched_rt_runtime_us ]; then
	echo -1 > /proc/sys/kernel/sched_rt_runtime_us
fi

ISP_MEDIA_SENSOR_DRIVER=/usr/lib/libvvcam.so "${SCRIPT_DIR}/isp_media_server" >/tmp/isp.err.log 2>&1 
sleep 1
echo 0 mode=1 > /proc/vsi/isp_subdev0
echo 1 mode=1 > /proc/vsi/isp_subdev0
echo 2 mode=1 > /proc/vsi/isp_subdev0

# 等待修复初始化bug后再启用这个自动配置功能
# python3 -c "import k230_sensor; cap = k230_sensor.Sensor(1920, 1080).sensor_init()" &
# python3 -c "import k230_sensor; cap = k230_sensor.Sensor(1920, 1080, 1).sensor_init()" &