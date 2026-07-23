# vvcam-lib
用于k230的mipi摄像头&isp的用户空间所需程序。需要配合vvcam的内核驱动使用。

安装
```bash
sudo ./install.sh
```
开机时需要先运行 run_isp_media.sh 脚本

## python库
sensor库继承自opencv的 cv2.VideoCapture ，加入了对摄像头的初始化功能。返回值与 opencv的该api完全通用
```python
import k230_sensor as sensor

cap = sensor.Sensor(1, 1280, 960)
ret, img = cap.read() # 摄像头读取一帧图像    
```
