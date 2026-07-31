
用于测试捕获摄像头

在开发板上捕获图像原始数据
```shell

# 编译
make

# 以 NV12 格式捕获 10 帧，将原始数据保存为.nv12文件
./v4l2-cap -f NV12 -n 10

# 以 BGR24 格式捕获 10 帧，将原始数据保存为.bgr文件
./v4l2-cap -f BGR24 -n 10

# 将捕获的原始数据转换为JPG
python3 bin2jpg.py 
```