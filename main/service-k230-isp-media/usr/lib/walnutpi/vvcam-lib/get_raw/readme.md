# 获取nv12测试帧

```shell
# 编译
make

# 读取一张nv12图片保存到本地名为xxx.bin
./get_frame 

# 将路径下所以.bin文件转为同名jpg保存起来
python convert_bin_to_jpg.py
```