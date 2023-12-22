#!/bin/bash

# 遍历当前路径下的所有文件夹
for dir in */; do
    # 检查是否存在DEBIAN/control文件
    if [[ -f "${dir}DEBIAN/control" ]]; then
        # 使用sed命令将版本号都改成1.0.0
        sed -i '/Version/c\Version: 1.0.0' "${dir}DEBIAN/control"
    fi
done
