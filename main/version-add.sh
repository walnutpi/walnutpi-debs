#!/bin/bash

# 遍历当前路径下的所有文件夹
for dir in */; do
    # 检查是否存在DEBIAN/control文件
    if [[ -f "${dir}DEBIAN/control" ]]; then
        # 使用perl命令修改版本号的第三位
        perl -i -pe 's/(Version: \d+\.\d+\.)(\d+)/$1.($2+1)/e' "${dir}DEBIAN/control"
    fi
done
