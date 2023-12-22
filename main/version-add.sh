#!/bin/bash

# 遍历当前路径下的所有文件夹
for dir in */; do
    # 检查是否存在DEBIAN/control文件
    if [[ -f "${dir}DEBIAN/control" ]]; then
        # 使用awk命令修改版本号的第三位
        awk -F'[:.]' '
            BEGIN {OFS="";}
            /Version/ {
                $2=": ";
                $6+=1;
                if ($6 == 10) {
                    $6 = 0;
                    $4+=1;
                    if ($4 == 10) {
                        $4 = 0;
                        $2+=1;
                    }
                }
            }
            {print $0}
        ' "${dir}DEBIAN/control" > temp && mv temp "${dir}DEBIAN/control"
    fi
done
