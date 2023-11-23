#!/bin/bash

# 获取脚本的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${SCRIPT_DIR}/output"
SAVE_DIR="${SCRIPT_DIR}/save"

if [[ -d $OUTPUT_DIR ]]; then
    rm -r  $OUTPUT_DIR
fi

if [[ ! -d ${OUTPUT_DIR} ]]; then
    mkdir ${OUTPUT_DIR}
fi
if [[ -d $SAVE_DIR ]]; then
    cp ${SAVE_DIR}/*  ${OUTPUT_DIR}
fi


# 遍历当前目录下的所有文件夹
for dir in ${SCRIPT_DIR}/*/ ; do
    if [[ $dir == ${OUTPUT_DIR}/ ]]; then
        continue
    fi
    if [[ $dir == ${SAVE_DIR}/ ]]; then
        continue
    fi
    cd $dir
    
    # 检查DEBIAN/gen.sh文件是否存在，如果存在就运行它
    if [[ -f DEBIAN/gen.sh ]]; then
        cd DEBIAN
        bash gen.sh
        cd $dir
    fi
    package_name=$(grep -oP '(?<=Package: ).*' DEBIAN/control)
    version=$(grep -oP '(?<=Version: ).*' DEBIAN/control)
    architecture=$(grep -oP '(?<=Architecture: ).*' DEBIAN/control)
    
    # 计算大小写入DEBIAN/control文件的Installed-Size:行
    size=$(du -sk --exclude=DEBIAN . | cut -f1)
    sed -i "/Installed-Size:/c\Installed-Size: $size" DEBIAN/control
    
    cd ..
    dpkg -b "$dir" "${OUTPUT_DIR}/${package_name}_${version}_${architecture}.deb"
done
