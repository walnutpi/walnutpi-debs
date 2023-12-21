#!/bin/bash

# 获取脚本的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${SCRIPT_DIR}/output"
SAVE_DIR="${SCRIPT_DIR}/save"

if [[ -d $OUTPUT_DIR ]]; then
    rm -r  $OUTPUT_DIR
fi

[[ ! -d ${OUTPUT_DIR} ]] && mkdir ${OUTPUT_DIR}

[[ -d $SAVE_DIR ]] && cp ${SAVE_DIR}/*  ${OUTPUT_DIR}

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
    
    # 计算大小
    size=$(du -sk --exclude=DEBIAN . | cut -f1)
    
    # 获取DEBIAN/control文件中的Installed-Size:行的值
    old_size=$(grep -oP '(?<=Installed-Size: ).*' DEBIAN/control)
    
    # 如果新的大小和旧的大小不同，就写入新的大小
    if [[ $size != $old_size ]]; then
        sed -i "/Installed-Size:/c\Installed-Size: $size" DEBIAN/control
    fi
    
    deb_file="${OUTPUT_DIR}/${package_name}_${version}_${architecture}.deb"
    
    # # 检查deb包是否存在，如果存在并且生成时间晚于文件夹的修改时间，就跳过这个包
    # if [[ -f $deb_file ]]; then
    #     deb_time=$(stat -c %Y "$deb_file")
    #     dir_time=$(find "$dir" -type f -exec stat -c %Y {} \; | sort -nr | head -1)
    #     if (( deb_time >= dir_time )); then
    #         continue
    #     fi
    # fi
    
    cd ..
    dpkg -b "$dir" "$deb_file"
done
