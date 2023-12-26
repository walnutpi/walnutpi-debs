#!/bin/bash

SOURCE_DIR=$1
OUTPUT_DIR=$2
SAVE_DIR="${SOURCE_DIR}/save"
COUNT_SUCCESS=0
if [[ ! -d ${OUTPUT_DIR} ]]; then
    echo "创建${OUTPUT_DIR}"
    mkdir ${OUTPUT_DIR}
fi

# [[ ! -d ${OUTPUT_DIR} ]] && mkdir ${OUTPUT_DIR}

[[ -d $SAVE_DIR ]] && cp ${SAVE_DIR}/*  ${OUTPUT_DIR}

# 遍历当前目录下的所有文件夹
for dir in ${SOURCE_DIR}/*/ ; do
    
    if [[ $dir == ${SAVE_DIR}/ ]]; then
        continue
    fi
    cd $dir
    
    package_name=$(grep -oP '(?<=Package: ).*' DEBIAN/control)
    version=$(grep -oP '(?<=Version: ).*' DEBIAN/control)
    architecture=$(grep -oP '(?<=Architecture: ).*' DEBIAN/control)
    
    deb_file="${OUTPUT_DIR}/${package_name}_${version}_${architecture}.deb"
    if [[ -f $deb_file ]]; then
        # echo -e "\t存在与control版本号相同包,跳过"
        continue
    fi
    echo -e "构建: \t${dir}"
    
    let COUNT_SUCCESS++
    # 检查DEBIAN/gen.sh文件是否存在，如果存在就运行它
    if [[ -f DEBIAN/gen.sh ]]; then
        cd DEBIAN
        bash gen.sh
        cd $dir
    fi
    
    # 计算大小
    size=$(du -sk --exclude=DEBIAN . | cut -f1)
    
    # 获取DEBIAN/control文件中的Installed-Size:行的值
    old_size=$(grep -oP '(?<=Installed-Size: ).*' DEBIAN/control)
    
    # 如果新的大小和旧的大小不同，就写入新的大小
    if [[ $size != $old_size ]]; then
        sed -i "/Installed-Size:/c\Installed-Size: $size" DEBIAN/control
    fi
    
    
    cd ..
    dpkg -b "$dir" "$deb_file"
done
if (( $COUNT_SUCCESS > 0 )); then
    return 0
fi
return 1