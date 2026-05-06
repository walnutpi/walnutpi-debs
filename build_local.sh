#!/bin/bash
PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT="${PATH_PWD}/output"
if [[ ! -d $OUTPUT ]]; then
    mkdir -p ${OUTPUT}
fi

# 生成deb包的函数
_generate_deb_packages() {
    local SOURCE_DIR=$1
    local OUTPUT_DIR=$2
    local COUNT_SUCCESS=0
    
    if [[ ! -d ${OUTPUT_DIR} ]]; then
        echo "创建${OUTPUT_DIR}"
        mkdir -p ${OUTPUT_DIR}
    fi

    # 遍历当前目录下的所有文件夹
    for dir in $(ls -d ${SOURCE_DIR}/*/ ) ; do
        cd $dir
        
        package_name=$(grep -oP '(?<=Package: ).*' DEBIAN/control)
        version=$(grep -oP '(?<=Version: ).*' DEBIAN/control)
        architecture=$(grep -oP '(?<=Architecture: ).*' DEBIAN/control)
        
        deb_file="${OUTPUT_DIR}/${package_name}_${version}_${architecture}.deb"
        if [[ -f $deb_file ]]; then
            # echo -e "\t存在包 ${deb_file} ,跳过"
            cd ..
            continue
        fi
        echo -e "构建: \t${deb_file}"
        
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
        # 使用 gzip 压缩格式构建 deb 包
        dpkg-deb -Zgzip -b "$dir" "$deb_file"
    done

    if (( $COUNT_SUCCESS > 0 )); then
        return 0
    fi
    return 1
}

# 主构建函数
_build_directory() {
    local directory=$1
    _generate_deb_packages ${PATH_PWD}/$directory ${OUTPUT}/
    if [[ $? == 0 ]]; then
        echo "${directory} 构建成功"
    else
        echo "${directory} 无deb包更新"
    fi
}

# 执行构建
_build_directory "main"
_build_directory "arm64"