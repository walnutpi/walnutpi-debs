#!/bin/bash
URL_TOOL="https://github.com/walnutpi/wpi-update.git"


PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT="${PATH_PWD}/server"
SCRIPT_GEN="${PATH_PWD}/generate.sh"
DIR_TOOL="${PATH_PWD}/$(basename "$URL_TOOL" .git)"

if [[ ! -d $OUTPUT ]]; then
    mkdir ${OUTPUT}
fi


clone_url() {
    local git_url="$1"
    dir_name=$(basename "$git_url" .git)
    
    if [ -d "$dir_name" ]; then
        cd "$dir_name"
        git config --global --add safe.directory $(pwd)
        echo "pull : $git_url"
        git pull
    else
        echo "clone : $git_url"
        git clone $git_url
    fi
}




cp release-cn.log ${OUTPUT}/
cp release-en.log ${OUTPUT}/

clone_url $URL_TOOL
# cp wpi-update ${OUTPUT}/
cp $DIR_TOOL/wpi-update  ${OUTPUT}/

COUNT_DEB_SUCCESS=0

# 生成deb包
source $SCRIPT_GEN ${PATH_PWD}/main ${OUTPUT}/main/
if [ $? == 0 ]; then
    let COUNT_DEB_SUCCESS++
    # 生成包索引
    bin_all=${OUTPUT}/dists/bookworm/main/binary-all
    mkdir  -p ${bin_all}
    cd ${OUTPUT}
    dpkg-scanpackages main /dev/null > ${bin_all}/Packages
fi

# 生成deb包
source $SCRIPT_GEN ${PATH_PWD}/arm64 ${OUTPUT}/arm64/
if [[ $? == 0 ]]; then
    let COUNT_DEB_SUCCESS++
    # 生成包索引
    bin_arm64=${OUTPUT}/dists/bookworm/main/binary-arm64
    mkdir  -p ${bin_arm64}
    cd ${OUTPUT}
    dpkg-scanpackages arm64 /dev/null > ${bin_arm64}/Packages
    
fi

# dpkg-scanpackages main  /dev/null | gzip> ${bin_all}/Packages.gz
# dpkg-scanpackages arm64  /dev/null | gzip> ${bin_arm64}/Packages.gz
# gzip -c -d ${bin_all}/Packages.gz  > ${bin_all}/Packages
# gzip -c -d ${bin_arm64}/Packages.gz  > ${bin_arm64}/Packages


# 有包更新就重新生成release
if (( $COUNT_DEB_SUCCESS > 0 )); then
    cd $PATH_PWD
    apt-ftparchive -c=bookworm.conf release ${OUTPUT}/dists/bookworm > ${OUTPUT}/dists/bookworm/Release
else
    echo "没有任何包有更新"
fi



