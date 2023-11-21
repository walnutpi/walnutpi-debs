#!/bin/bash
PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT="${PATH_PWD}/output"

if [[ -d $OUTPUT ]]; then
    rm -r  $OUTPUT
fi
mkdir ${OUTPUT}


#################################
#           生成deb包           #
#################################
mkdir ${OUTPUT}/main
./main/gen.sh
cp main/output/*.deb ${OUTPUT}/main/


mkdir ${OUTPUT}/arm64
./arm64/gen.sh
cp arm64/output/*.deb ${OUTPUT}/arm64/


#################################
#           生成dists           #
#################################
cd ${OUTPUT}
bin_all=${OUTPUT}/dists/bookworm/main/binary-all
bin_arm64=${OUTPUT}/dists/bookworm/main/binary-arm64
mkdir  -p ${bin_all}
mkdir  -p ${bin_arm64}
dpkg-scanpackages main  /dev/null | gzip> ${bin_all}/Packages.gz 
dpkg-scanpackages arm64  /dev/null | gzip> ${bin_arm64}/Packages.gz 

gzip -c -d ${bin_all}/Packages.gz  > ${bin_all}/Packages
gzip -c -d ${bin_arm64}/Packages.gz  > ${bin_arm64}/Packages

cd $PATH_PWD
apt-ftparchive -c=bookworm.conf release ${OUTPUT}/dists/bookworm > ${OUTPUT}/dists/bookworm/Release





