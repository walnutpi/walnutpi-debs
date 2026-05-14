#!/bin/bash
PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT="${PATH_PWD}/output"
PATH_WPI_SERVER="${PATH_PWD}/wpi-update-server/"
if [ -d ${PATH_WPI_SERVER} ]; then
    OUTPUT=${PATH_WPI_SERVER}
fi
if [[ ! -d $OUTPUT ]]; then
    mkdir -p ${OUTPUT}
fi

bash ${PATH_PWD}/build_local.sh
bash ${PATH_PWD}/build_source.sh

if [ -d $PATH_WPI_SERVER ]; then
    cd ${PATH_WPI_SERVER}
    bash ./build.sh
fi
