#!/bin/bash
PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
SCRIPT_GEN="${PATH_PWD}/generate.sh"
OUTPUT="${PATH_PWD}/wpi-update-server/web/debian"
SCRIPT_SERVER_BUILD="${PATH_PWD}/wpi-update-server/build.sh"
if [[ ! -d $OUTPUT ]]; then
    mkdir -p ${OUTPUT}
fi



_generate_deb() {
    local dir=$1
    source $SCRIPT_GEN ${PATH_PWD}/$dir ${OUTPUT}/$dir
    if [[ $? == 0 ]]; then
        if [ -f ${SCRIPT_SERVER_BUILD} ]; then
            source ${SCRIPT_SERVER_BUILD}
        fi
    else
        echo "${dir} 无deb包更新"
    fi
}
_generate_deb "main"
_generate_deb "arm64"