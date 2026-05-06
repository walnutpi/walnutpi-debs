#!/bin/bash
PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT="${PATH_PWD}/output"
if [[ ! -d $OUTPUT ]]; then
    mkdir -p ${OUTPUT}
fi

bash ${PATH_PWD}/build_local.sh
bash ${PATH_PWD}/build_source.sh
