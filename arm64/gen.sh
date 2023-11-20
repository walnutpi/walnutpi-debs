#!/bin/bash

# 获取脚本的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${SCRIPT_DIR}/output"

if [[ -d $OUTPUT_DIR ]]; then
    rm -r  $OUTPUT_DIR
fi

if [[ ! -d ${OUTPUT_DIR} ]]; then
    mkdir ${OUTPUT_DIR}
fi

cp $SCRIPT_DIR/save/*.deb $OUTPUT_DIR
