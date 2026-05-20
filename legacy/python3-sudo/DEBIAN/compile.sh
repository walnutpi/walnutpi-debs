#!/bin/bash

# 获取脚本的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}

gcc Python.c -o Python
chmod 755 Python
mv Python ../usr/bin/


gcc Python3.c -o Python3
chmod 755 Python3
mv Python3 ../usr/bin/


