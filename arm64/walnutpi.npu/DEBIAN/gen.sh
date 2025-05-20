#!/bin/bash

PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

clone() {
    url="$1"

    project_name=$(basename "$url" .git)
    target_path="${PATH_PWD}/../usr/lib/walnutpi/$project_name"
    if [[ -d "$target_path" ]]; then
        rm  -rf "$target_path"
    fi
    
    if [[ ! -d "${PATH_PWD}/../usr/lib/walnutpi" ]]; then
        mkdir -p ${PATH_PWD}/../usr/lib/walnutpi
    fi

    max_attempts=5
    attempt_num=1
    until [[ $attempt_num -gt $max_attempts ]]
    do
        git clone --depth=1 "$url" "$target_path" && break
        attempt_num=$((attempt_num+1))
        echo "git clone failed, attempt $attempt_num..."
        sleep 3
    done
    
    if [[ $attempt_num -gt $max_attempts ]]; then
        echo "git clone failed after $max_attempts attempts, exiting..."
        exit 1
    fi

}

clone "https://github.com/walnutpi/walnutpi.npu.git"
