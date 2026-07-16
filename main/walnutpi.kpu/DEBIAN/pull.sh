#!/bin/bash

PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

clone() {
    url="$1"

    project_name=$(basename "$url" .git)
    target_path="${PATH_PWD}/../usr/lib/walnutpi"

    if [[ -d "$target_path" ]]; then
        rm -rf "$target_path"
    fi
    mkdir -p ${target_path}
    
    max_attempts=5
    attempt_num=1
    until [[ $attempt_num -gt $max_attempts ]]
    do
        git clone --depth=1 "$url" "$target_path/$project_name" && break
        attempt_num=$((attempt_num+1))
        echo "git clone failed, attempt $attempt_num..."
        sleep 3
    done
    
    if [[ $attempt_num -gt $max_attempts ]]; then
        echo "git clone failed after $max_attempts attempts, exiting..."
        exit 1
    fi

    rm -r "$target_path/$project_name/.git"

}

clone "https://github.com/walnutpi/walnutpi.kpu.git"
