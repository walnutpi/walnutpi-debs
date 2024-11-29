#!/bin/bash

PATH_PWD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

clone_or_pull() {
    url="$1"

    project_name=$(basename "$url" .git)
    target_path="${PATH_PWD}/../usr/lib/walnutpi/$project_name"

    if [[ ! -d "$target_path" ]]; then
        
        if [[ ! -d "${PATH_PWD}/../usr/lib/walnutpi" ]]; then
            mkdir -p ${PATH_PWD}/../usr/lib/walnutpi
        fi

        max_attempts=5
        attempt_num=1
        until [[ $attempt_num -gt $max_attempts ]]
        do
            git clone "$url" "$target_path" && break
            attempt_num=$((attempt_num+1))
            echo "git clone failed, attempt $attempt_num..."
            sleep 3
        done
        
        if [[ $attempt_num -gt $max_attempts ]]; then
            echo "git clone failed after $max_attempts attempts, exiting..."
            exit 1
        fi
    else
        cd "$target_path"
        git pull
    fi
}

clone_or_pull "https://github.com/walnutpi/Adafruit_Python_PlatformDetect.git"
clone_or_pull "https://github.com/walnutpi/Adafruit_Blinka.git"

FILE_SETUP="${PATH_PWD}/../usr/lib/walnutpi/Adafruit_Python_PlatformDetect/setup.py"

# 在ubuntu22上，需要提供setup.py才能用pip本地安装
if [ ! -f $FILE_SETUP ]; then
    # touch $FILE_SETUP

    cat << EOF > $FILE_SETUP
#!/usr/bin/env python
import setuptools
if __name__ == "__main__":
    setuptools.setup()
EOF

fi