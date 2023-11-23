#!/bin/bash


clone_or_pull() {
    url="$1"

    project_name=$(basename "$url" .git)
    target_path="../opt/$project_name"

    if [[ ! -d "$target_path" ]]; then
        
        if [[ ! -d "../opt" ]]; then
            mkdir ../opt
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

clone_or_pull "https://github.com/walnutpi/set-lcd.git"
