#!/bin/bash

# 创建source目录（如果不存在）
if [ ! -d "source" ]; then
    echo "Creating source directory..."
    mkdir -p source
fi

# 读取package.list文件并处理每一行
while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行和以#开头的注释行
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # 去除行首尾空白字符
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # 再次检查是否为空或注释
    if [[ -z "$line" ]] || [[ "$line" =~ ^# ]]; then
        continue
    fi
    
    # 从URL中提取仓库名称
    repo_name=$(basename "$line" .git)
    
    # 检查source目录下是否已存在该仓库
    if [ -d "source/$repo_name" ]; then
        echo "Repository $repo_name already exists. Pulling latest changes..."
        cd "source/$repo_name"
        git pull
        cd ../..
    else
        echo "Cloning repository $repo_name..."
        git clone "$line" "source/$repo_name"
    fi
done < package.list
