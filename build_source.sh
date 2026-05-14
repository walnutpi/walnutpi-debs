#!/bin/bash
# 脚本功能：扫描source文件夹下的每个子项目，如果存在pack_as_deb.sh则执行它


PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$PROJECT_ROOT/source"
OUTPUT_DIR="$PROJECT_ROOT/output"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

PATH_WPI_SERVER="${PROJECT_ROOT}/wpi-update-server/web/debian/pool/main"
if [ -d ${PATH_WPI_SERVER} ]; then
    OUTPUT_DIR=${PATH_WPI_SERVER}
fi

# 检查source目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: source目录不存在: $SOURCE_DIR"
    exit 1
fi

# 计数器
total_count=0
executed_count=0
skipped_count=0

# 遍历source目录下的每个子项目
for project_dir in "$SOURCE_DIR"/*/; do
    # 检查是否是目录
    if [ ! -d "$project_dir" ]; then
        continue
    fi
    
    total_count=$((total_count + 1))
    
    # 获取项目名称
    project_name=$(basename "$project_dir")
    pack_script="$project_dir/pack_as_deb.sh"
    
    # 检查pack_as_deb.sh是否存在
    if [ -f "$pack_script" ]; then
        # 确保脚本有执行权限
        chmod +x "$pack_script"
        
        # 执行pack_as_deb.sh并传入output目录路径
        cd "$project_dir" || continue
        if bash "$pack_script" "$OUTPUT_DIR"; then
            # echo "  ✓ 项目 $project_name 打包成功"
            executed_count=$((executed_count + 1))
        else
            echo "  ✗ 项目 $project_name 打包失败"
        fi
        cd "$PROJECT_ROOT" || exit 1
    else
        skipped_count=$((skipped_count + 1))
    fi
done
echo "已执行: $executed_count"
if [ $skipped_count -gt 0 ]; then
    echo "无构建脚本: $skipped_count"
fi