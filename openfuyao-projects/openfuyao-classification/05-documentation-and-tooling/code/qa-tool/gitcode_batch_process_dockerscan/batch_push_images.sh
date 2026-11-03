#!/bin/bash

# 配置参数 - 根据实际情况修改
SOURCE_REGISTRY=""    # 源镜像仓库（如果镜像名已包含仓库，可留空）
TARGET_REGISTRY="" # 目标镜像仓库
TARGET_USER=""             # 目标仓库用户名
TARGET_PASSWORD=""                 # 目标仓库密码（可留空，会提示输入或从环境变量获取）
IMAGES_LIST="images.txt"           # 需要上传的镜像列表文件
OUTPUT_DIR="./docker_images"       # 保存tar文件的目录
LOG_FILE="process_log.txt"         # 日志文件

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 记录日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 检查必要工具
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        echo "错误: 未安装docker，请先安装docker"
        exit 1
    fi

    if [ ! -f "$IMAGES_LIST" ]; then
        echo "错误: 镜像列表文件 $IMAGES_LIST 不存在"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "错误: Docker服务未启动，请先启动Docker"
        exit 1
    fi
}

# 登录目标仓库（支持自动输入密码）
login_to_target() {
    log "登录到目标镜像仓库 $TARGET_REGISTRY..."
    
    # 从环境变量获取密码（优先）
    if [ -z "$TARGET_PASSWORD" ] && [ -n "$DOCKER_REGISTRY_PASSWORD" ]; then
        TARGET_PASSWORD="$DOCKER_REGISTRY_PASSWORD"
    fi
    
    # 如果仍无密码，则提示用户输入
    if [ -z "$TARGET_PASSWORD" ]; then
        read -s -p "请输入目标仓库 $TARGET_REGISTRY 的密码: " TARGET_PASSWORD
        echo
    fi
    
    # 执行登录
    if ! echo "$TARGET_PASSWORD" | docker login $TARGET_REGISTRY -u $TARGET_USER --password-stdin; then
        log "错误: 登录目标镜像仓库失败"
        exit 1
    fi
    log "登录成功"
}

# 处理单个镜像
process_image() {
    local image=$1
    
    # 如果源仓库非空，则拼接完整的源镜像名
    if [ -n "$SOURCE_REGISTRY" ]; then
        local source_image="$SOURCE_REGISTRY/$image"
    else
        local source_image="$image"
    fi
    
    # 构建目标镜像名
    local target_image="$TARGET_REGISTRY/$image"
    
    # 生成tar文件名（替换/和:为-）
    local tar_filename=$(echo "$image" | tr '/:' '-')".tar"
    local tar_path="$OUTPUT_DIR/$tar_filename"
    
    log "----------------------------------------"
    log "开始处理镜像: $image"
    log "源镜像: $source_image"
    log "目标镜像: $target_image"
    log "tar文件: $tar_path"
    
    # 1. 拉取镜像
    log "拉取镜像..."
    if ! docker pull "$source_image"; then
        log "错误: 拉取镜像 $source_image 失败"
        return 1
    fi
    
    # 2. 保存镜像为tar文件
    log "保存镜像为tar文件..."
    if ! docker save -o "$tar_path" "$source_image"; then
        log "错误: 保存镜像 $source_image 失败"
        return 1
    fi
    
    # 3. 为镜像打目标标签
    log "为镜像打目标标签..."
    if ! docker tag "$source_image" "$target_image"; then
        log "错误: 为镜像打标签失败"
        return 1
    fi
    
    # 4. 推送镜像到目标仓库
    log "推送镜像到目标仓库..."
    if ! docker push "$target_image"; then
        log "错误: 推送镜像 $target_image 失败"
        return 1
    fi
    
    # 5. 清理临时镜像（可选）
    log "清理临时镜像..."
    docker rmi "$target_image"
    
    log "镜像 $image 处理完成"
    return 0
}

# 主函数
main() {
    # 初始化日志
    echo "===== 镜像批量处理开始于 $(date) =====" > $LOG_FILE
    
    # 检查依赖
    check_dependencies
    
    # 登录目标仓库
    login_to_target
    
    # 读取镜像列表并处理
    local success_count=0
    local fail_count=0
    local total_count=$(wc -l < $IMAGES_LIST)
    
    log "发现 $total_count 个镜像，开始处理..."
    
    while IFS= read -r image; do
        # 跳过空行
        if [ -z "$image" ]; then
            continue
        fi
        
        # 处理镜像
        process_image "$image"
        
        if [ $? -eq 0 ]; then
            ((success_count++))
        else
            ((fail_count++))
        fi
    done < "$IMAGES_LIST"
    
    # 处理结果汇总
    log "----------------------------------------"
    log "处理完成！总计: $total_count, 成功: $success_count, 失败: $fail_count"
    log "详细日志请查看: $LOG_FILE"
    log "===== 镜像批量处理结束于 $(date) =====" >> $LOG_FILE
    
    # 登出目标仓库
    docker logout $TARGET_REGISTRY
}

# 执行主函数
main
    
