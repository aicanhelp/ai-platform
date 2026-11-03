#!/bin/bash

# 离线环境检测工具 - Containerd 版本
# 功能：检查网络连通性、检测所有残留镜像和容器

set -e

# 颜色定义（仅用于终端显示）
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# 日志文件
LOG_FILE="./offline_env_check.log"
CONFIG_FILE="./checker_config.conf"

# 输出函数
log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="[INFO][$timestamp] $1"
    echo -e "${BLUE}${message}${NC}"          # 终端显示带颜色
    echo "$message" >> "$LOG_FILE"            # 日志文件无颜色
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="[SUCCESS][$timestamp] $1"
    echo -e "${GREEN}${message}${NC}"          
    echo "$message" >> "$LOG_FILE"            
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="[WARNING][$timestamp] $1"
    echo -e "${YELLOW}${message}${NC}"        
    echo "$message" >> "$LOG_FILE"            
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="[ERROR][$timestamp] $1"
    echo -e "${RED}${message}${NC}"           
    echo "$message" >> "$LOG_FILE"          
}

load_config_advanced() {
    local config_file="$1"
    
    # 预处理：移除 Windows 换行符
    local temp_file=$(mktemp)
    sed 's/\r$//' "$config_file" > "$temp_file"
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # 跳过注释和空行
        line=$(echo "$line" | sed 's/#.*$//')
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$line" ]] && continue
        
        # 解析数组
        if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*)=\((.*)\) ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local array_content="${BASH_REMATCH[2]}"
            
            # 安全地定义数组
            IFS=' ' read -ra "$var_name" <<< "$array_content"
            log_info "设置数组: $var_name=($array_content)"
            
        # 解析普通变量
        elif [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*)=([^[:space:]]+)$ ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local var_value="${BASH_REMATCH[2]}"
            
            declare -g "$var_name"="$var_value"
            log_info "设置变量: $var_name=$var_value"
        fi
    done < "$temp_file"
    
    rm -f "$temp_file"
}

# 配置加载函数 - 必须先定义和调用
load_config() {
    # 设置默认值
    NETWORK_TARGETS=("www.baidu.com" "www.google.com" "182.61.202.46" "142.250.199.206")
    CONTAINERD_NAMESPACES=("k8s.io")
    CHECK_DIRS=("/var/lib/containerd" "/var/lib/docker")
    CLEANUP_DIRS=("/var/lib/containerd" "/var/lib/docker")
    SIZE_THRESHOLD=10
    SAFE_MODE=true
    
    # 如果配置文件存在，用配置文件覆盖默认值
    if [[ -f "$CONFIG_FILE" ]]; then
        log_info "加载配置文件: $CONFIG_FILE"
        load_config_advanced "$CONFIG_FILE"
        log_info "配置覆盖完成"
    else
        log_warning "配置文件不存在，使用默认配置"
    fi
}

# 检查网络连通性
check_network() {
    log_info "开始网络连通性检查..."
    local online_count=0
    
    for target in "${NETWORK_TARGETS[@]}"; do
        if ping -c 2 -W 2 "$target" &> /dev/null; then
            log_error "网络连通: 可以访问 $target - 环境可能不是完全离线!"
            online_count=$((online_count + 1))
        else
            log_success "网络断开: 无法访问 $target"
        fi
    done
    
    # 检查HTTP代理
    if [[ -n "$http_proxy" || -n "$HTTP_PROXY" ]]; then
        log_warning "检测到HTTP代理设置，可能会影响离线环境"
    fi
    
    if [[ $online_count -gt 0 ]]; then
        log_error "网络检查失败: 发现 $online_count 个在线连接"
        return 1
    else
        log_success "网络检查通过: 环境处于离线状态"
        return 0
    fi
}

# 检查 containerd 是否运行
check_containerd_status() {
    log_info "检查 Containerd 状态..."
    
    if ! systemctl is-active --quiet containerd 2>/dev/null; then
        log_warning "Containerd 未运行"
        return 1
    else
        log_success "Containerd 正在运行"
        return 0
    fi
}

# 检查所有 containerd 命名空间中的镜像
check_all_containerd_images() {
    log_info "检查所有 Containerd 命名空间中的镜像..."
    
    if ! command -v ctr &> /dev/null; then
        log_warning "ctr 命令未找到，跳过镜像检查"
        return 0
    fi
    
    local total_images=0
    local found_images=false
    
    for namespace in "${CONTAINERD_NAMESPACES[@]}"; do
        log_info "检查命名空间: $namespace"
        
        # 检查镜像
        local images=$(ctr -n "$namespace" images list -q 2>/dev/null || true)
        
        if [[ -n "$images" ]]; then
            found_images=true
            local count=0
            
            while IFS= read -r image; do
                if [[ -n "$image" ]]; then
                    log_error "发现残留镜像 [$namespace]: $image"
                    count=$((count + 1))
                    total_images=$((total_images + 1))
                fi
            done <<< "$images"
            
            log_info "命名空间 $namespace 中发现 $count 个镜像"
        else
            log_success "命名空间 $namespace 中没有发现镜像"
        fi
    done
    
    if [[ "$found_images" == "true" ]]; then
        log_error "镜像检查失败: 总共发现 $total_images 个残留镜像"
        return 1
    else
        log_success "镜像检查通过: 未发现任何镜像"
        return 0
    fi
}

# 检查 crictl 容器状态（Kubernetes 环境）
check_crictl_resources() {
    log_info "检查 CRI 资源..."
    
    if ! command -v crictl &> /dev/null; then
        log_info "crictl 未安装，跳过 CRI 资源检查"
        return 0
    fi
    
    local found_resources=false
    
    # 检查镜像
    local images=$(crictl images --quiet 2>/dev/null)
    local image_count=0
    if [[ -n "$images" ]]; then
        found_resources=true
        image_count=$(echo "$images" | wc -l)
        log_error "发现 $image_count 个 CRI 镜像:"
        crictl images 2>/dev/null | while read -r image; do
            if [[ -n "$image" && "$image" != "IMAGE ID"* ]]; then
                log_error "  - $image"
            fi
        done
    fi
    
    if [[ "$found_resources" == "true" ]]; then
        log_error "CRI 资源检查失败: 发现 $image_count 个镜像"
        return 1
    else
        log_success "CRI 资源检查通过: 未发现任何资源"
        return 0
    fi
}

# 检查文件系统残留
check_filesystem() {
    log_info "检查文件系统残留..."
    local found_residual=false
    
    # 检查目录
    for dir in "${CHECK_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            local size=$(du -sm "$dir" 2>/dev/null | cut -f1 || echo "0")
            if [[ "$size" -gt $SIZE_THRESHOLD ]]; then
                log_error "发现容器相关目录: $dir (${size}MB)"
                found_residual=true
            fi
        fi
    done
    
    if [[ "$found_residual" == "true" ]]; then
        return 1
    else
        log_success "文件系统检查通过"
        return 0
    fi
}

# 生成报告
generate_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 终端显示
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}离线环境检测报告 - $timestamp${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}主机名: $(hostname)${NC}"
    echo -e "${BLUE}IP地址: $(hostname -I 2>/dev/null || echo '未知')${NC}"
    echo -e "${BLUE}容器运行时: Containerd${NC}"
    echo -e "${BLUE}================================================${NC}"
    
    # 日志文件记录（无颜色）
    echo "================================================" >> "$LOG_FILE"
    echo "离线环境检测报告 - $timestamp" >> "$LOG_FILE"
    echo "================================================" >> "$LOG_FILE"
    echo "主机名: $(hostname)" >> "$LOG_FILE"
    echo "IP地址: $(hostname -I 2>/dev/null || echo '未知')" >> "$LOG_FILE"
    echo "容器运行时: Containerd" >> "$LOG_FILE"
    echo "================================================" >> "$LOG_FILE"
}

# 主函数
main() {
    log_info "开始离线环境检测..."
    
    # 初始化日志
    echo "=== 离线环境检测开始 ===" > "$LOG_FILE"
    
    generate_report
    
    local overall_success=true
    
    # 执行各项检查
    if ! check_network; then
        overall_success=false
    fi

    if ! check_filesystem; then
        overall_success=false
    fi
    
    if ! check_containerd_status; then
        overall_success=false
    fi
    
    if ! check_all_containerd_images; then
        overall_success=false
    fi
    
    if ! check_crictl_resources; then
        overall_success=false
    fi
    
    echo "================================================" | tee -a "$LOG_FILE"
    if [[ "$overall_success" == "true" ]]; then
        log_success "所有检查通过！环境干净，适合测试"
        exit 0
    else
        log_error "检查未通过！发现残留镜像、容器或网络连接"
        log_info "使用 '$(basename "$0") --clean' 清理所有残留资源"
        exit 1
    fi
}

# 清理文件系统残留 - 直接删除文件夹
cleanup_filesystem() {
    log_info "开始清理文件系统残留..."
    
    if [[ "$SAFE_MODE" == "true" ]]; then
        log_warning "警告：这将直接删除以下文件夹及其所有内容："
        for dir in "${CLEANUP_DIRS[@]}"; do
            log_warning "$dir"
        done
        log_warning "此操作不可恢复，是否继续？[y/N]"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "用户取消文件系统清理操作"
            return 0
        fi
    fi
    
    local cleaned_count=0
    local error_count=0
    
    # 遍历所有配置的清理目录
    for dir in "${CLEANUP_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            log_info "删除文件夹: $dir"
            
            # 直接删除整个文件夹
            if rm -rf "$dir" 2>/dev/null; then
                log_success "成功删除文件夹: $dir"
                cleaned_count=$((cleaned_count + 1))
            else
                log_error "无法删除文件夹: $dir (权限不足或目录不存在)"
                error_count=$((error_count + 1))
            fi
        else
            log_info "文件夹不存在，跳过: $dir"
        fi
    done
    
    if [[ $error_count -eq 0 ]]; then
        log_success "文件系统清理完成: 成功删除 $cleaned_count 个文件夹"
        return 0
    else
        log_warning "文件系统清理完成: 成功删除 $cleaned_count 个文件夹，失败 $error_count 个"
        return 1
    fi
}

# 全面清理函数
cleanup_all_resources() {
    log_warning "开始全面清理容器资源..."
    
    if [[ "$SAFE_MODE" == "true" ]]; then
        echo -e "${YELLOW}警告：这将删除所有容器镜像和容器，是否继续？[y/N]${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "用户取消操作"
            exit 0
        fi
    fi
    
    # 停止并删除所有 CRI 资源
    if command -v crictl &> /dev/null; then
        log_info "清理 CRI 资源..."
        
        # 停止所有 Pod
        crictl pods --quiet 2>/dev/null | while read -r pod; do
            if [[ -n "$pod" ]]; then
                log_info "停止 Pod: $pod"
                crictl stopp "$pod" 2>/dev/null || true
                crictl rmp "$pod" 2>/dev/null || true
            fi
        done
        
        # 停止所有容器
        crictl ps --quiet 2>/dev/null | while read -r container; do
            if [[ -n "$container" ]]; then
                log_info "停止容器: $container"
                crictl stop "$container" 2>/dev/null || true
                crictl rm "$container" 2>/dev/null || true
            fi
        done
        
        # 删除所有镜像
        crictl images --quiet 2>/dev/null | while read -r image; do
            if [[ -n "$image" ]]; then
                log_info "删除镜像: $image"
                crictl rmi "$image" 2>/dev/null || true
            fi
        done
    fi
    
    # 清理所有 containerd 命名空间
    for namespace in "${CONTAINERD_NAMESPACES[@]}"; do
        log_info "清理命名空间: $namespace"
        
        # 删除所有容器
        if command -v ctr &> /dev/null; then
            ctr -n "$namespace" containers list -q 2>/dev/null | while read -r container; do
                if [[ -n "$container" ]]; then
                    log_info "删除容器: $container"
                    ctr -n "$namespace" containers delete "$container" 2>/dev/null || true
                fi
            done
            
            # 删除所有镜像
            ctr -n "$namespace" images list -q 2>/dev/null | while read -r image; do
                if [[ -n "$image" ]]; then
                    log_info "删除镜像: $image"
                    ctr -n "$namespace" images remove "$image" 2>/dev/null || true
                fi
            done
        fi
    done
    
    # 清理 containerd 存储
    log_info "清理 containerd 存储..."
    if command -v ctr &> /dev/null; then
        ctr content ls -q 2>/dev/null | while read -r content; do
            if [[ -n "$content" ]]; then
                ctr content rm "$content" 2>/dev/null || true
            fi
        done
    fi
    
    cleanup_filesystem
    
    log_success "清理完成！所有容器资源已被清除"
}

# 显示帮助
show_help() {
    echo "离线环境检测工具 - Containerd 版本"
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -c, --clean         清理所有容器资源（镜像、容器等）"
    echo "  -f, --force         强制清理（跳过确认）"
    echo "  -l, --list          列出所有发现的资源"
}

# 列出所有资源
list_all_resources() {
    log_info "列出所有容器资源..."
    
    echo "=== Containerd 镜像 ==="
    for namespace in "${CONTAINERD_NAMESPACES[@]}"; do
        echo "命名空间: $namespace"
        ctr -n "$namespace" images list 2>/dev/null || echo "  无镜像"
    done
    
    echo "=== Containerd 容器 ==="
    for namespace in "${CONTAINERD_NAMESPACES[@]}"; do
        echo "命名空间: $namespace"
        ctr -n "$namespace" containers list 2>/dev/null || echo "  无容器"
    done
    
    if command -v crictl &> /dev/null; then
        echo "=== CRI Pods ==="
        crictl pods 2>/dev/null || echo "  无 Pod"
        
        echo "=== CRI 容器 ==="
        crictl ps 2>/dev/null || echo "  无容器"
        
        echo "=== CRI 镜像 ==="
        crictl images 2>/dev/null || echo "  无镜像"
    fi
}

# 参数处理
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -c|--clean)
        load_config
        cleanup_all_resources
        exit 0
        ;;
    -f|--force)
        load_config
        SAFE_MODE=false
        cleanup_all_resources
        exit 0
        ;;
    -l|--list)
        load_config
        list_all_resources
        exit 0
        ;;
    *)
        load_config
        main
        ;;
esac