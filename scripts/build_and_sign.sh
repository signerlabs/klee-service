#!/bin/bash

# 日志文件 - 按日期划分
TODAY=$(date '+%Y%m%d')
LOG_FILE="logs/build_sign_${TODAY}.log"

# 确保日志目录存在
mkdir -p logs

VERSION_RECORD_FILE=".build_version"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # 无颜色

# 进度条函数
show_progress() {
    local message="$1"
    local progress="$2"
    local total="$3"
    local percent=$((progress * 100 / total))
    local completed=$((progress * 50 / total))
    local remaining=$((50 - completed))
    
    # 构建进度条
    local progress_bar="["
    for ((i=0; i<completed; i++)); do
        progress_bar+="="
    done
    
    if [ $completed -lt 50 ]; then
        progress_bar+=">"
        for ((i=0; i<remaining-1; i++)); do
            progress_bar+=" "
        done
    else
        progress_bar+="="
    fi
    
    progress_bar+="] ${percent}%"
    
    # 输出进度条和消息
    echo -ne "\r${CYAN}$message $progress_bar${NC}"
    
    # 如果完成，换行
    if [ $progress -eq $total ]; then
        echo ""
    fi
}

# 日志函数
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message"
    echo "$message" >> "$LOG_FILE"
}

# 彩色日志
log_info() {
    log "${BLUE}信息: $1${NC}"
}

log_success() {
    log "${GREEN}成功: $1${NC}"
}

log_warning() {
    log "${YELLOW}警告: $1${NC}"
}

log_error() {
    log "${RED}错误: $1${NC}"
}

# 错误处理
handle_error() {
    log_error "$1"
    exit 1
}

# 用户确认函数
confirm_step() {
    local prompt="$1"
    local default="$2"
    
    # 显示提示并获取用户输入
    if [ "$default" = "Y" ]; then
        read -p "$prompt [Y/n]: " response
        response=${response:-Y}
    else
        read -p "$prompt [y/N]: " response
        response=${response:-N}
    fi
    
    # 检查响应
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# 允许用户修改文件名
customize_filename() {
    local original_path="$1"
    local file_type="$2"  # "DMG" 或 "ZIP" 或 "TAR.GZ"
    
    # 获取原始文件名
    local dir_name=$(dirname "$original_path")
    local base_name=$(basename "$original_path")
    
    log_info "生成的${file_type}文件: $base_name"
    log_info "完整路径: $original_path"
    
    if confirm_step "是否要自定义${file_type}文件名?" "N"; then
        log_info "请输入新的文件名(不含路径和扩展名):"
        log_info "当前名称: ${base_name%.*}"
        read new_name
        
        if [ -z "$new_name" ]; then
            log_warning "未提供新文件名，保持原名: $base_name"
            echo "$original_path"
            return 0
        fi
        
        # 构建新路径
        local extension="${base_name##*.}"
        local new_path="${dir_name}/${new_name}.${extension}"
        
        log_info "新的文件路径: $new_path"
        if [ -f "$new_path" ]; then
            if confirm_step "文件 $new_path 已存在，是否覆盖?" "N"; then
                mv "$original_path" "$new_path"
                log_success "文件已重命名为: $new_path"
                echo "$new_path"
                return 0
            else
                log_warning "取消覆盖，保持原名: $base_name"
                echo "$original_path"
                return 0
            fi
        else
            mv "$original_path" "$new_path"
            log_success "文件已重命名为: $new_path"
            echo "$new_path"
            return 0
        fi
    else
        log_info "保持原文件名: $base_name"
        echo "$original_path"
        return 0
    fi
}

# 检测平台函数
detect_platform() {
    local platform=$(uname -s)
    echo "$platform"
}

# 交互式获取版本号
get_version_interactive() {
    # 检查常见的版本文件位置
    if [ -f "version.txt" ]; then
        local version=$(cat version.txt | tr -d '\r\n')
        log_info "从version.txt检测到版本号: $version"
        
        if confirm_step "是否使用检测到的版本号'$version'?" "Y"; then
            echo "$version"
            return 0
        fi
    fi
    
    if [ -f "app/__init__.py" ]; then
        local version=$(grep -E "__version__\s*=\s*['\"]" app/__init__.py | sed -E "s/__version__\s*=\s*['\"]([^'\"]+)['\"].*/\1/")
        if [ ! -z "$version" ]; then
            log_info "从Python包检测到版本号: $version"
            
            if confirm_step "是否使用检测到的版本号'$version'?" "Y"; then
                echo "$version"
                return 0
            fi
        fi
    fi
    
    # 如果没有检测到版本或用户拒绝使用检测到的版本，提示输入
    log_info "请输入应用版本号(例如: 1.0.0):"
    read user_version
    
    if [ -z "$user_version" ]; then
        log_warning "未提供版本号，将使用时间戳作为版本"
        user_version="$(date '+%Y%m%d%H%M%S')"
    fi
    
    # 询问是否保存版本到version.txt
    if confirm_step "是否将版本号'$user_version'保存到version.txt文件?" "Y"; then
        echo "$user_version" > version.txt
        log_success "已保存版本号到version.txt"
    fi
    
    echo "$user_version"
    return 0
}

# 获取当前版本号
get_current_version() {
    local version=""
    
    # 方法1: 从version.txt文件获取版本
    log_info "开始获取版本号..."
    log_info "尝试从version.txt获取版本号..."
    if [ -f "version.txt" ]; then
        version=$(cat version.txt | tr -d '\r\n')
        log_info "从version.txt获取版本号: $version"
        echo "$version"
        return 0
    fi
    log_info "尝试从version.txt获取版本号失败"
    
    log_info "尝试从Python包获取版本号..."
    # 方法2: 从Python包的__version__获取
    if [ -f "app/__init__.py" ]; then
        version=$(grep -E "__version__\s*=\s*['\"]" app/__init__.py | sed -E "s/__version__\s*=\s*['\"]([^'\"]+)['\"].*/\1/")
        if [ ! -z "$version" ]; then
            log_info "从Python包获取版本号: $version"
            echo "$version"
            return 0
        fi
    fi
    log_info "尝试从Python包获取版本号失败"
    
    # 如果无法自动检测版本，切换到交互式模式
    log_info "切换到交互式模式获取版本号..."
    version=$(get_version_interactive)
    echo "$version"
}

# 检查是否需要重新编译
need_rebuild() {
    local current_version="$1"
    local app_name="$2"
    
    # 如果没有版本记录文件，需要重新编译
    if [ ! -f "$VERSION_RECORD_FILE" ]; then
        log_info "无版本记录文件，需要重新编译"
        return 0 # 需要重新编译
    fi
    
    # 读取记录的版本号
    local recorded_version=$(cat "$VERSION_RECORD_FILE")
    log_info "记录的版本号: $recorded_version"
    log_info "当前版本号: $current_version"
    
    # 如果版本不同，需要重新编译
    if [ "$current_version" != "$recorded_version" ]; then
        log_info "版本不同，需要重新编译"
        return 0 # 需要重新编译
    fi
    
    # 检查编译产物是否存在
    if [ ! -d "dist/$app_name" ]; then
        log_info "编译产物不存在，需要重新编译"
        return 0 # 需要重新编译
    fi
    
    log_info "当前版本已经编译，可以跳过编译步骤"
    return 1 # 不需要重新编译
}

# 记录当前版本
record_version() {
    local version="$1"
    echo "$version" > "$VERSION_RECORD_FILE"
    log_info "记录版本号: $version"
}

# 根据平台选择编译命令
build_by_platform() {
    local platform="$1"
    log_info "检测到平台: $platform"
    
    case "$platform" in
        Darwin)
            log_info "执行macOS平台编译命令..."
            python -m PyInstaller --clean main.spec
            ;;
        Linux)
            log_info "执行Linux平台编译命令..."
            python -m PyInstaller --clean main.spec
            ;;
        MINGW*|CYGWIN*|MSYS*)
            log_info "执行Windows平台编译命令..."
            python -m PyInstaller --clean main.spec
            ;;
        *)
            log_info "未知平台: $platform，尝试执行通用编译命令..."
            python -m PyInstaller --clean main.spec
            ;;
    esac
    
    return $?
}

# 获取编译产物名称
get_app_name() {
    # 如果dist目录不存在，返回空
    if [ ! -d "dist" ]; then
        return 1
    fi
    
    # 获取dist目录中的第一个项目
    local app_name=$(ls dist | head -n 1)
    if [ -z "$app_name" ]; then
        return 1
    fi
    
    echo "$app_name"
    return 0
}

# 签名相关功能（从sign.sh集成）
# ===========================================================

# 签名一个文件
sign_file() {
    local file_path="$1"
    local cert_hash="$2"
    local entitlements_path="$3"
    local file_name=$(basename "$file_path")
    
    if [ ! -f "$file_path" ]; then
        log_warning "跳过签名: 文件不存在 - $file_path"
        return 1
    fi
    
    # 判断是否需要使用entitlements
    if [ -n "$entitlements_path" ] && [ -f "$entitlements_path" ]; then
        codesign --force --timestamp --options=runtime --entitlements "$entitlements_path" --sign "$cert_hash" "$file_path" 2>/dev/null
    else
        codesign --force --timestamp --options=runtime --sign "$cert_hash" "$file_path" 2>/dev/null
    fi
    
    # 检查签名结果
    if [ $? -ne 0 ]; then
        log_error "签名失败: $file_path"
        return 1
    fi
    
    return 0
}

# 验证文件签名
verify_signature() {
    local file_path="$1"
    
    codesign -v "$file_path" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        log_error "签名验证失败: $file_path"
        return 1
    fi
    
    return 0
}

# 签名Python相关文件和框架
sign_python_files() {
    local app_path="$1"
    local cert_hash="$2"
    local entitlements_path="$3"
    
    log_info "开始签名Python相关文件和框架..."
    
    # 检查Python框架是否存在
    local framework_path="${app_path}/_internal/Python.framework"
    if [ -d "$framework_path" ]; then
        # 获取要签名的文件总数
        local total_files=$(find "$framework_path" -type f -not -path "*/\.*" | wc -l | tr -d ' ')
        local current=0
        
        log_info "检测到 $total_files 个Python框架文件需要签名"
        
        find "$framework_path" -type f -not -path "*/\.*" | while read file; do
            current=$((current + 1))
            show_progress "签名Python框架文件" $current $total_files
            sign_file "$file" "$cert_hash" ""
        done
        
        # 签名主框架
        log_info "签名Python主框架..."
        sign_file "$framework_path" "$cert_hash" "$entitlements_path"
        verify_signature "$framework_path"
        
        log_success "Python框架签名完成"
    else
        log_warning "未找到Python.framework，跳过此步骤"
    fi
    
    # 签名Python解释器
    local python_path="${app_path}/_internal/Python"
    if [ -f "$python_path" ]; then
        log_info "签名Python解释器..."
        sign_file "$python_path" "$cert_hash" "$entitlements_path"
        verify_signature "$python_path"
        log_success "Python解释器签名完成"
    else
        log_warning "未找到Python解释器，跳过此步骤"
    fi
}

# 签名主可执行文件
sign_main_executable() {
    local app_path="$1"
    local cert_hash="$2"
    local entitlements_path="$3"
    
    log_info "签名主可执行文件..."
    
    local main_exec="${app_path}/$(basename "$app_path")"
    if [ -f "$main_exec" ]; then
        sign_file "$main_exec" "$cert_hash" "$entitlements_path"
        
        # 验证签名
        if verify_signature "$main_exec"; then
            log_success "主可执行文件签名成功: $main_exec"
        else
            log_error "主可执行文件签名验证失败!"
            return 1
        fi
    else
        log_error "未找到主可执行文件: $main_exec"
        return 1
    fi
    
    return 0
}

# 签名_internal目录下的所有文件
sign_internal_files() {
    local app_path="$1"
    local cert_hash="$2"
    local entitlements_path="$3"
    
    log_info "开始签名_internal目录中的文件..."
    
    local internal_path="${app_path}/_internal"
    if [ ! -d "$internal_path" ]; then
        log_error "_internal目录不存在: $internal_path"
        return 1
    fi
    
    # 计算要签名的文件总数（排除已签名的Python和框架）
    local total_files=$(find "$internal_path" -type f -not -path "*/Python.framework/*" -not -path "*/Python" -not -path "*/\.*" | wc -l | tr -d ' ')
    local current=0
    
    log_info "检测到 $total_files 个文件需要签名"
    
    find "$internal_path" -type f -not -path "*/Python.framework/*" -not -path "*/Python" -not -path "*/\.*" | while read file; do
        current=$((current + 1))
        show_progress "签名内部文件" $current $total_files
        sign_file "$file" "$cert_hash" ""
    done
    
    log_success "_internal目录文件签名完成"
    return 0
}

# 执行完整签名流程
perform_signing() {
    local app_path="$1"
    local cert_hash="$2"
    local entitlements_path="$3"
    
    log_info "开始执行完整签名流程..."
    
    # 1. 签名Python相关文件
    sign_python_files "$app_path" "$cert_hash" "$entitlements_path"
    if [ $? -ne 0 ]; then
        log_warning "Python文件签名可能存在问题，继续执行下一步..."
    fi
    
    # 2. 签名主可执行文件
    sign_main_executable "$app_path" "$cert_hash" "$entitlements_path"
    if [ $? -ne 0 ]; then
        log_error "主可执行文件签名失败，中止签名流程"
        return 1
    fi
    
    # 3. 签名_internal目录下的文件
    sign_internal_files "$app_path" "$cert_hash" "$entitlements_path"
    if [ $? -ne 0 ]; then
        log_warning "内部文件签名可能存在问题，但继续执行下一步..."
    fi
    
    log_success "所有签名步骤完成"
    return 0
}

# ===========================================================

# 主函数
main() {
    log_info "===== 开始构建和签名流程 ====="
    log_info "日志文件: $LOG_FILE"
    
    # 1. 检测平台
    local platform=$(detect_platform)
    log_info "检测到系统平台: $platform"
    
    # 2. 获取当前版本
    local current_version=$(get_current_version)
    log_info "当前版本: $current_version"
    
    # 3. 获取应用名称(如果已经编译)
    local app_name=$(get_app_name)
    
    # 4. 判断是否需要编译
    local need_build=1
    if [ ! -z "$app_name" ]; then
        need_rebuild "$current_version" "$app_name"
        need_build=$?
        
        if [ $need_build -eq 1 ]; then
            log_info "检测到已经构建的应用: $app_name (版本: $current_version)"
            if confirm_step "是否跳过编译步骤，直接进行签名/打包?" "Y"; then
                need_build=0
            else
                need_build=1
            fi
        fi
    fi
    
    if [ $need_build -eq 1 ]; then
        # 显示编译步骤信息并询问用户
        log_info "即将执行构建步骤，这将会:"
        log_info "1. 使用PyInstaller构建独立的可执行应用"
        log_info "2. 生成dist/目录下的应用产物"
        log_info "3. 预计耗时5-10分钟(根据项目大小)"
        
        if ! confirm_step "是否继续执行构建步骤?" "Y"; then
            log_warning "用户取消了构建步骤，退出程序"
            exit 0
        fi
        
        # 执行构建
        log_info "开始执行编译命令..."
        build_by_platform "$platform"
        
        # 检查编译是否成功
        if [ $? -ne 0 ]; then
            handle_error "编译失败，无法继续签名步骤"
        fi
        
        log_success "编译成功完成"
        
        # 记录当前版本
        record_version "$current_version"
        
        # 重新获取应用名称
        app_name=$(get_app_name)
        if [ -z "$app_name" ]; then
            handle_error "编译后未找到应用"
        fi
    else
        log_info "跳过编译步骤"
    fi
    
    local app_path="dist/$app_name"
    log_info "应用路径: $app_path"
    
    # 创建release目录
    mkdir -p releases
    log_info "创建releases目录成功"
    
    # 5. 开始签名流程
    if [ "$platform" = "Darwin" ]; then
        log_info "即将执行macOS签名步骤，这将会:"
        log_info "1. 为应用中的可执行文件添加代码签名"
        log_info "2. 验证签名的有效性"
        log_info "3. 生成已签名的应用包"
        
        if ! confirm_step "是否继续执行签名步骤?" "Y"; then
            log_warning "用户取消了签名步骤，将直接进行打包"
        else
            # 使用集成的签名函数替代外部脚本
            log_info "开始执行签名流程..."
            perform_signing "$app_path" "$CERT_HASH" "scripts/entitlements.plist"
            
            if [ $? -ne 0 ]; then
                log_error "签名过程失败"
                if ! confirm_step "签名失败，是否继续打包未签名的应用?" "N"; then
                    exit 1
                fi
            else
                log_success "签名成功完成"
            fi
        fi
        
        # 6. 打包步骤
        log_info "即将执行打包步骤，这将会:"
        log_info "1. 创建可分发的应用包(DMG或ZIP)"
        log_info "2. 保存至releases/目录"
        log_info "3. 最终产品将包含应用版本号"
        
        if ! confirm_step "是否继续执行打包步骤?" "Y"; then
            log_warning "用户取消了打包步骤，退出程序"
            exit 0
        fi
        
        # 创建DMG镜像
        log_info "尝试创建DMG镜像..."
        if command -v hdiutil &> /dev/null; then
            local dmg_name="releases/${app_name}-${current_version}.dmg"
            hdiutil create -volname "$app_name" -srcfolder "$app_path" -ov -format UDZO "$dmg_name"
            
            if [ $? -eq 0 ]; then
                log_success "DMG镜像创建成功"
                # 提供文件名自定义选项
                dmg_name=$(customize_filename "$dmg_name" "DMG")
                log_success "最终DMG文件: $(basename "$dmg_name")"
            else
                log_warning "DMG镜像创建失败，将使用ZIP格式"
                if [ -f "${app_name}.zip" ]; then
                    local zip_name="releases/${app_name}-${current_version}.zip"
                    mv "${app_name}.zip" "$zip_name"
                    # 提供文件名自定义选项
                    zip_name=$(customize_filename "$zip_name" "ZIP")
                    log_success "最终ZIP文件: $(basename "$zip_name")"
                else
                    # 创建ZIP包
                    cd dist
                    local zip_name="../releases/${app_name}-${current_version}.zip"
                    zip -r "$zip_name" "$app_name"
                    cd ..
                    # 提供文件名自定义选项
                    zip_name=$(customize_filename "$zip_name" "ZIP")
                    log_success "最终ZIP文件: $(basename "$zip_name")"
                fi
            fi
        else
            log_warning "未发现hdiutil命令，使用ZIP格式"
            if [ -f "${app_name}.zip" ]; then
                local zip_name="releases/${app_name}-${current_version}.zip"
                mv "${app_name}.zip" "$zip_name"
                # 提供文件名自定义选项
                zip_name=$(customize_filename "$zip_name" "ZIP")
                log_success "最终ZIP文件: $(basename "$zip_name")"
            else
                # 创建ZIP包
                cd dist
                local zip_name="../releases/${app_name}-${current_version}.zip"
                zip -r "$zip_name" "$app_name"
                cd ..
                # 提供文件名自定义选项
                zip_name=$(customize_filename "$zip_name" "ZIP")
                log_success "最终ZIP文件: $(basename "$zip_name")"
            fi
        fi
    else
        # 非macOS平台的打包步骤
        log_info "即将在${platform}平台执行打包步骤，这将会:"
        log_info "1. 创建可分发的应用包(ZIP或TAR.GZ)"
        log_info "2. 保存至releases/目录"
        log_info "3. 最终产品将包含应用版本号"
        
        if ! confirm_step "是否继续执行打包步骤?" "Y"; then
            log_warning "用户取消了打包步骤，退出程序"
            exit 0
        fi
        
        # 创建压缩包
        log_info "创建应用压缩包..."
        
        # 检测平台并选择合适的压缩命令
        if command -v zip &> /dev/null; then
            cd dist
            local zip_name="../releases/${app_name}-${current_version}.zip"
            zip -r "$zip_name" "$app_name"
            cd ..
            # 提供文件名自定义选项
            zip_name=$(customize_filename "$zip_name" "ZIP")
            log_success "最终ZIP文件: $(basename "$zip_name")"
        elif command -v tar &> /dev/null; then
            cd dist
            local tar_name="../releases/${app_name}-${current_version}.tar.gz"
            tar -czf "$tar_name" "$app_name"
            cd ..
            # 提供文件名自定义选项
            tar_name=$(customize_filename "$tar_name" "TAR.GZ")
            log_success "最终TAR.GZ文件: $(basename "$tar_name")"
        else
            log_error "无法找到合适的压缩工具(zip或tar)，无法完成打包"
            if confirm_step "是否复制未打包的应用到releases目录?" "Y"; then
                local folder_name="releases/${app_name}-${current_version}"
                mkdir -p "$folder_name"
                cp -R "$app_path/"* "$folder_name/"
                log_success "已复制应用到$folder_name/"
            fi
        fi
    fi
    
    log_success "===== 构建和签名流程全部完成 ====="
    log_info "最终产品位于 releases/ 目录"
    log_info "产品版本: $current_version"
    log_info "今日构建日志: $LOG_FILE"
}

# 执行主函数
main 