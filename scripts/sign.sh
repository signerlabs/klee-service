#!/bin/bash

# 定义变量
CERT_HASH="55D8EDB450467EF4D0E22F606118C87C9130E85B"
APP_PATH="main"
ENTITLEMENTS_PATH="entitlements.plist"

find main -name ".DS_Store" -delete
find main -name "__MACOSX" -delete

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 签名 Python 相关文件
sign_python_files() {
    log "开始签名 Python 相关文件..."
    
    # 签名 Python 二进制
    if [ -f "$APP_PATH/_internal/Python" ]; then
        log "签名 Python 二进制..."
        codesign --force --deep \
                 --sign "$CERT_HASH" \
                 --options runtime \
                 --entitlements "$ENTITLEMENTS_PATH" \
                 --timestamp \
                 "$APP_PATH/_internal/Python"
        
        if [ $? -eq 0 ]; then
            log "Python 二进制签名成功"
        else
            log "错误: Python 二进制签名失败"
        fi
    fi
    
    # 签名 Python.framework
    if [ -f "$APP_PATH/_internal/Python.framework/Python" ]; then
        log "签名 Python.framework..."
        codesign --force --deep \
                 --sign "$CERT_HASH" \
                 --options runtime \
                 --entitlements "$ENTITLEMENTS_PATH" \
                 --timestamp \
                 "$APP_PATH/_internal/Python.framework/Python"
        
        if [ $? -eq 0 ]; then
            log "Python.framework 签名成功"
        else
            log "错误: Python.framework 签名失败"
        fi
    fi
}

# 签名主程序
sign_main_executable() {
    log "开始签名主程序..."
    
    codesign --force --deep \
             --sign "$CERT_HASH" \
             --entitlements "$ENTITLEMENTS_PATH" \
             --options runtime \
             --timestamp \
             "$APP_PATH/main"
    
    if [ $? -eq 0 ]; then
        log "主程序签名成功"
        
        # 验证签名
        log "验证主程序签名..."
        codesign -vvv --deep --strict "$APP_PATH/main"
        if [ $? -eq 0 ]; then
            log "主程序签名验证通过"
        else
            log "错误: 主程序签名验证失败"
            exit 1
        fi
    else
        log "错误: 主程序签名失败"
        exit 1
    fi
}

# 签名内部文件
sign_internal_files() {
    log "开始签名 _internal 目录下的文件..."
    
    find "$APP_PATH/_internal" -type f -exec codesign --force \
         --sign "$CERT_HASH" \
         --options runtime \
         --timestamp {} \;
    
    if [ $? -eq 0 ]; then
        log "_internal 目录文件签名完成"
    else
        log "警告: 部分内部文件可能签名失败"
    fi
}

# 主函数
main() {
    # 检查必要文件
    if [ ! -f "$APP_PATH/main" ]; then
        log "错误: 主程序文件不存在"
        exit 1
    fi
    
    if [ ! -d "$APP_PATH/_internal" ]; then
        log "错误: _internal 目录不存在"
        exit 1
    fi
    
    if [ ! -f "$ENTITLEMENTS_PATH" ]; then
        log "错误: entitlements.plist 文件不存在"
        exit 1
    fi
    
    # 执行签名
    sign_python_files
    sign_main_executable
    sign_internal_files
    
    log "所有签名步骤完成，开始压缩"
    ditto -v -c -k --sequesterRsrc --keepParent main main.zip
    log "压缩完成"

}

# 执行主函数
main