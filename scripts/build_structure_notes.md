# 编译和签名流程产物与目录结构说明

## 项目结构示例

project_root/
├── app/                    # 应用源代码
│   ├── __init__.py         # 可能包含版本信息
│   ├── main.py             # 程序入口
│   └── ...
├── dist/                   # PyInstaller编译后产物目录
├── releases/               # 最终发布产品目录
├── scripts/                # 构建和签名脚本
│   ├── build_and_sign.sh   # 主构建脚本
│   ├── sign.sh             # 签名脚本
│   └── entitlements.plist  # macOS权限文件
├── main.spec               # PyInstaller规格文件
├── version.txt             # 可选的版本文件
└── .build_version          # 构建版本记录文件

## 编译流程与产物

### 编译前（源代码）

所有平台通用的主要源文件：
- Python源代码（在app/目录下）
- 资源文件（如images/, data/等）
- 配置文件（config.json, settings.yml等）

### 编译过程

build_and_sign.sh脚本会：
1. 检测平台（macOS/Linux/Windows）
2. 获取当前版本（从version.txt, __version__或git提交）
3. 判断是否需要重新编译
4. 调用PyInstaller执行编译
5. 根据平台执行签名（仅macOS）
6. 打包并移动到releases/目录

### 编译后的产物（按平台）

#### macOS平台

dist/
└── AppName/                # 应用Bundle
    ├── AppName             # 主可执行文件
    ├── Info.plist          # 应用信息
    └── _internal/          # PyInstaller内部文件
        ├── Python          # Python解释器
        ├── Python.framework/
        └── ...其他库和资源

最终产品：

releases/
└── AppName-1.0.0.dmg       # 首选格式，包含应用Bundle

或

releases/
└── AppName-1.0.0.zip       # 备用格式，包含应用Bundle

#### Linux平台

dist/
└── AppName/                # 应用目录
    ├── AppName             # 主可执行文件 (ELF格式)
    └── _internal/          # PyInstaller内部文件
        ├── Python          # Python解释器
        └── ...其他库和资源

最终产品：

releases/
└── AppName-1.0.0.tar.gz    # 压缩包，包含所有文件

#### Windows平台

dist/
└── AppName/                # 应用目录
    ├── AppName.exe         # 主可执行文件
    ├── python3X.dll        # Python DLL
    └── _internal/          # PyInstaller内部文件
        └── ...其他库和资源

最终产品：

releases/
└── AppName-1.0.0.zip       # 压缩包，包含所有文件

## 版本命名规则

最终产物按以下规则命名：
- 应用名称-版本号.扩展名
- 应用名称：从dist目录获取
- 版本号：从代码或版本文件获取
- 扩展名：根据平台和打包方式

例如：
- macOS: MyApp-1.2.3.dmg 或 MyApp-1.2.3.zip
- Linux: MyApp-1.2.3.tar.gz
- Windows: MyApp-1.2.3.zip

## 特别说明

1. 版本控制：
   - 相同版本的应用不会重新编译
   - 版本号变更会触发重新编译
   - 未显式指定版本时会使用git提交哈希或时间戳

2. 签名过程（仅macOS）：
   - 对所有可执行文件签名
   - 对Python解释器和框架签名
   - 验证签名有效性

3. 多版本并存：
   - releases/目录可同时保存多个版本
   - 按版本号区分不同版本的产品

## 常见问题与解决方案

1. 编译失败：
   - 检查依赖项是否齐全 (pip install -r requirements.txt)
   - 确认PyInstaller已安装 (pip install pyinstaller)
   - 查看详细日志 (build_sign_*.log)

2. 签名错误（macOS）：
   - 确认证书有效 (security find-identity -v)
   - 检查entitlements.plist文件是否正确
   - 可能需要更新证书或重新登录Apple开发者账户

3. 跳过编译但需要强制重新编译：
   - 删除.build_version文件
   - 或修改version.txt中的版本号
   - 或使用参数--force-rebuild(如已实现) 