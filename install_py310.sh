#!/bin/bash
# STTN Python 3.10 安装脚本
# 使用方法: bash install_py310.sh [cuda11|cuda12|cpu]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Python 版本
check_python_version() {
    print_info "检查 Python 版本..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 未安装"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    print_info "检测到 Python $PYTHON_VERSION"

    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
        print_info "✓ Python 版本符合要求 (>= 3.10)"
    else
        print_error "Python 版本不符合要求，需要 >= 3.10"
        print_info "请安装 Python 3.10 或更高版本"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    print_info "创建虚拟环境..."

    if [ -d "venv" ]; then
        print_warning "虚拟环境已存在"
        read -p "是否删除并重新创建? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            python3 -m venv venv
            print_info "✓ 虚拟环境已重新创建"
        else
            print_info "使用现有虚拟环境"
        fi
    else
        python3 -m venv venv
        print_info "✓ 虚拟环境已创建"
    fi
}

# 激活虚拟环境
activate_venv() {
    print_info "激活虚拟环境..."
    source venv/bin/activate
    print_info "✓ 虚拟环境已激活"
}

# 升级 pip
upgrade_pip() {
    print_info "升级 pip..."
    pip install --upgrade pip setuptools wheel
    print_info "✓ pip 已升级"
}

# 安装基础依赖
install_base_deps() {
    print_info "安装基础依赖..."
    pip install -e .
    print_info "✓ 基础依赖已安装"
}

# 安装 CUDA 相关依赖
install_cuda_deps() {
    local cuda_version=$1

    if [ "$cuda_version" == "cuda11" ]; then
        print_info "安装 CUDA 11.x 依赖..."
        pip install -e ".[cuda]"
        print_info "✓ CUDA 11.x 依赖已安装"
    elif [ "$cuda_version" == "cuda12" ]; then
        print_info "安装 CUDA 12.x 依赖..."
        pip install -e ".[cuda12]"
        print_info "✓ CUDA 12.x 依赖已安装"
    elif [ "$cuda_version" == "cpu" ]; then
        print_info "CPU 模式，跳过 CUDA 依赖"
    else
        print_warning "未指定 CUDA 版本，跳过 CUDA 依赖"
        print_info "如需 CUDA 支持，请手动运行："
        print_info "  pip install -e '.[cuda]'    # CUDA 11.x"
        print_info "  pip install -e '.[cuda12]'  # CUDA 12.x"
    fi
}

# 检查 CUDA
check_cuda() {
    print_info "检查 CUDA..."

    if command -v nvidia-smi &> /dev/null; then
        print_info "✓ NVIDIA 驱动已安装"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
            print_info "✓ CUDA Toolkit 已安装: $CUDA_VERSION"
        else
            print_warning "CUDA Toolkit 未安装或未在 PATH 中"
        fi
    else
        print_warning "NVIDIA 驱动未安装，将使用 CPU 模式"
    fi
}

# 运行测试
run_tests() {
    print_info "运行测试..."

    if [ -f "test_upgrade.py" ]; then
        python test_upgrade.py
    else
        print_warning "测试脚本未找到，跳过测试"
    fi
}

# 显示使用说明
show_usage() {
    cat << EOF
STTN Python 3.10 安装脚本

使用方法:
    bash install_py310.sh [选项]

选项:
    cuda11    安装 CUDA 11.x 支持
    cuda12    安装 CUDA 12.x 支持
    cpu       仅安装 CPU 版本
    (无)      安装基础依赖，不安装 CUDA 支持

示例:
    bash install_py310.sh cuda12    # 安装 CUDA 12.x 版本
    bash install_py310.sh cpu       # 安装 CPU 版本
    bash install_py310.sh           # 仅安装基础依赖

EOF
}

# 主函数
main() {
    echo "========================================"
    echo "STTN Python 3.10 安装脚本"
    echo "========================================"
    echo

    # 解析参数
    CUDA_MODE=${1:-""}

    if [ "$CUDA_MODE" == "--help" ] || [ "$CUDA_MODE" == "-h" ]; then
        show_usage
        exit 0
    fi

    # 执行安装步骤
    check_python_version
    check_cuda
    create_venv
    activate_venv
    upgrade_pip
    install_base_deps
    install_cuda_deps "$CUDA_MODE"

    echo
    echo "========================================"
    print_info "安装完成！"
    echo "========================================"
    echo

    # 显示下一步操作
    print_info "下一步操作："
    echo "  1. 激活虚拟环境:"
    echo "     source venv/bin/activate"
    echo
    echo "  2. 运行测试:"
    echo "     python test_upgrade.py"
    echo
    echo "  3. 开始训练:"
    echo "     python train.py -c configs/youtube-vos.json"
    echo

    # 询问是否运行测试
    read -p "是否现在运行测试? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        run_tests
    fi

    echo
    print_info "安装脚本执行完毕！"
}

# 运行主函数
main "$@"
