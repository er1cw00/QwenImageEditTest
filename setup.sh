


export HF_ENDPOINT=https://hf-mirror.com


# --- 安装依赖包 ---
install_whl_from_url() {
    local url=$1
    local filename=$2

    echo "-------------------------------------------------------"
    echo "🚀 准备安装: $filename"
    
    # 1. 使用 curl 下载
    # -f: HTTP 错误时报错退出  -L: 跟随重定向  --retry: 失败重试
    # -C -: 支持断点续传（如果文件下载了一半可以接着下） -x http://192.168.3.100:7890 
    echo "📥 正在下载..."
    if curl -f -L --retry 3 --retry-delay 5 --connect-timeout 30 -C - "$url" -o "$filename"; then
        echo "✅ 下载成功。"
    else
        echo "❌ 错误: 下载失败，请检查网络或 URL 是否正确。"
        return 1
    fi

    # 2. 使用 uv 安装
    echo "📦 正在使用 uv 安装..."
    if uv pip install "$filename"; then
        echo "✅ 安装成功。"
    else
        echo "❌ 错误: pip 安装失败。"
        # 安装失败也建议删除损坏的文件，防止下次运行断点续传出错
        rm -f "$filename"
        return 1
    fi

    # 3. 删除清理
    echo "🧹 正在清理安装包..."
    rm -f "$filename"
    echo "✨ 完成！"
    echo "-------------------------------------------------------"
}

pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install triton==3.4.0

echo "📥 安装 flash_attn v2.8.3"
install_whl_from_url "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"  "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
echo "📥 安装 nunchaku v1.0.2"
install_whl_from_url "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.0.2/nunchaku-1.0.2+torch2.8-cp312-cp312-linux_x86_64.whl" "nunchaku-1.0.2+torch2.8-cp312-cp312-linux_x86_64.whl"  
echo "📥 安装 gptqmodel v5.6.12"
install_whl_from_url "https://github.com/ModelCloud/GPTQModel/releases/download/v5.6.12/gptqmodel-5.6.12+cu128torch2.8-cp312-cp312-linux_x86_64.whl" "gptqmodel-5.6.12+cu128torch2.8-cp312-cp312-linux_x86_64.whl"

pip install -r requirements.txt --index-url "https://mirrors.aliyun.com/pypi/simple"
pip install torchao==0.13.0 --index-url "https://mirrors.aliyun.com/pypi/simple/"