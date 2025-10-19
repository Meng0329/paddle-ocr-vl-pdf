#!/bin/bash
# 配置检查脚本

echo "==================================================="
echo "       PaddleOCR-VL 配置检查工具"
echo "==================================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 extract_pdf.py 是否存在
if [ ! -f "extract_pdf.py" ]; then
    echo -e "${RED}✗ 错误：找不到 extract_pdf.py 文件${NC}"
    echo "  请在项目根目录运行此脚本"
    exit 1
fi

# 1. 检查 PDF 文件
echo -e "\n${YELLOW}[1/5] 检查 PDF 文件路径${NC}"
PDF_PATH=$(grep "pdf_path = " extract_pdf.py | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "   路径: $PDF_PATH"

if [ -f "$PDF_PATH" ]; then
    FILE_SIZE=$(du -h "$PDF_PATH" | cut -f1)
    echo -e "   ${GREEN}✓ 文件存在 (大小: $FILE_SIZE)${NC}"
else
    echo -e "   ${RED}✗ 文件不存在！${NC}"
    echo -e "   ${YELLOW}请修改 extract_pdf.py 第 178 行的 pdf_path${NC}"
    exit 1
fi

# 2. 检查输出目录
echo -e "\n${YELLOW}[2/5] 检查输出目录${NC}"
OUTPUT_DIR=$(grep "output_dir = " extract_pdf.py | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "   路径: $OUTPUT_DIR"

if [ -d "$OUTPUT_DIR" ]; then
    echo -e "   ${GREEN}✓ 目录存在${NC}"
    # 检查是否有写权限
    if [ -w "$OUTPUT_DIR" ]; then
        echo -e "   ${GREEN}✓ 有写入权限${NC}"
    else
        echo -e "   ${RED}✗ 没有写入权限${NC}"
        exit 1
    fi
else
    echo -e "   ${YELLOW}! 目录不存在，将自动创建${NC}"
    mkdir -p "$OUTPUT_DIR" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "   ${GREEN}✓ 目录创建成功${NC}"
    else
        echo -e "   ${RED}✗ 无法创建目录${NC}"
        exit 1
    fi
fi

# 3. 检查 VLM 模型
echo -e "\n${YELLOW}[3/5] 检查 VLM 模型${NC}"
VL_MODEL_DIR="./PaddleOCR-VL/PaddleOCR-VL-0.9B"
VL_MODEL_FILE="$VL_MODEL_DIR/model.safetensors"

if [ -d "$VL_MODEL_DIR" ]; then
    echo -e "   ${GREEN}✓ 模型目录存在${NC}"
    if [ -f "$VL_MODEL_FILE" ]; then
        FILE_SIZE=$(du -h "$VL_MODEL_FILE" | cut -f1)
        echo -e "   ${GREEN}✓ 模型文件存在 (大小: $FILE_SIZE)${NC}"
    else
        echo -e "   ${RED}✗ 模型文件 model.safetensors 不存在${NC}"
        echo -e "   ${YELLOW}请下载模型：https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/files${NC}"
        exit 1
    fi
else
    echo -e "   ${RED}✗ 模型目录不存在${NC}"
    echo -e "   ${YELLOW}请下载模型到: $VL_MODEL_DIR${NC}"
    exit 1
fi

# 4. 检查 Layout 模型
echo -e "\n${YELLOW}[4/5] 检查 Layout 模型${NC}"
LAYOUT_MODEL_DIR="./PaddleOCR-VL/PP-DocLayoutV2"
LAYOUT_MODEL_FILE="$LAYOUT_MODEL_DIR/inference.pdmodel"

if [ -d "$LAYOUT_MODEL_DIR" ]; then
    echo -e "   ${GREEN}✓ 模型目录存在${NC}"
    if [ -f "$LAYOUT_MODEL_FILE" ]; then
        FILE_SIZE=$(du -h "$LAYOUT_MODEL_FILE" | cut -f1)
        echo -e "   ${GREEN}✓ 模型文件存在 (大小: $FILE_SIZE)${NC}"
    else
        echo -e "   ${RED}✗ 模型文件 inference.pdmodel 不存在${NC}"
        echo -e "   ${YELLOW}请下载模型：https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/files${NC}"
        exit 1
    fi
else
    echo -e "   ${RED}✗ 模型目录不存在${NC}"
    echo -e "   ${YELLOW}请下载模型到: $LAYOUT_MODEL_DIR${NC}"
    exit 1
fi

# 5. 检查 GPU
echo -e "\n${YELLOW}[5/5] 检查 GPU${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo -e "   ${GREEN}✓ 检测到 $GPU_COUNT 块 GPU${NC}"
    
    # 显示 GPU 信息
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
        echo "     - $line"
    done
else
    echo -e "   ${RED}✗ 未检测到 NVIDIA GPU 或 nvidia-smi 未安装${NC}"
    exit 1
fi

# 总结
echo -e "\n${GREEN}==================================================="
echo -e "           ✓ 所有配置检查通过！"
echo -e "===================================================${NC}"
echo -e "\n可以运行以下命令开始处理："
echo -e "${YELLOW}  source .venv/bin/activate${NC}"
echo -e "${YELLOW}  nohup python extract_pdf.py > extract_run.log 2>&1 &${NC}"
echo -e "\n监控进度："
echo -e "${YELLOW}  ./monitor.sh${NC}"
echo ""

