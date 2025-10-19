#!/bin/bash
# 实时查看 PDF 提取日志

LOG_FILE="/home/test/wh/graphrag-main/graphrag-main/ocr2/output/extract.log"
PROGRESS_FILE="/home/test/wh/graphrag-main/graphrag-main/ocr2/output/progress.json"

echo "========================================"
echo "  PDF 提取进度监控"
echo "========================================"
echo ""

# 检查日志文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "日志文件不存在，请先运行 extract_pdf.py"
    exit 1
fi

# 显示当前进度
if [ -f "$PROGRESS_FILE" ]; then
    echo "--- 当前进度 ---"
    cat "$PROGRESS_FILE" | grep -E "(total_pages|processed_pages|last_update|elapsed_time)" | head -4
    echo ""
fi

echo "--- 实时日志（按 Ctrl+C 退出）---"
echo ""

# 实时跟踪日志
tail -f "$LOG_FILE"

