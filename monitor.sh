#!/bin/bash
# 实时监控 PDF 提取进度

OUTPUT_DIR="/home/test/wh/graphrag-main/graphrag-main/ocr2/output"
LOG_FILE="$OUTPUT_DIR/extract.log"
PROGRESS_FILE="$OUTPUT_DIR/progress.json"

clear

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "           PDF 提取进度实时监控"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    # GPU 状态
    echo "【GPU 状态】"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s\n  显存: %s MB / %s MB (%.1f%%)\n  使用率: %s%%  温度: %s°C\n\n", $1, $2, $3, $4, ($3/$4)*100, $5, $6}'
    
    echo "────────────────────────────────────────────────────────────"
    
    # 进程状态
    echo "【进程状态】"
    PROCESS_INFO=$(ps aux | grep "python.*extract_pdf.py" | grep -v grep)
    if [ -n "$PROCESS_INFO" ]; then
        PID=$(echo "$PROCESS_INFO" | awk '{print $2}')
        CPU=$(echo "$PROCESS_INFO" | awk '{print $3}')
        MEM=$(echo "$PROCESS_INFO" | awk '{print $4}')
        TIME=$(echo "$PROCESS_INFO" | awk '{print $10}')
        echo "  PID: $PID"
        echo "  CPU: ${CPU}%  |  内存: ${MEM}%  |  运行时间: $TIME"
    else
        echo "  进程未运行"
    fi
    echo ""
    
    echo "────────────────────────────────────────────────────────────"
    
    # 处理进度
    if [ -f "$PROGRESS_FILE" ]; then
        echo "【处理进度】"
        TOTAL=$(grep -o '"total_pages": [0-9]*' "$PROGRESS_FILE" | awk '{print $2}')
        PROCESSED=$(grep -o '"processed_pages": [0-9]*' "$PROGRESS_FILE" | awk '{print $2}')
        LAST_UPDATE=$(grep -o '"last_update": "[^"]*"' "$PROGRESS_FILE" | cut -d'"' -f4)
        ELAPSED=$(grep -o '"elapsed_time_seconds": [0-9.]*' "$PROGRESS_FILE" | awk '{print $2}')
        
        if [ -n "$TOTAL" ] && [ -n "$PROCESSED" ]; then
            PERCENT=$(echo "scale=2; $PROCESSED * 100 / $TOTAL" | bc)
            ELAPSED_MIN=$(echo "scale=1; $ELAPSED / 60" | bc)
            
            if [ "$PROCESSED" -gt 0 ]; then
                SPEED=$(echo "scale=2; $PROCESSED / $ELAPSED" | bc)
                ETA_SEC=$(echo "scale=0; ($TOTAL - $PROCESSED) / $SPEED" | bc)
                ETA_MIN=$(echo "scale=1; $ETA_SEC / 60" | bc)
            else
                SPEED="0.00"
                ETA_MIN="--"
            fi
            
            echo "  已处理: $PROCESSED / $TOTAL 页 (${PERCENT}%)"
            echo "  速度: ${SPEED} 页/秒"
            echo "  已用时间: ${ELAPSED_MIN} 分钟"
            echo "  预计剩余: ${ETA_MIN} 分钟"
            echo "  最后更新: $LAST_UPDATE"
        fi
    else
        echo "【处理进度】"
        echo "  初始化中..."
    fi
    echo ""
    
    echo "────────────────────────────────────────────────────────────"
    
    # 最新日志
    echo "【最新日志】（最后 5 行）"
    if [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE" | sed 's/^/  /'
    else
        echo "  日志文件未生成"
    fi
    echo ""
    
    echo "────────────────────────────────────────────────────────────"
    echo "更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "按 Ctrl+C 退出监控"
    
    sleep 5
done

