#!/bin/bash
# D4 Target 候选组合扫描 - 快速启动脚本

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "D4 Target 候选组合扫描"
echo "========================================================================"
echo ""
echo "项目路径: $PROJECT_ROOT"
echo "脚本路径: $SCRIPT_DIR/scan_d4_target_candidates.py"
echo ""
echo "开始扫描..."
echo ""

# 运行扫描脚本
python scripts/scan_d4_target_candidates.py

echo ""
echo "========================================================================"
echo "扫描完成！"
echo "========================================================================"
echo ""
echo "输出文件位置: outputs/dataset_audit/"
echo ""
echo "查看报告："
echo "  cat outputs/dataset_audit/d4_target_candidates_summary.md"
echo ""
echo "或在编辑器中打开："
echo "  open outputs/dataset_audit/d4_target_candidates_summary.md"
echo ""
