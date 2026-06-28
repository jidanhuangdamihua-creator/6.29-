#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if command -v python3 >/dev/null 2>&1; then
  python3 scripts/launcher.py
else
  echo "未找到 python3，请先安装 Python 3.9+。"
fi

echo
read -n 1 -s -r -p "按任意键关闭窗口..."
echo
