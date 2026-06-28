#!/usr/bin/env bash
set -euo pipefail

echo "开始准备 GCP Python 环境..."

sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3-pip

python3.9 -m venv .venv

./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "环境准备完成"
