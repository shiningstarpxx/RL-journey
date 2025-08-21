#!/bin/bash

echo "🚀 RLHF on Mac Mini 快速启动"
echo "================================"

# 检查Python环境
echo "📋 检查Python环境..."
python3 --version

# 安装依赖
echo "📦 安装依赖包..."
pip3 install -r requirements.txt

# 创建必要的目录
echo "📁 创建目录结构..."
mkdir -p data models logs checkpoints

# 运行演示
echo "🎯 开始运行RLHF演示..."
echo "================================"

python3 scripts/demo.py

echo ""
echo "✅ 演示完成！"
echo "📊 查看logs目录中的训练记录"
echo "�� 查看models目录中的保存模型"
