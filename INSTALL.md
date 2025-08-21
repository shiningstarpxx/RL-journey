# RLHF on Mac Mini 安装指南

## 系统要求

- **操作系统**: macOS 10.15 或更高版本
- **硬件**: Mac Mini (任何型号)
- **内存**: 至少8GB RAM (推荐16GB)
- **存储**: 至少10GB可用空间
- **Python**: 3.8 或更高版本

## 快速安装

### 1. 克隆项目
```bash
git clone <repository-url>
cd RL-journey
```

### 2. 创建虚拟环境 (推荐)
```bash
# 使用conda
conda create -n rlhf python=3.9
conda activate rlhf

# 或使用venv
python3 -m venv rlhf_env
source rlhf_env/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 运行演示
```bash
# 快速启动
./run_demo.sh

# 或手动运行
python scripts/demo.py
```

## 详细安装步骤

### 步骤1: 环境准备

1. **检查Python版本**
   ```bash
   python3 --version
   # 确保版本 >= 3.8
   ```

2. **检查PyTorch安装**
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
   ```

### 步骤2: 安装PyTorch

根据你的Mac型号选择合适的PyTorch版本：

**Apple Silicon (M1/M2) Mac Mini:**
```bash
pip3 install torch torchvision torchaudio
```

**Intel Mac Mini:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 步骤3: 安装其他依赖

```bash
pip install transformers datasets scikit-learn matplotlib tqdm wandb accelerate peft trl
```

### 步骤4: 验证安装

```bash
python3 -c "
import torch
import transformers
import datasets
print('✅ 所有依赖安装成功!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
"
```

## 项目结构

```
RL-journey/
├── data/                   # 数据集目录
├── models/                 # 模型保存目录
├── src/                    # 源代码
│   ├── data/              # 数据处理
│   ├── models/            # 模型定义
│   ├── rl/                # RL算法实现
│   └── utils/             # 工具函数
├── notebooks/             # Jupyter notebooks
├── configs/               # 配置文件
├── scripts/               # 训练脚本
├── logs/                  # 训练日志
├── checkpoints/           # 检查点
├── requirements.txt       # 依赖列表
├── run_demo.sh           # 快速启动脚本
└── README.md             # 项目说明
```

## 使用方法

### 1. 基础模型训练

```bash
python scripts/train_base_model.py \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_samples 1000
```

### 2. RLHF训练

```bash
python scripts/train_rlhf.py \
    --base_model_path models/base_classifier.pth \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-5
```

### 3. RNN生成模型训练

```bash
python scripts/train_rnn_generator.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_samples 1000
```

### 4. 完整演示

```bash
python scripts/demo.py
```

## 配置说明

### 数据配置 (configs/base_config.py)

```python
DATA_CONFIG = {
    'dataset_name': 'imdb',           # 数据集名称
    'model_name': 'distilbert-base-uncased',  # 预训练模型
    'max_length': 128,                # 最大序列长度
    'batch_size': 16,                 # 批次大小
    'train_samples': 1000,            # 训练样本数量
    'val_samples': 200,               # 验证样本数量
}
```

### 模型配置

```python
MODEL_CONFIG = {
    'num_classes': 2,                 # 分类数量
    'dropout': 0.1,                   # Dropout率
    'freeze_layers': 6,               # 冻结层数
    'vocab_size': 10000,              # 词汇表大小
    'embedding_dim': 128,             # 嵌入维度
    'hidden_dim': 256,                # 隐藏层维度
}
```

### 训练配置

```python
TRAINING_CONFIG = {
    'epochs': 10,                     # 训练轮数
    'learning_rate': 2e-5,            # 学习率
    'weight_decay': 0.01,             # 权重衰减
    'gradient_clip': 1.0,             # 梯度裁剪
    'early_stopping_patience': 5,     # 早停耐心值
}
```

## 性能优化建议

### 1. 内存优化

- 减少批次大小: `--batch_size 8`
- 减少序列长度: `--max_length 64`
- 减少训练样本: `--max_samples 500`

### 2. 计算优化

- 使用单进程: `num_workers=0`
- 关闭混合精度: `mixed_precision=False`
- 冻结部分层: `freeze_layers=6`

### 3. 存储优化

- 定期清理日志: `rm -rf logs/*`
- 只保存最佳模型
- 使用压缩存储

## 故障排除

### 常见问题

1. **内存不足**
   ```
   错误: CUDA out of memory
   解决: 减少batch_size或max_samples
   ```

2. **依赖冲突**
   ```
   错误: ImportError
   解决: 重新创建虚拟环境
   ```

3. **下载失败**
   ```
   错误: Connection timeout
   解决: 使用国内镜像源
   ```

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=.
python -u scripts/demo.py 2>&1 | tee debug.log
```

### 性能监控

```bash
# 监控内存使用
top -pid $(pgrep -f "python.*demo.py")

# 监控GPU使用 (如果有)
nvidia-smi
```

## 扩展功能

### 1. 自定义数据集

```python
# 在src/data/dataset.py中添加
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # 实现自定义数据集
        pass
```

### 2. 自定义模型

```python
# 在src/models/base_models.py中添加
class CustomModel(nn.Module):
    def __init__(self, config):
        # 实现自定义模型
        pass
```

### 3. 自定义奖励函数

```python
# 在src/utils/training_utils.py中修改
def custom_reward_function(text):
    # 实现自定义奖励函数
    return reward_score
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系维护者。
