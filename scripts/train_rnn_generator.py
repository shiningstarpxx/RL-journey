#!/usr/bin/env python3
"""
RNN生成模型训练脚本
适用于Mac Mini的轻量级文本生成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
import random

from src.data.dataset import SimpleRNNDataset
from src.models.base_models import SimpleRNN
from src.utils.training_utils import (
    save_model, print_model_info, get_device_info
)
from configs.base_config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, DEVICE_CONFIG


def create_simple_text_data(num_samples=1000, max_length=20):
    """创建简单的文本数据用于演示"""
    texts = []
    
    # 简单的句子模板
    templates = [
        "I love this movie because it is {}",
        "This film is {} and I enjoyed it",
        "The story is {} and the acting is {}",
        "I think this is a {} movie",
        "The director did a {} job with this {} film"
    ]
    
    adjectives = [
        "amazing", "wonderful", "excellent", "great", "good",
        "bad", "terrible", "awful", "horrible", "disappointing",
        "interesting", "boring", "exciting", "funny", "sad"
    ]
    
    for _ in range(num_samples):
        template = random.choice(templates)
        if template.count("{}") == 1:
            text = template.format(random.choice(adjectives))
        elif template.count("{}") == 2:
            text = template.format(random.choice(adjectives), random.choice(adjectives))
        else:
            text = template
        
        # 限制长度
        words = text.split()[:max_length]
        texts.append(" ".join(words))
    
    return texts


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output, _ = model(input_ids)
        
        # 计算损失
        loss = criterion(output.view(-1, model.vocab_size), target_ids.view(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def generate_text_sample(model, dataset, device, prompt="I love", max_length=20):
    """生成文本样本"""
    model.eval()
    
    # 编码提示
    prompt_tokens = prompt.lower().split()
    input_indices = []
    
    for token in prompt_tokens:
        if token in dataset.token_to_idx:
            input_indices.append(dataset.token_to_idx[token])
        else:
            input_indices.append(dataset.token_to_idx['<UNK>'])
    
    if not input_indices:
        input_indices = [dataset.token_to_idx['<START>']]
    
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
    
    # 生成文本
    generated_tokens = input_indices.copy()
    
    with torch.no_grad():
        hidden = model.init_hidden(1, device)
        
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            
            # 获取下一个token的概率
            next_token_logits = output[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # 采样下一个token
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
            
            if next_token == dataset.token_to_idx['<END>']:
                break
            
            generated_tokens.append(next_token)
            input_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
    
    # 解码生成的文本
    generated_text = []
    for token_idx in generated_tokens:
        if token_idx in dataset.idx_to_token:
            token = dataset.idx_to_token[token_idx]
            if token not in ['<START>', '<PAD>']:
                generated_text.append(token)
    
    return " ".join(generated_text)


def main():
    parser = argparse.ArgumentParser(description='训练RNN生成模型')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--num_samples', type=int, default=1000, help='训练样本数量')
    parser.add_argument('--max_length', type=int, default=20, help='最大序列长度')
    parser.add_argument('--save_path', type=str, default='models/rnn_generator.pth', help='模型保存路径')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = get_device_info()
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # 创建目录
    os.makedirs(PATH_CONFIG['model_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['log_dir'], exist_ok=True)
    
    # 创建训练数据
    print("Creating training data...")
    train_texts = create_simple_text_data(args.num_samples, args.max_length)
    
    # 创建数据集
    print("Creating dataset...")
    dataset = SimpleRNNDataset(
        texts=train_texts,
        tokenizer=None,  # 不使用tokenizer，使用简单的词汇表
        max_length=args.max_length
    )
    
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Training samples: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 创建模型
    print("Creating RNN model...")
    model = SimpleRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    print_model_info(model)
    
    # 移动到设备
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.token_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # 训练记录
    metrics = {
        'train_loss': [],
        'lr': []
    }
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        
        # 学习率调度
        scheduler.step(train_loss)
        
        # 记录指标
        metrics['train_loss'].append(train_loss)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # 生成样本
        if epoch % 5 == 0:
            print("\nGenerating sample texts:")
            for prompt in ["I love", "This movie", "The film"]:
                generated = generate_text_sample(model, dataset, device, prompt)
                print(f"'{prompt}' -> '{generated}'")
        
        # 保存模型
        if epoch == 0 or train_loss < min(metrics['train_loss'][:-1]):
            save_model(
                model,
                args.save_path,
                config={
                    'epoch': epoch,
                    'loss': train_loss,
                    'vocab_size': dataset.vocab_size,
                    'vocab': dataset.vocab
                }
            )
            print(f"Model saved with loss: {train_loss:.4f}")
    
    # 保存训练记录
    with open(f"{PATH_CONFIG['log_dir']}/rnn_training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Training completed!")
    print(f"Final loss: {metrics['train_loss'][-1]:.4f}")
    
    # 最终生成示例
    print("\nFinal generated examples:")
    for prompt in ["I love", "This movie", "The film", "It was", "Great"]:
        generated = generate_text_sample(model, dataset, device, prompt)
        print(f"'{prompt}' -> '{generated}'")


if __name__ == "__main__":
    main()
