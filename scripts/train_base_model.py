#!/usr/bin/env python3
"""
基础模型训练脚本
适用于Mac Mini的轻量级训练
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

from src.data.dataset import IMDBDataset
from src.models.base_models import LightweightClassifier
from src.utils.training_utils import (
    evaluate_model, save_model, plot_training_curves, 
    EarlyStopping, LearningRateScheduler, print_model_info, get_device_info
)
from configs.base_config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, DEVICE_CONFIG


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # 统计
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser(description='训练基础模型')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'], help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=DATA_CONFIG['batch_size'], help='批次大小')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'], help='学习率')
    parser.add_argument('--max_samples', type=int, default=DATA_CONFIG['train_samples'], help='最大训练样本数')
    parser.add_argument('--model_name', type=str, default=DATA_CONFIG['model_name'], help='模型名称')
    parser.add_argument('--save_path', type=str, default='models/base_classifier.pth', help='模型保存路径')
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
    
    # 加载数据
    print("Loading dataset...")
    dataset_loader = IMDBDataset(
        model_name=args.model_name,
        max_length=DATA_CONFIG['max_length']
    )
    
    train_dataset = dataset_loader.load_data(
        split="train", 
        max_samples=args.max_samples
    )
    val_dataset = dataset_loader.load_data(
        split="test", 
        max_samples=DATA_CONFIG['val_samples']
    )
    
    train_loader = dataset_loader.get_dataloader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = dataset_loader.get_dataloader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 创建模型
    print("Creating model...")
    model = LightweightClassifier(
        model_name=args.model_name,
        num_classes=MODEL_CONFIG['num_classes'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    print_model_info(model)
    
    # 移动到设备
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=TRAINING_CONFIG['lr_scheduler_factor'],
        patience=TRAINING_CONFIG['lr_scheduler_patience'],
        verbose=True
    )
    
    # 早停
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG['early_stopping_patience']
    )
    
    # 训练记录
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        # 记录指标
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['val_f1'].append(val_metrics['f1'])
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # 早停检查
        if early_stopping(val_metrics['loss'], model):
            print("Early stopping triggered!")
            break
        
        # 保存最佳模型
        if epoch == 0 or val_metrics['f1'] > max(metrics['val_f1'][:-1]):
            save_model(
                model, 
                args.save_path,
                config={
                    'model_name': args.model_name,
                    'epoch': epoch,
                    'val_f1': val_metrics['f1'],
                    'val_acc': val_metrics['accuracy']
                }
            )
            print(f"Model saved with F1: {val_metrics['f1']:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(metrics, save_path=f"{PATH_CONFIG['log_dir']}/training_curves.png")
    
    # 保存训练记录
    with open(f"{PATH_CONFIG['log_dir']}/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Training completed!")
    print(f"Best F1: {max(metrics['val_f1']):.4f}")
    print(f"Best Acc: {max(metrics['val_acc']):.4f}")


if __name__ == "__main__":
    main()
