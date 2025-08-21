#!/usr/bin/env python3
"""
RLHF训练脚本
适用于Mac Mini的轻量级RLHF训练
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

from src.data.dataset import IMDBDataset
from src.models.base_models import LightweightClassifier, PolicyNetwork, RewardModel
from src.rl.ppo import PPOTrainer, RLHFTrainer, ExperienceBuffer
from src.utils.training_utils import (
    save_model, load_model, print_model_info, get_device_info,
    create_synthetic_rewards, compute_reward
)
from configs.base_config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, RL_CONFIG, RLHF_CONFIG, PATH_CONFIG, DEVICE_CONFIG


def collect_experience(policy_model, dataloader, device, num_episodes=100):
    """收集经验数据"""
    experiences = []
    policy_model.eval()
    
    print("Collecting experience...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting")):
            if i >= num_episodes:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 获取策略输出
            policy_logits, values, _ = policy_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 采样动作
            probs = torch.softmax(policy_logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze()
            
            # 计算奖励（使用合成奖励函数）
            texts = [f"Sample text {j}" for j in range(len(actions))]  # 这里应该使用真实的文本
            rewards = create_synthetic_rewards(texts)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # 存储经验
            for j in range(len(actions)):
                experience = {
                    'state': input_ids[j],
                    'action': actions[j],
                    'reward': rewards[j],
                    'value': values[j],
                    'log_prob': torch.log(probs[j, actions[j]]),
                    'done': True  # 简化处理
                }
                experiences.append(experience)
    
    return experiences


def train_reward_model(reward_model, dataloader, device, epochs=5):
    """训练奖励模型"""
    print("Training reward model...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(reward_model.parameters(), lr=1e-4)
    
    reward_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Reward Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 生成合成奖励
            batch_size = input_ids.size(0)
            synthetic_rewards = torch.randn(batch_size, 1, device=device)  # 随机奖励用于演示
            
            # 前向传播
            predicted_rewards = reward_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(predicted_rewards, synthetic_rewards)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Reward Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


def main():
    parser = argparse.ArgumentParser(description='训练RLHF模型')
    parser.add_argument('--base_model_path', type=str, default='models/base_classifier.pth', help='基础模型路径')
    parser.add_argument('--epochs', type=int, default=5, help='RLHF训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--beta', type=float, default=RLHF_CONFIG['beta'], help='KL散度惩罚系数')
    parser.add_argument('--save_path', type=str, default='models/rlhf_model.pth', help='模型保存路径')
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
        model_name=DATA_CONFIG['model_name'],
        max_length=DATA_CONFIG['max_length']
    )
    
    train_dataset = dataset_loader.load_data(
        split="train", 
        max_samples=500  # 减少样本数量
    )
    
    train_loader = dataset_loader.get_dataloader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # 创建基础模型
    print("Creating base model...")
    base_model = LightweightClassifier(
        model_name=DATA_CONFIG['model_name'],
        num_classes=MODEL_CONFIG['num_classes'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    # 加载预训练权重
    if os.path.exists(args.base_model_path):
        print(f"Loading base model from {args.base_model_path}")
        base_model = load_model(base_model, args.base_model_path, device)
    else:
        print("No pre-trained base model found. Using random initialization.")
    
    base_model = base_model.to(device)
    
    # 创建策略网络
    print("Creating policy network...")
    policy_model = PolicyNetwork(base_model, action_dim=2)
    policy_model = policy_model.to(device)
    
    # 创建奖励模型
    print("Creating reward model...")
    reward_model = RewardModel(base_model)
    reward_model = reward_model.to(device)
    
    # 创建参考模型（复制基础模型）
    print("Creating reference model...")
    reference_model = LightweightClassifier(
        model_name=DATA_CONFIG['model_name'],
        num_classes=MODEL_CONFIG['num_classes'],
        dropout=MODEL_CONFIG['dropout']
    )
    reference_model.load_state_dict(base_model.state_dict())
    reference_model = reference_model.to(device)
    
    print_model_info(policy_model)
    
    # 训练奖励模型
    train_reward_model(reward_model, train_loader, device)
    
    # 创建RLHF训练器
    rlhf_trainer = RLHFTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        reference_model=reference_model,
        lr=args.lr,
        beta=args.beta,
        device=device
    )
    
    # 训练记录
    metrics = {
        'loss': [],
        'kl_penalty': [],
        'reward': []
    }
    
    print("Starting RLHF training...")
    
    for epoch in range(args.epochs):
        print(f"\nRLHF Epoch {epoch+1}/{args.epochs}")
        
        epoch_loss = 0
        epoch_kl = 0
        epoch_reward = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"RLHF Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 生成合成奖励
            batch_size = input_ids.size(0)
            synthetic_rewards = torch.randn(batch_size, device=device)  # 随机奖励用于演示
            
            # 生成动作（简化处理）
            actions = torch.randint(0, 2, (batch_size,), device=device)
            
            # RLHF训练步骤
            step_metrics = rlhf_trainer.train_step(
                states=input_ids,
                actions=actions,
                rewards=synthetic_rewards
            )
            
            epoch_loss += step_metrics['loss']
            epoch_kl += step_metrics['kl_penalty']
            epoch_reward += step_metrics['reward']
            num_batches += 1
        
        # 计算平均指标
        avg_loss = epoch_loss / num_batches
        avg_kl = epoch_kl / num_batches
        avg_reward = epoch_reward / num_batches
        
        # 记录指标
        metrics['loss'].append(avg_loss)
        metrics['kl_penalty'].append(avg_kl)
        metrics['reward'].append(avg_reward)
        
        print(f"Loss: {avg_loss:.4f}, KL Penalty: {avg_kl:.4f}, Reward: {avg_reward:.4f}")
        
        # 保存模型
        if epoch == 0 or avg_reward > max(metrics['reward'][:-1]):
            save_model(
                policy_model,
                args.save_path,
                config={
                    'epoch': epoch,
                    'loss': avg_loss,
                    'kl_penalty': avg_kl,
                    'reward': avg_reward
                }
            )
            print(f"Model saved with reward: {avg_reward:.4f}")
    
    # 保存训练记录
    with open(f"{PATH_CONFIG['log_dir']}/rlhf_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("RLHF training completed!")
    print(f"Final reward: {metrics['reward'][-1]:.4f}")


if __name__ == "__main__":
    main()
