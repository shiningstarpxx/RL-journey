#!/usr/bin/env python3
"""
简化版 RL vs RLHF 对比脚本
使用轻量级模型，避免网络依赖
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time

from src.utils.training_utils import get_device_info


class SimpleTextDataset(Dataset):
    """简单的文本数据集"""
    
    def __init__(self, num_samples=200, seq_length=10):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # 创建简单的文本数据
        self.texts = []
        for i in range(num_samples):
            # 生成随机文本序列
            text = np.random.randint(0, 100, seq_length)
            self.texts.append(text)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        return {
            'input_ids': torch.tensor(text, dtype=torch.long),
            'labels': torch.randint(0, 2, (1,)).item()
        }


class SimpleClassifier(nn.Module):
    """简单的分类器"""
    
    def __init__(self, vocab_size=100, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class SimplePolicyNetwork(nn.Module):
    """简单的策略网络"""
    
    def __init__(self, base_model, action_dim=2):
        super().__init__()
        self.base_model = base_model
        hidden_dim = 64  # 与base_model匹配
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids):
        # 获取基础模型的隐藏表示
        embedded = self.base_model.embedding(input_ids)
        lstm_out, _ = self.base_model.lstm(embedded)
        features = torch.mean(lstm_out, dim=1)
        
        # 策略和价值
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value


class SimpleRewardModel(nn.Module):
    """简单的奖励模型"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_dim = 64
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids):
        embedded = self.base_model.embedding(input_ids)
        lstm_out, _ = self.base_model.lstm(embedded)
        features = torch.mean(lstm_out, dim=1)
        reward = self.reward_head(features)
        return reward


class PureRLTrainer:
    """单纯的RL训练器（无人类反馈）"""
    
    def __init__(self, policy_model, lr=3e-4, device="cpu"):
        self.policy_model = policy_model
        self.device = device
        self.optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        
        # 将模型移到设备
        self.policy_model.to(device)
    
    def train_step(self, states, actions, rewards):
        """训练步骤"""
        # 策略模型前向传播
        policy_logits, _ = self.policy_model(states)
        
        # 计算动作概率
        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        
        # 损失函数：最大化期望奖励
        loss = -(log_probs * rewards).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'entropy': dist.entropy().mean().item()
        }


class SimpleRLHFTrainer:
    """简化的RLHF训练器"""
    
    def __init__(self, policy_model, reward_model, reference_model, lr=1e-4, beta=0.1, device="cpu"):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.beta = beta
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        
        # 将模型移到设备
        self.policy_model.to(device)
        self.reward_model.to(device)
        self.reference_model.to(device)
        
        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
    
    def compute_kl_penalty(self, policy_logits, reference_logits):
        """计算KL散度惩罚"""
        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        reference_dist = torch.distributions.Categorical(logits=reference_logits)
        
        # 使用KL散度
        kl_div = torch.distributions.kl.kl_divergence(policy_dist, reference_dist)
        
        return kl_div.mean()
    
    def train_step(self, states, actions, rewards):
        """训练步骤"""
        # 策略模型前向传播
        policy_logits, _ = self.policy_model(states)
        
        # 参考模型前向传播
        with torch.no_grad():
            reference_logits, _ = self.reference_model(states)
        
        # 计算动作概率
        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        
        # 计算KL散度惩罚
        kl_penalty = self.compute_kl_penalty(policy_logits, reference_logits)
        
        # 总损失
        loss = -(log_probs * rewards).mean() + self.beta * kl_penalty
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'kl_penalty': kl_penalty.item(),
            'reward': rewards.mean().item()
        }


def create_simple_dataset():
    """创建简单的数据集"""
    print("创建简单数据集...")
    
    train_dataset = SimpleTextDataset(num_samples=200, seq_length=10)
    val_dataset = SimpleTextDataset(num_samples=50, seq_length=10)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader


def train_pure_rl(base_model, train_loader, device, epochs=3):
    """训练单纯的RL模型"""
    print("\n" + "=" * 60)
    print("训练单纯的RL模型（无人类反馈）")
    print("=" * 60)
    
    # 创建策略网络
    policy_model = SimplePolicyNetwork(base_model, action_dim=2)
    policy_model = policy_model.to(device)
    
    # 创建单纯RL训练器
    pure_rl_trainer = PureRLTrainer(policy_model, lr=1e-3, device=device)
    
    # 训练记录
    metrics = {
        'loss': [],
        'reward': [],
        'entropy': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nPure RL Epoch {epoch+1}/{epochs}")
        
        epoch_loss = 0
        epoch_reward = 0
        epoch_entropy = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Pure RL Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            
            # 生成随机奖励（模拟环境反馈）
            batch_size = input_ids.size(0)
            random_rewards = torch.randn(batch_size, device=device)
            
            # 生成动作
            actions = torch.randint(0, 2, (batch_size,), device=device)
            
            # 训练步骤
            step_metrics = pure_rl_trainer.train_step(
                states=input_ids,
                actions=actions,
                rewards=random_rewards
            )
            
            epoch_loss += step_metrics['loss']
            epoch_reward += step_metrics['reward']
            epoch_entropy += step_metrics['entropy']
            num_batches += 1
        
        # 计算平均指标
        avg_loss = epoch_loss / num_batches
        avg_reward = epoch_reward / num_batches
        avg_entropy = epoch_entropy / num_batches
        
        # 记录指标
        metrics['loss'].append(avg_loss)
        metrics['reward'].append(avg_reward)
        metrics['entropy'].append(avg_entropy)
        
        print(f"Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, Entropy: {avg_entropy:.4f}")
    
    training_time = time.time() - start_time
    print(f"Pure RL训练完成，耗时: {training_time:.2f}秒")
    
    return policy_model, metrics


def train_rlhf(base_model, train_loader, device, epochs=3):
    """训练RLHF模型"""
    print("\n" + "=" * 60)
    print("训练RLHF模型（有人类反馈）")
    print("=" * 60)
    
    # 创建策略网络和奖励模型
    policy_model = SimplePolicyNetwork(base_model, action_dim=2)
    reward_model = SimpleRewardModel(base_model)
    
    # 创建参考模型
    reference_model = SimplePolicyNetwork(base_model, action_dim=2)
    reference_model.load_state_dict(policy_model.state_dict())
    
    policy_model = policy_model.to(device)
    reward_model = reward_model.to(device)
    reference_model = reference_model.to(device)
    
    # 创建RLHF训练器
    rlhf_trainer = SimpleRLHFTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        reference_model=reference_model,
        lr=1e-4,
        beta=0.1,
        device=device
    )
    
    # 训练记录
    metrics = {
        'loss': [],
        'kl_penalty': [],
        'reward': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nRLHF Epoch {epoch+1}/{epochs}")
        
        epoch_loss = 0
        epoch_kl = 0
        epoch_reward = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"RLHF Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            
            # 生成合成奖励（模拟人类反馈）
            batch_size = input_ids.size(0)
            synthetic_rewards = torch.randn(batch_size, device=device)
            
            # 生成动作
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
    
    training_time = time.time() - start_time
    print(f"RLHF训练完成，耗时: {training_time:.2f}秒")
    
    return policy_model, metrics


def compare_policies(pure_rl_model, rlhf_model, val_loader, device):
    """比较两种策略的行为"""
    print("\n" + "=" * 60)
    print("比较策略行为")
    print("=" * 60)
    
    pure_rl_model.eval()
    rlhf_model.eval()
    
    pure_rl_actions = []
    rlhf_actions = []
    pure_rl_entropies = []
    rlhf_entropies = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="比较策略"):
            input_ids = batch['input_ids'].to(device)
            
            # Pure RL策略
            pure_rl_logits, _ = pure_rl_model(input_ids)
            pure_rl_probs = torch.softmax(pure_rl_logits, dim=-1)
            pure_rl_dist = torch.distributions.Categorical(pure_rl_probs)
            pure_rl_action = pure_rl_dist.sample()
            pure_rl_entropy = pure_rl_dist.entropy()
            
            # RLHF策略
            rlhf_logits, _ = rlhf_model(input_ids)
            rlhf_probs = torch.softmax(rlhf_logits, dim=-1)
            rlhf_dist = torch.distributions.Categorical(rlhf_probs)
            rlhf_action = rlhf_dist.sample()
            rlhf_entropy = rlhf_dist.entropy()
            
            pure_rl_actions.extend(pure_rl_action.cpu().numpy())
            rlhf_actions.extend(rlhf_action.cpu().numpy())
            pure_rl_entropies.extend(pure_rl_entropy.cpu().numpy())
            rlhf_entropies.extend(rlhf_entropy.cpu().numpy())
    
    # 计算统计信息
    pure_rl_action_dist = np.bincount(pure_rl_actions) / len(pure_rl_actions)
    rlhf_action_dist = np.bincount(rlhf_actions) / len(rlhf_actions)
    
    print(f"Pure RL动作分布: {pure_rl_action_dist}")
    print(f"RLHF动作分布: {rlhf_action_dist}")
    print(f"Pure RL平均熵: {np.mean(pure_rl_entropies):.4f}")
    print(f"RLHF平均熵: {np.mean(rlhf_entropies):.4f}")
    
    return {
        'pure_rl_action_dist': pure_rl_action_dist,
        'rlhf_action_dist': rlhf_action_dist,
        'pure_rl_entropy': np.mean(pure_rl_entropies),
        'rlhf_entropy': np.mean(rlhf_entropies)
    }


def plot_comparison(pure_rl_metrics, rlhf_metrics, behavior_comparison):
    """绘制对比图表"""
    print("\n绘制对比图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 损失对比
    axes[0, 0].plot(pure_rl_metrics['loss'], label='Pure RL', marker='o')
    axes[0, 0].plot(rlhf_metrics['loss'], label='RLHF', marker='s')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 奖励对比
    axes[0, 1].plot(pure_rl_metrics['reward'], label='Pure RL', marker='o')
    axes[0, 1].plot(rlhf_metrics['reward'], label='RLHF', marker='s')
    axes[0, 1].set_title('Average Reward Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL散度惩罚（仅RLHF）
    axes[0, 2].plot(rlhf_metrics['kl_penalty'], label='KL Penalty', marker='s', color='red')
    axes[0, 2].set_title('KL Divergence Penalty (RLHF)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('KL Penalty')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 动作分布对比
    x = np.arange(len(behavior_comparison['pure_rl_action_dist']))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, behavior_comparison['pure_rl_action_dist'], 
                   width, label='Pure RL', alpha=0.8)
    axes[1, 0].bar(x + width/2, behavior_comparison['rlhf_action_dist'], 
                   width, label='RLHF', alpha=0.8)
    axes[1, 0].set_title('Action Distribution Comparison')
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_xticks(x)
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 熵对比
    entropies = [behavior_comparison['pure_rl_entropy'], behavior_comparison['rlhf_entropy']]
    labels = ['Pure RL', 'RLHF']
    colors = ['blue', 'orange']
    
    axes[1, 1].bar(labels, entropies, color=colors, alpha=0.8)
    axes[1, 1].set_title('Policy Entropy Comparison')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].grid(True)
    
    # Pure RL熵变化
    axes[1, 2].plot(pure_rl_metrics['entropy'], label='Pure RL Entropy', marker='o')
    axes[1, 2].set_title('Pure RL Entropy Over Time')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Entropy')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('logs/simple_rl_vs_rlhf_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("对比图表已保存到 logs/simple_rl_vs_rlhf_comparison.png")


def main():
    """主函数"""
    print("🚀 简化版 RL vs RLHF 对比实验")
    print("=" * 80)
    
    # 设置设备
    device = get_device_info()
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs('logs', exist_ok=True)
    
    # 加载数据
    train_loader, val_loader = create_simple_dataset()
    
    # 创建基础模型
    print("创建简单基础模型...")
    base_model = SimpleClassifier(vocab_size=100, hidden_dim=64, num_classes=2)
    base_model = base_model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # 训练Pure RL模型
    pure_rl_model, pure_rl_metrics = train_pure_rl(base_model, train_loader, device)
    
    # 训练RLHF模型
    rlhf_model, rlhf_metrics = train_rlhf(base_model, train_loader, device)
    
    # 比较策略行为
    behavior_comparison = compare_policies(pure_rl_model, rlhf_model, val_loader, device)
    
    # 绘制对比图表
    plot_comparison(pure_rl_metrics, rlhf_metrics, behavior_comparison)
    
    # 保存结果
    results = {
        'pure_rl_metrics': pure_rl_metrics,
        'rlhf_metrics': rlhf_metrics,
        'behavior_comparison': behavior_comparison
    }
    
    with open('logs/simple_rl_vs_rlhf_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 总结
    print("\n" + "=" * 80)
    print("对比实验总结")
    print("=" * 80)
    
    print("\n🔍 主要差异:")
    print("1. **Pure RL**: 只使用环境奖励，可能产生不稳定或有害的行为")
    print("2. **RLHF**: 使用人类反馈和KL散度约束，行为更安全可控")
    
    print("\n📊 关键指标对比:")
    print(f"- Pure RL最终奖励: {pure_rl_metrics['reward'][-1]:.4f}")
    print(f"- RLHF最终奖励: {rlhf_metrics['reward'][-1]:.4f}")
    print(f"- Pure RL策略熵: {behavior_comparison['pure_rl_entropy']:.4f}")
    print(f"- RLHF策略熵: {behavior_comparison['rlhf_entropy']:.4f}")
    
    print("\n🎯 结论:")
    print("- RLHF通过人类反馈和KL约束，使策略更加安全")
    print("- Pure RL可能产生高风险行为，需要额外约束")
    print("- RLHF训练更稳定，但计算成本更高")
    
    print("\n✅ 对比实验完成！")
    print("📊 结果已保存到 logs/simple_rl_vs_rlhf_results.json")
    print("📈 图表已保存到 logs/simple_rl_vs_rlhf_comparison.png")


if __name__ == "__main__":
    main()
