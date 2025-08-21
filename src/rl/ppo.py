import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Any
import copy


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        reward_model: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.device = device
        
        # 超参数
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 优化器
        self.policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value_model.parameters(), lr=lr)
        
        # 将模型移到设备
        self.policy_model.to(device)
        self.value_model.to(device)
        self.reward_model.to(device)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计(GAE)"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算策略损失"""
        # 前向传播
        policy_logits, values, _ = self.policy_model(states)
        
        # 计算新的动作概率
        dist = Categorical(logits=policy_logits)
        new_log_probs = dist.log_prob(actions)
        
        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # 熵损失
        entropy_loss = -dist.entropy().mean()
        
        return policy_loss, value_loss, entropy_loss
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int = 4
    ) -> Dict[str, float]:
        """更新模型"""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(num_epochs):
            # 计算损失
            policy_loss, value_loss, entropy_loss = self.compute_policy_loss(
                states, actions, old_log_probs, advantages, returns
            )
            
            # 总损失
            total_loss = (
                policy_loss +
                self.value_loss_coef * value_loss +
                self.entropy_coef * entropy_loss
            )
            
            # 反向传播
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        return {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy_loss': total_entropy_loss / num_epochs
        }
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作"""
        with torch.no_grad():
            policy_logits, value, _ = self.policy_model(state.unsqueeze(0))
            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.squeeze(), log_prob.squeeze(), value.squeeze()


class ExperienceBuffer:
    """经验缓冲区"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, experience: Dict[str, Any]):
        """添加经验"""
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样经验"""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # 转换为张量
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.stack([exp['action'] for exp in batch])
        rewards = torch.stack([exp['reward'] for exp in batch])
        log_probs = torch.stack([exp['log_prob'] for exp in batch])
        values = torch.stack([exp['value'] for exp in batch])
        dones = torch.stack([exp['done'] for exp in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values,
            'dones': dones
        }
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class RLHFTrainer:
    """RLHF训练器"""
    
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        reference_model: nn.Module,
        lr: float = 1e-5,
        beta: float = 0.1,
        device: str = "cpu"
    ):
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
    
    def compute_kl_penalty(self, policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
        """计算KL散度惩罚"""
        policy_dist = Categorical(logits=policy_logits)
        reference_dist = Categorical(logits=reference_logits)
        
        kl_div = torch.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            F.softmax(reference_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl_div
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """训练步骤"""
        # 策略模型前向传播
        policy_logits, _, _ = self.policy_model(states)
        
        # 参考模型前向传播
        with torch.no_grad():
            reference_logits, _, _ = self.reference_model(states)
        
        # 计算动作概率
        dist = Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        
        # 计算奖励
        reward_pred = self.reward_model(states).squeeze()
        
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
