import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Any, Tuple
import time


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_model(model: nn.Module, save_path: str, config: Dict[str, Any] = None):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型状态
    torch.save(model.state_dict(), save_path)
    
    # 保存配置
    if config:
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def load_model(model: nn.Module, load_path: str, device: str = "cpu") -> nn.Module:
    """加载模型"""
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model


def plot_training_curves(metrics: Dict[str, List[float]], save_path: str = None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 损失曲线
    if 'train_loss' in metrics:
        axes[0, 0].plot(metrics['train_loss'], label='Train Loss')
    if 'val_loss' in metrics:
        axes[0, 0].plot(metrics['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # 准确率曲线
    if 'train_acc' in metrics:
        axes[0, 1].plot(metrics['train_acc'], label='Train Acc')
    if 'val_acc' in metrics:
        axes[0, 1].plot(metrics['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    
    # F1分数曲线
    if 'train_f1' in metrics:
        axes[1, 0].plot(metrics['train_f1'], label='Train F1')
    if 'val_f1' in metrics:
        axes[1, 0].plot(metrics['val_f1'], label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    
    # 学习率曲线
    if 'lr' in metrics:
        axes[1, 1].plot(metrics['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        return False


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer, initial_lr: float, decay_factor: float = 0.5, patience: int = 3):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.counter = 0
        self.best_loss = None
    
    def step(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
    
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.decay_factor


def generate_text(model: nn.Module, tokenizer, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
    """生成文本"""
    model.eval()
    
    # 编码提示
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def compute_reward(text: str, reward_model: nn.Module, tokenizer) -> float:
    """计算文本奖励"""
    reward_model.eval()
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    
    with torch.no_grad():
        reward = reward_model(**inputs)
    
    return reward.item()


def create_synthetic_rewards(texts: List[str]) -> List[float]:
    """创建合成奖励（用于演示）"""
    rewards = []
    for text in texts:
        # 简单的启发式奖励函数
        reward = 0.0
        
        # 长度奖励
        if 10 <= len(text.split()) <= 50:
            reward += 0.3
        
        # 情感奖励（简单关键词检测）
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            reward += 0.4
        elif neg_count > pos_count:
            reward -= 0.2
        
        # 多样性奖励
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        if total_words > 0:
            diversity = unique_words / total_words
            reward += diversity * 0.3
        
        rewards.append(reward)
    
    return rewards


def print_model_info(model: nn.Module):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")


def get_device_info() -> str:
    """获取设备信息"""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"
