#!/usr/bin/env python3
"""
RLHF演示脚本
展示完整的RLHF流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.data.dataset import IMDBDataset, SimpleRNNDataset
from src.models.base_models import LightweightClassifier, SimpleRNN, PolicyNetwork, RewardModel
from src.rl.ppo import RLHFTrainer
from src.utils.training_utils import (
    evaluate_model, save_model, print_model_info, get_device_info,
    create_synthetic_rewards, generate_text_sample
)
from configs.base_config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, RLHF_CONFIG


def demo_base_model():
    """演示基础模型训练"""
    print("=" * 50)
    print("阶段1: 基础模型训练")
    print("=" * 50)
    
    device = get_device_info()
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载IMDB数据集...")
    dataset_loader = IMDBDataset(
        model_name=DATA_CONFIG['model_name'],
        max_length=DATA_CONFIG['max_length']
    )
    
    train_dataset = dataset_loader.load_data(split="train", max_samples=200)
    val_dataset = dataset_loader.load_data(split="test", max_samples=50)
    
    train_loader = dataset_loader.get_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_loader = dataset_loader.get_dataloader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    
    # 创建模型
    print("创建基础分类模型...")
    base_model = LightweightClassifier(
        model_name=DATA_CONFIG['model_name'],
        num_classes=MODEL_CONFIG['num_classes'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    print_model_info(base_model)
    base_model = base_model.to(device)
    
    # 训练
    print("开始训练基础模型...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=2e-5)
    
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}/2")
        
        # 训练
        base_model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        val_metrics = evaluate_model(base_model, val_loader, device)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    print("基础模型训练完成！")
    return base_model


def demo_rlhf(base_model):
    """演示RLHF训练"""
    print("\n" + "=" * 50)
    print("阶段2: RLHF训练")
    print("=" * 50)
    
    device = get_device_info()
    
    # 加载数据
    dataset_loader = IMDBDataset(
        model_name=DATA_CONFIG['model_name'],
        max_length=DATA_CONFIG['max_length']
    )
    
    train_dataset = dataset_loader.load_data(split="train", max_samples=100)
    train_loader = dataset_loader.get_dataloader(train_dataset, batch_size=4, shuffle=True)
    
    # 创建策略网络和奖励模型
    print("创建策略网络和奖励模型...")
    policy_model = PolicyNetwork(base_model, action_dim=2)
    reward_model = RewardModel(base_model)
    
    # 创建参考模型
    reference_model = LightweightClassifier(
        model_name=DATA_CONFIG['model_name'],
        num_classes=MODEL_CONFIG['num_classes'],
        dropout=MODEL_CONFIG['dropout']
    )
    reference_model.load_state_dict(base_model.state_dict())
    
    policy_model = policy_model.to(device)
    reward_model = reward_model.to(device)
    reference_model = reference_model.to(device)
    
    print_model_info(policy_model)
    
    # 创建RLHF训练器
    rlhf_trainer = RLHFTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        reference_model=reference_model,
        lr=1e-5,
        beta=0.1,
        device=device
    )
    
    # 训练
    print("开始RLHF训练...")
    
    for epoch in range(2):
        print(f"\nRLHF Epoch {epoch+1}/2")
        
        epoch_loss = 0
        epoch_reward = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"RLHF Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 生成合成奖励
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
            epoch_reward += step_metrics['reward']
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_reward = epoch_reward / num_batches
        
        print(f"Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
    
    print("RLHF训练完成！")
    return policy_model


def demo_rnn_generator():
    """演示RNN生成模型"""
    print("\n" + "=" * 50)
    print("阶段3: RNN生成模型")
    print("=" * 50)
    
    device = get_device_info()
    
    # 创建简单的文本数据
    print("创建训练数据...")
    
    def create_simple_text_data(num_samples=200):
        texts = []
        templates = [
            "I love this movie because it is {}",
            "This film is {} and I enjoyed it",
            "The story is {} and the acting is {}",
            "I think this is a {} movie"
        ]
        
        adjectives = [
            "amazing", "wonderful", "excellent", "great", "good",
            "bad", "terrible", "awful", "horrible", "disappointing"
        ]
        
        for _ in range(num_samples):
            template = np.random.choice(templates)
            if template.count("{}") == 1:
                text = template.format(np.random.choice(adjectives))
            elif template.count("{}") == 2:
                text = template.format(np.random.choice(adjectives), np.random.choice(adjectives))
            else:
                text = template
            
            words = text.split()[:15]
            texts.append(" ".join(words))
        
        return texts
    
    train_texts = create_simple_text_data(200)
    
    # 创建RNN数据集
    dataset = SimpleRNNDataset(
        texts=train_texts,
        tokenizer=None,
        max_length=15
    )
    
    print(f"词汇表大小: {dataset.vocab_size}")
    print(f"训练样本: {len(dataset)}")
    
    # 创建RNN模型
    print("创建RNN生成模型...")
    rnn_model = SimpleRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )
    
    print_model_info(rnn_model)
    rnn_model = rnn_model.to(device)
    
    # 训练
    print("开始训练RNN模型...")
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.token_to_idx['<PAD>'])
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
    
    for epoch in range(5):
        print(f"\nRNN Epoch {epoch+1}/5")
        
        rnn_model.train()
        epoch_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            output, _ = rnn_model(input_ids)
            
            loss = criterion(output.view(-1, rnn_model.vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Loss: {avg_loss:.4f}")
        
        # 生成样本
        if epoch % 2 == 0:
            print("\n生成样本:")
            for prompt in ["I love", "This movie", "The film"]:
                generated = generate_text_sample(rnn_model, dataset, device, prompt)
                print(f"'{prompt}' -> '{generated}'")
    
    print("RNN模型训练完成！")
    return rnn_model, dataset


def main():
    """主演示函数"""
    print("RLHF on Mac Mini 完整演示")
    print("=" * 60)
    
    # 阶段1: 基础模型
    base_model = demo_base_model()
    
    # 阶段2: RLHF
    policy_model = demo_rlhf(base_model)
    
    # 阶段3: RNN生成
    rnn_model, dataset = demo_rnn_generator()
    
    # 总结
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    print("\n项目总结:")
    print("1. ✅ 基础模型训练 (监督学习)")
    print("2. ✅ RLHF训练 (强化学习)")
    print("3. ✅ RNN生成模型 (生成式AI)")
    
    print(f"\n模型参数统计:")
    print(f"- 基础模型: {sum(p.numel() for p in base_model.parameters()):,} 参数")
    print(f"- 策略网络: {sum(p.numel() for p in policy_model.parameters()):,} 参数")
    print(f"- RNN模型: {sum(p.numel() for p in rnn_model.parameters()):,} 参数")
    
    print(f"\n设备信息: {get_device_info()}")
    print("\n所有模型都经过优化，适合在Mac Mini上运行！")


if __name__ == "__main__":
    main()
