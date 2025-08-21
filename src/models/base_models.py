import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional


class LightweightClassifier(nn.Module):
    """轻量级文本分类模型"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        
        # 加载预训练模型
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 获取隐藏层维度
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 冻结部分层以减少计算量
        self._freeze_layers()
    
    def _freeze_layers(self):
        """冻结部分层以减少计算量"""
        # 冻结前6层（总共12层）
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(6):
            for param in self.encoder.transformer.layer[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        # 编码
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]标记的输出
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits


class SimpleRNN(nn.Module):
    """简单的RNN生成模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, hidden=None):
        # 嵌入
        embedded = self.dropout(self.embedding(input_ids))
        
        # LSTM前向传播
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 输出层
        output = self.output(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))


class PolicyNetwork(nn.Module):
    """策略网络，用于RLHF"""
    
    def __init__(self, base_model: nn.Module, action_dim: int = 2):
        super().__init__()
        self.base_model = base_model
        
        # 获取基础模型的输出维度
        if hasattr(base_model, 'classifier'):
            # 对于分类模型
            policy_input_dim = base_model.classifier[-1].in_features
        else:
            # 对于RNN模型
            policy_input_dim = base_model.hidden_dim
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, policy_input_dim // 2),
            nn.ReLU(),
            nn.Linear(policy_input_dim // 2, action_dim)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(policy_input_dim, policy_input_dim // 2),
            nn.ReLU(),
            nn.Linear(policy_input_dim // 2, 1)
        )
    
    def forward(self, *args, **kwargs):
        # 获取基础模型的输出
        if hasattr(self.base_model, 'classifier'):
            # 分类模型
            outputs = self.base_model(*args, **kwargs)
            features = self.base_model.classifier[:-2](self.base_model.encoder(*args, **kwargs).last_hidden_state[:, 0, :])
        else:
            # RNN模型
            outputs, hidden = self.base_model(*args, **kwargs)
            features = hidden[0][-1]  # 最后一层的隐藏状态
        
        # 策略和价值
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value, outputs


class RewardModel(nn.Module):
    """奖励模型"""
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        
        # 获取基础模型的输出维度
        if hasattr(base_model, 'classifier'):
            reward_input_dim = base_model.classifier[-1].in_features
        else:
            reward_input_dim = base_model.hidden_dim
        
        # 奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(reward_input_dim, reward_input_dim // 2),
            nn.ReLU(),
            nn.Linear(reward_input_dim // 2, 1)
        )
    
    def forward(self, *args, **kwargs):
        # 获取基础模型的输出
        if hasattr(self.base_model, 'classifier'):
            features = self.base_model.classifier[:-2](self.base_model.encoder(*args, **kwargs).last_hidden_state[:, 0, :])
        else:
            outputs, hidden = self.base_model(*args, **kwargs)
            features = hidden[0][-1]
        
        # 奖励
        reward = self.reward_head(features)
        
        return reward
