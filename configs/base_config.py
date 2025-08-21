"""
基础配置文件
"""

# 数据配置
DATA_CONFIG = {
    'dataset_name': 'imdb',
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'train_samples': 1000,  # 限制训练样本数量以适应Mac Mini
    'val_samples': 200,
    'test_samples': 200,
    'num_workers': 0  # Mac Mini建议使用单进程
}

# 模型配置
MODEL_CONFIG = {
    'num_classes': 2,
    'dropout': 0.1,
    'freeze_layers': 6,  # 冻结前6层
    'vocab_size': 10000,  # 对于RNN模型
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2
}

# 训练配置
TRAINING_CONFIG = {
    'epochs': 10,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'gradient_clip': 1.0,
    'early_stopping_patience': 5,
    'lr_scheduler_patience': 3,
    'lr_scheduler_factor': 0.5
}

# RL配置
RL_CONFIG = {
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'ppo_epochs': 4,
    'buffer_size': 1000
}

# RLHF配置
RLHF_CONFIG = {
    'beta': 0.1,  # KL散度惩罚系数
    'reward_scale': 1.0,
    'reference_model_path': None,
    'reward_model_path': None
}

# 路径配置
PATH_CONFIG = {
    'data_dir': 'data',
    'model_dir': 'models',
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints'
}

# 设备配置
DEVICE_CONFIG = {
    'device': 'auto',  # 'auto', 'cpu', 'mps', 'cuda'
    'mixed_precision': False,  # Mac Mini建议关闭混合精度
    'memory_efficient': True
}
