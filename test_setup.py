#!/usr/bin/env python3
"""
项目设置测试脚本
验证所有组件是否正常工作
"""

import sys
import os
import torch
import numpy as np

def test_imports():
    """测试导入"""
    print("🔍 测试导入...")
    
    try:
        from src.data.dataset import IMDBDataset, SimpleRNNDataset
        print("✅ 数据集模块导入成功")
    except Exception as e:
        print(f"❌ 数据集模块导入失败: {e}")
        return False
    
    try:
        from src.models.base_models import LightweightClassifier, SimpleRNN, PolicyNetwork, RewardModel
        print("✅ 模型模块导入成功")
    except Exception as e:
        print(f"❌ 模型模块导入失败: {e}")
        return False
    
    try:
        from src.rl.ppo import PPOTrainer, RLHFTrainer
        print("✅ RL模块导入成功")
    except Exception as e:
        print(f"❌ RL模块导入失败: {e}")
        return False
    
    try:
        from src.utils.training_utils import get_device_info, print_model_info
        print("✅ 工具模块导入成功")
    except Exception as e:
        print(f"❌ 工具模块导入失败: {e}")
        return False
    
    return True


def test_device():
    """测试设备"""
    print("\n🔍 测试设备...")
    
    device = get_device_info()
    print(f"✅ 检测到设备: {device}")
    
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) 可用")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU 可用")
    else:
        print("ℹ️  使用CPU训练")
    
    return True


def test_models():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        # 测试轻量级分类器
        classifier = LightweightClassifier(
            model_name="distilbert-base-uncased",
            num_classes=2,
            dropout=0.1
        )
        print("✅ 轻量级分类器创建成功")
        
        # 测试RNN模型
        rnn = SimpleRNN(
            vocab_size=1000,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        print("✅ RNN模型创建成功")
        
        # 测试策略网络
        policy = PolicyNetwork(classifier, action_dim=2)
        print("✅ 策略网络创建成功")
        
        # 测试奖励模型
        reward = RewardModel(classifier)
        print("✅ 奖励模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False


def test_data():
    """测试数据处理"""
    print("\n🔍 测试数据处理...")
    
    try:
        # 创建简单的文本数据
        texts = [
            "I love this movie because it is amazing",
            "This film is terrible and I hated it",
            "The story is great and the acting is wonderful"
        ]
        
        # 测试RNN数据集
        dataset = SimpleRNNDataset(
            texts=texts,
            tokenizer=None,
            max_length=10
        )
        print(f"✅ RNN数据集创建成功，词汇表大小: {dataset.vocab_size}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"✅ 数据样本加载成功，输入形状: {sample['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return False


def test_training_components():
    """测试训练组件"""
    print("\n🔍 测试训练组件...")
    
    try:
        # 创建简单的模型和数据
        model = SimpleRNN(vocab_size=100, embedding_dim=32, hidden_dim=64)
        device = get_device_info()
        model = model.to(device)
        
        # 测试优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print("✅ 优化器创建成功")
        
        # 测试损失函数
        criterion = torch.nn.CrossEntropyLoss()
        print("✅ 损失函数创建成功")
        
        # 测试前向传播
        dummy_input = torch.randint(0, 100, (2, 5)).to(device)
        output, hidden = model(dummy_input)
        print(f"✅ 前向传播成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 RLHF on Mac Mini 项目测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_device,
        test_models,
        test_data,
        test_training_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目设置正确。")
        print("\n下一步:")
        print("1. 运行演示: python scripts/demo.py")
        print("2. 或使用快速启动: ./run_demo.sh")
    else:
        print("⚠️  部分测试失败，请检查环境设置。")
        print("\n建议:")
        print("1. 检查依赖安装: pip install -r requirements.txt")
        print("2. 检查Python版本: python3 --version")
        print("3. 检查PyTorch安装: python3 -c 'import torch; print(torch.__version__)'")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
