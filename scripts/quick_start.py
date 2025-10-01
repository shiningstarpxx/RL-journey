#!/usr/bin/env python3
"""
强化学习学习项目 - 快速启动脚本

这个脚本将帮助你快速开始强化学习的学习之旅
"""

import sys
import os
import subprocess
import time

def print_banner():
    """打印项目横幅"""
    print("=" * 80)
    print("🧠 强化学习从零开始学习项目")
    print("=" * 80)
    print("欢迎来到强化学习的世界！")
    print("这个项目将帮助你系统性地学习强化学习")
    print("从基础概念到现代算法，一步步掌握RL技术")
    print("=" * 80)

def check_environment():
    """检查环境设置"""
    print("🔍 检查环境设置...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy未安装")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib未安装")
        return False
    
    try:
        import gymnasium
        print(f"✅ Gymnasium: {gymnasium.__version__}")
    except ImportError:
        print("❌ Gymnasium未安装")
        return False
    
    return True

def test_grid_world():
    """测试网格世界环境"""
    print("\n🧪 测试网格世界环境...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath('.')))
        from environments.grid_world import create_simple_grid_world
        
        env = create_simple_grid_world()
        print(f"✅ 网格世界环境创建成功!")
        print(f"   环境大小: {env.size}x{env.size}")
        print(f"   状态空间: {env.state_space}")
        print(f"   动作空间: {env.action_space}")
        
        # 测试环境交互
        state = env.reset()
        print(f"   初始状态: {state}")
        
        action = 1  # 向右
        next_state, reward, done, info = env.step(action)
        print(f"   执行动作 {action}: 状态={next_state}, 奖励={reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 网格世界环境测试失败: {e}")
        return False

def test_q_learning():
    """测试Q-Learning算法"""
    print("\n🧪 测试Q-Learning算法...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath('.')))
        from environments.grid_world import create_simple_grid_world
        from algorithms.tabular.q_learning import QLearning
        
        env = create_simple_grid_world()
        q_learning = QLearning(
            state_space=env.state_space,
            action_space=env.action_space,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
        
        print(f"✅ Q-Learning算法创建成功!")
        print(f"   Q表大小: {q_learning.Q.shape}")
        print(f"   学习率: {q_learning.learning_rate}")
        print(f"   折扣因子: {q_learning.discount_factor}")
        
        # 测试一个episode
        state = env.reset()
        state_idx = env.get_state_index(state)
        
        for step in range(10):
            action = q_learning.get_action(state_idx)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.get_state_index(next_state)
            
            q_learning.update(state_idx, action, reward, next_state_idx, done)
            
            state = next_state
            state_idx = next_state_idx
            
            if done:
                break
        
        print(f"   训练测试完成，Q表已更新")
        
        return True
        
    except Exception as e:
        print(f"❌ Q-Learning算法测试失败: {e}")
        return False

def show_learning_path():
    """显示学习路径"""
    print("\n📚 学习路径概览")
    print("=" * 50)
    
    learning_stages = [
        ("第1周", "基础概念", "RL基本概念、MDP、贝尔曼方程"),
        ("第2周", "Q-Learning", "时序差分学习、ε-贪婪策略"),
        ("第3周", "Policy Gradient", "策略梯度定理、REINFORCE算法"),
        ("第4周", "Actor-Critic", "A2C算法、优势函数"),
        ("第5-6周", "深度强化学习", "DQN、经验回放、目标网络"),
        ("第7-8周", "现代算法", "PPO、SAC、最大熵RL")
    ]
    
    for week, topic, description in learning_stages:
        print(f"{week:8} | {topic:15} | {description}")

def show_quick_experiments():
    """显示快速实验选项"""
    print("\n🚀 快速实验选项")
    print("=" * 50)
    
    experiments = [
        ("1", "基础Q-Learning实验", "python experiments/experiment_1_q_learning.py"),
        ("2", "网格世界环境演示", "python environments/grid_world.py"),
        ("3", "Q-Learning算法测试", "python algorithms/tabular/q_learning.py"),
        ("4", "启动Jupyter Notebook", "jupyter notebook"),
        ("5", "查看学习指南", "open LEARNING_GUIDE.md")
    ]
    
    for num, name, command in experiments:
        print(f"{num}. {name}")
        print(f"   命令: {command}")

def run_quick_experiment():
    """运行快速实验"""
    print("\n🎯 选择要运行的实验:")
    print("1. 基础Q-Learning实验 (推荐)")
    print("2. 网格世界环境演示")
    print("3. Q-Learning算法测试")
    print("4. 启动Jupyter Notebook")
    print("5. 查看学习指南")
    print("0. 退出")
    
    while True:
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            print("\n🚀 运行基础Q-Learning实验...")
            try:
                subprocess.run([sys.executable, "experiments/experiment_1_q_learning.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ 实验运行失败: {e}")
            except FileNotFoundError:
                print("❌ 实验文件未找到，请确保在正确的目录中运行")
            break
        elif choice == "2":
            print("\n🚀 运行网格世界环境演示...")
            try:
                subprocess.run([sys.executable, "environments/grid_world.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ 演示运行失败: {e}")
            except FileNotFoundError:
                print("❌ 演示文件未找到")
            break
        elif choice == "3":
            print("\n🚀 运行Q-Learning算法测试...")
            try:
                subprocess.run([sys.executable, "algorithms/tabular/q_learning.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ 测试运行失败: {e}")
            except FileNotFoundError:
                print("❌ 测试文件未找到")
            break
        elif choice == "4":
            print("\n🚀 启动Jupyter Notebook...")
            try:
                subprocess.run(["jupyter", "notebook"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Jupyter启动失败: {e}")
            except FileNotFoundError:
                print("❌ Jupyter未安装，请运行: pip install jupyter")
            break
        elif choice == "5":
            print("\n📖 查看学习指南...")
            try:
                if os.name == 'nt':  # Windows
                    os.startfile("LEARNING_GUIDE.md")
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(["open", "LEARNING_GUIDE.md"], check=True)
            except:
                print("无法自动打开文件，请手动打开 LEARNING_GUIDE.md")
            break
        else:
            print("❌ 无效选择，请输入 0-5")

def show_next_steps():
    """显示下一步建议"""
    print("\n🎯 下一步学习建议")
    print("=" * 50)
    
    print("1. 📖 阅读学习指南")
    print("   - 查看 LEARNING_GUIDE.md")
    print("   - 了解完整的学习路径")
    
    print("\n2. 🧪 运行第一个实验")
    print("   - 执行: python experiments/experiment_1_q_learning.py")
    print("   - 观察Q-Learning的学习过程")
    
    print("\n3. 📚 学习基础概念")
    print("   - 阅读强化学习导论第1-3章")
    print("   - 理解MDP和贝尔曼方程")
    
    print("\n4. 💻 动手实践")
    print("   - 修改环境参数")
    print("   - 调整算法参数")
    print("   - 观察对性能的影响")
    
    print("\n5. 📝 记录学习")
    print("   - 写学习笔记")
    print("   - 记录问题和解决方案")
    print("   - 分享学习心得")

def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请先安装依赖:")
        print("pip install -r requirements.txt")
        return
    
    # 测试组件
    grid_world_ok = test_grid_world()
    q_learning_ok = test_q_learning()
    
    if not grid_world_ok or not q_learning_ok:
        print("\n❌ 组件测试失败，请检查代码")
        return
    
    print("\n✅ 所有测试通过！环境设置成功！")
    
    # 显示学习路径
    show_learning_path()
    
    # 显示快速实验
    show_quick_experiments()
    
    # 运行实验
    run_quick_experiment()
    
    # 显示下一步建议
    show_next_steps()
    
    print("\n" + "=" * 80)
    print("🎉 强化学习学习项目启动成功！")
    print("开始你的RL学习之旅吧！")
    print("=" * 80)

if __name__ == "__main__":
    main()
