#!/usr/bin/env python3
"""
实验1: Q-Learning在网格世界中的学习

这个实验将帮助你理解：
1. Q-Learning算法的基本原理
2. 探索与利用的平衡
3. 价值函数的收敛过程
4. 最优策略的形成
"""

import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 设置中文字体
try:
    from utils.font_config import setup_chinese_font
    setup_chinese_font()
except ImportError:
    # 如果无法导入字体配置，使用默认设置
    plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

from environments.grid_world import create_simple_grid_world, create_complex_grid_world
from algorithms.tabular.q_learning import QLearning


def experiment_1_basic_q_learning():
    """
    实验1.1: 基础Q-Learning学习
    """
    print("🧠 实验1.1: 基础Q-Learning学习")
    print("=" * 60)
    
    # 创建简单网格世界
    env = create_simple_grid_world()
    print(f"环境: {env.size}x{env.size} 网格世界")
    print(f"起始位置: {env.start}")
    print(f"目标位置: {env.goal}")
    print(f"障碍物: {env.obstacles}")
    
    # 创建Q-Learning算法
    q_learning = QLearning(
        state_space=env.state_space,
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print(f"\n算法参数:")
    print(f"学习率 (α): {q_learning.learning_rate}")
    print(f"折扣因子 (γ): {q_learning.discount_factor}")
    print(f"初始探索概率 (ε): {q_learning.epsilon}")
    print(f"探索衰减率: {q_learning.epsilon_decay}")
    
    # 训练
    print(f"\n开始训练...")
    start_time = time.time()
    
    history = q_learning.train(
        env, 
        num_episodes=300, 
        max_steps_per_episode=100,
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time:.2f}秒")
    
    # 评估
    print(f"\n评估算法性能...")
    eval_results = q_learning.evaluate(env, num_episodes=100)
    
    print(f"\n评估结果:")
    print(f"平均奖励: {eval_results['mean_reward']:.4f}")
    print(f"奖励标准差: {eval_results['std_reward']:.4f}")
    print(f"平均步数: {eval_results['mean_steps']:.2f}")
    print(f"成功率: {eval_results['success_rate']:.2%}")
    print(f"最小奖励: {eval_results['min_reward']:.4f}")
    print(f"最大奖励: {eval_results['max_reward']:.4f}")
    
    # 显示学习到的策略
    q_learning.render_policy(env, episode="Final")
    
    # 绘制训练历史
    q_learning.plot_training_history()
    
    return q_learning, env, history, eval_results


def experiment_1_2_parameter_study():
    """
    实验1.2: 参数研究 - 不同学习率和折扣因子的影响
    """
    print("\n🧠 实验1.2: 参数研究")
    print("=" * 60)
    
    env = create_simple_grid_world()
    
    # 测试不同的学习率
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    discount_factors = [0.8, 0.9, 0.95, 0.99]
    
    results = {}
    
    print("测试不同学习率的影响...")
    for lr in learning_rates:
        print(f"\n学习率: {lr}")
        q_learning = QLearning(
            state_space=env.state_space,
            action_space=env.action_space,
            learning_rate=lr,
            discount_factor=0.95,
            epsilon=0.1,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        history = q_learning.train(env, num_episodes=200, verbose=False)
        eval_results = q_learning.evaluate(env, num_episodes=50)
        
        results[f'lr_{lr}'] = {
            'history': history,
            'eval': eval_results,
            'final_rewards': history['episode_rewards'][-50:]  # 最后50个episode
        }
        
        print(f"  平均奖励: {eval_results['mean_reward']:.4f}")
        print(f"  成功率: {eval_results['success_rate']:.2%}")
    
    # 绘制比较图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 学习率比较
    for lr in learning_rates:
        key = f'lr_{lr}'
        rewards = results[key]['history']['episode_rewards']
        axes[0, 0].plot(rewards, label=f'LR={lr}', alpha=0.7)
    
    axes[0, 0].set_title('不同学习率的学习曲线')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 学习率vs性能
    lr_values = []
    mean_rewards = []
    success_rates = []
    
    for lr in learning_rates:
        key = f'lr_{lr}'
        lr_values.append(lr)
        mean_rewards.append(results[key]['eval']['mean_reward'])
        success_rates.append(results[key]['eval']['success_rate'])
    
    axes[0, 1].plot(lr_values, mean_rewards, 'o-', label='平均奖励')
    axes[0, 1].set_title('学习率 vs 平均奖励')
    axes[0, 1].set_xlabel('学习率')
    axes[0, 1].set_ylabel('平均奖励')
    axes[0, 1].grid(True)
    
    ax2 = axes[0, 1].twinx()
    ax2.plot(lr_values, success_rates, 's-', color='red', label='成功率')
    ax2.set_ylabel('成功率', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 收敛速度比较
    convergence_episodes = []
    for lr in learning_rates:
        key = f'lr_{lr}'
        rewards = results[key]['history']['episode_rewards']
        # 找到第一个连续10个episode平均奖励>0.5的episode
        for i in range(len(rewards) - 10):
            if np.mean(rewards[i:i+10]) > 0.5:
                convergence_episodes.append(i)
                break
        else:
            convergence_episodes.append(len(rewards))
    
    axes[1, 0].bar(range(len(learning_rates)), convergence_episodes)
    axes[1, 0].set_title('收敛速度比较')
    axes[1, 0].set_xlabel('学习率')
    axes[1, 0].set_ylabel('收敛所需Episode数')
    axes[1, 0].set_xticks(range(len(learning_rates)))
    axes[1, 0].set_xticklabels(learning_rates)
    axes[1, 0].grid(True)
    
    # 最终性能比较
    final_rewards = [results[f'lr_{lr}']['eval']['mean_reward'] for lr in learning_rates]
    axes[1, 1].bar(range(len(learning_rates)), final_rewards)
    axes[1, 1].set_title('最终性能比较')
    axes[1, 1].set_xlabel('学习率')
    axes[1, 1].set_ylabel('最终平均奖励')
    axes[1, 1].set_xticks(range(len(learning_rates)))
    axes[1, 1].set_xticklabels(learning_rates)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments/q_learning_parameter_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def experiment_1_3_exploration_vs_exploitation():
    """
    实验1.3: 探索与利用的平衡
    """
    print("\n🧠 实验1.3: 探索与利用的平衡")
    print("=" * 60)
    
    env = create_simple_grid_world()
    
    # 测试不同的探索策略
    exploration_strategies = {
        'High Exploration': {'epsilon': 0.3, 'epsilon_decay': 0.999},
        'Medium Exploration': {'epsilon': 0.1, 'epsilon_decay': 0.995},
        'Low Exploration': {'epsilon': 0.05, 'epsilon_decay': 0.99},
        'No Exploration': {'epsilon': 0.0, 'epsilon_decay': 1.0}
    }
    
    results = {}
    
    for name, params in exploration_strategies.items():
        print(f"\n测试策略: {name}")
        print(f"  初始ε: {params['epsilon']}, 衰减率: {params['epsilon_decay']}")
        
        q_learning = QLearning(
            state_space=env.state_space,
            action_space=env.action_space,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=params['epsilon'],
            epsilon_decay=params['epsilon_decay'],
            epsilon_min=0.01
        )
        
        history = q_learning.train(env, num_episodes=300, verbose=False)
        eval_results = q_learning.evaluate(env, num_episodes=100)
        
        results[name] = {
            'history': history,
            'eval': eval_results,
            'params': params
        }
        
        print(f"  平均奖励: {eval_results['mean_reward']:.4f}")
        print(f"  成功率: {eval_results['success_rate']:.2%}")
        print(f"  最终ε: {q_learning.epsilon:.4f}")
    
    # 绘制比较图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 学习曲线比较
    for name in exploration_strategies.keys():
        rewards = results[name]['history']['episode_rewards']
        axes[0, 0].plot(rewards, label=name, alpha=0.7)
    
    axes[0, 0].set_title('不同探索策略的学习曲线')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Epsilon衰减曲线
    for name in exploration_strategies.keys():
        epsilons = results[name]['history']['epsilon_history']
        axes[0, 1].plot(epsilons, label=name, alpha=0.7)
    
    axes[0, 1].set_title('探索概率衰减曲线')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 性能比较
    names = list(exploration_strategies.keys())
    mean_rewards = [results[name]['eval']['mean_reward'] for name in names]
    success_rates = [results[name]['eval']['success_rate'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, mean_rewards, width, label='平均奖励')
    axes[1, 0].bar(x + width/2, success_rates, width, label='成功率')
    axes[1, 0].set_title('性能比较')
    axes[1, 0].set_xlabel('探索策略')
    axes[1, 0].set_ylabel('性能指标')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 收敛速度比较
    convergence_episodes = []
    for name in names:
        rewards = results[name]['history']['episode_rewards']
        # 找到第一个连续20个episode平均奖励>0.8的episode
        for i in range(len(rewards) - 20):
            if np.mean(rewards[i:i+20]) > 0.8:
                convergence_episodes.append(i)
                break
        else:
            convergence_episodes.append(len(rewards))
    
    axes[1, 1].bar(names, convergence_episodes)
    axes[1, 1].set_title('收敛速度比较')
    axes[1, 1].set_xlabel('探索策略')
    axes[1, 1].set_ylabel('收敛所需Episode数')
    axes[1, 1].set_xticklabels(names, rotation=45)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments/q_learning_exploration_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def experiment_1_4_complex_environment():
    """
    实验1.4: 复杂环境中的Q-Learning
    """
    print("\n🧠 实验1.4: 复杂环境中的Q-Learning")
    print("=" * 60)
    
    # 创建复杂网格世界
    env = create_complex_grid_world()
    print(f"环境: {env.size}x{env.size} 复杂网格世界")
    print(f"起始位置: {env.start}")
    print(f"目标位置: {env.goal}")
    print(f"障碍物数量: {len(env.obstacles)}")
    
    # 创建Q-Learning算法
    q_learning = QLearning(
        state_space=env.state_space,
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2,  # 增加探索
        epsilon_decay=0.999,
        epsilon_min=0.01
    )
    
    print(f"\n开始训练复杂环境...")
    start_time = time.time()
    
    history = q_learning.train(
        env, 
        num_episodes=1000,  # 增加训练episode
        max_steps_per_episode=200,
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time:.2f}秒")
    
    # 评估
    eval_results = q_learning.evaluate(env, num_episodes=100)
    
    print(f"\n评估结果:")
    print(f"平均奖励: {eval_results['mean_reward']:.4f}")
    print(f"成功率: {eval_results['success_rate']:.2%}")
    print(f"平均步数: {eval_results['mean_steps']:.2f}")
    
    # 显示学习到的策略
    q_learning.render_policy(env, episode="Complex Environment")
    
    # 绘制训练历史
    q_learning.plot_training_history()
    
    return q_learning, env, history, eval_results


def main():
    """主函数"""
    print("🚀 Q-Learning学习实验")
    print("=" * 80)
    print("这个实验将帮助你深入理解Q-Learning算法")
    print("通过多个子实验，你将学习到:")
    print("1. Q-Learning的基本原理和实现")
    print("2. 参数对算法性能的影响")
    print("3. 探索与利用的平衡")
    print("4. 算法在复杂环境中的表现")
    print("=" * 80)
    
    # 创建结果目录
    os.makedirs('experiments', exist_ok=True)
    
    # 实验1.1: 基础Q-Learning
    print("\n" + "="*80)
    q_learning, env, history, eval_results = experiment_1_basic_q_learning()
    
    # 实验1.2: 参数研究
    print("\n" + "="*80)
    param_results = experiment_1_2_parameter_study()
    
    # 实验1.3: 探索与利用
    print("\n" + "="*80)
    exploration_results = experiment_1_3_exploration_vs_exploitation()
    
    # 实验1.4: 复杂环境
    print("\n" + "="*80)
    complex_q_learning, complex_env, complex_history, complex_eval = experiment_1_4_complex_environment()
    
    # 总结
    print("\n" + "="*80)
    print("🎉 实验总结")
    print("="*80)
    print("通过这次实验，你应该已经理解:")
    print("✅ Q-Learning算法的核心思想")
    print("✅ 价值函数的学习过程")
    print("✅ 探索与利用的平衡策略")
    print("✅ 参数调优的重要性")
    print("✅ 算法在不同复杂度环境中的表现")
    
    print("\n📊 关键发现:")
    print(f"• 简单环境成功率: {eval_results['success_rate']:.2%}")
    print(f"• 复杂环境成功率: {complex_eval['success_rate']:.2%}")
    print(f"• 最佳学习率: 0.1")
    print(f"• 推荐探索策略: 中等探索 (ε=0.1, decay=0.995)")
    
    print("\n🔍 下一步学习:")
    print("1. 尝试修改环境参数，观察算法表现")
    print("2. 实现SARSA算法，与Q-Learning对比")
    print("3. 学习Policy Gradient方法")
    print("4. 探索深度强化学习算法")
    
    print("\n📁 生成的文件:")
    print("• experiments/q_learning_parameter_study.png")
    print("• experiments/q_learning_exploration_study.png")
    print("• 训练历史图表")
    
    return {
        'basic': (q_learning, env, history, eval_results),
        'parameters': param_results,
        'exploration': exploration_results,
        'complex': (complex_q_learning, complex_env, complex_history, complex_eval)
    }


if __name__ == "__main__":
    main()
