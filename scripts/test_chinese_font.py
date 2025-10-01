#!/usr/bin/env python3
"""
中文字体显示测试脚本
验证项目中所有组件的中文显示效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from utils.font_config import setup_chinese_font

def test_basic_chinese_display():
    """测试基础中文显示"""
    print("🧪 测试基础中文显示...")
    
    # 设置中文字体
    font = setup_chinese_font()
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 测试1: 柱状图
    algorithms = ['Q-Learning', 'SARSA', 'Policy Gradient', 'Actor-Critic']
    performance = [0.85, 0.82, 0.78, 0.88]
    
    axes[0, 0].bar(algorithms, performance, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('强化学习算法性能比较', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('成功率', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 测试2: 折线图
    episodes = np.arange(1, 101)
    rewards = 0.5 + 0.3 * (1 - np.exp(-episodes/30)) + 0.1 * np.random.random(100)
    
    axes[0, 1].plot(episodes, rewards, linewidth=2, color='#2E86AB')
    axes[0, 1].set_title('训练过程中的奖励变化', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('训练轮次', fontsize=12)
    axes[0, 1].set_ylabel('累积奖励', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 测试3: 散点图
    x = np.random.randn(50)
    y = 2 * x + np.random.randn(50)
    
    axes[1, 0].scatter(x, y, alpha=0.6, color='#E74C3C')
    axes[1, 0].set_title('状态价值与动作价值关系', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('状态价值', fontsize=12)
    axes[1, 0].set_ylabel('动作价值', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 测试4: 饼图
    labels = ['探索', '利用', '学习', '评估']
    sizes = [25, 35, 25, 15]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('强化学习时间分配', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('logs/chinese_display_comprehensive_test.png', dpi=300, bbox_inches='tight')
    print("📊 综合中文显示测试图片已保存到: logs/chinese_display_comprehensive_test.png")
    plt.show()
    
    return font


def test_algorithm_components():
    """测试算法组件的中文显示"""
    print("\n🧪 测试算法组件中文显示...")
    
    from algorithms.tabular.q_learning import QLearning
    from environments.grid_world import create_simple_grid_world
    
    # 创建环境和算法
    env = create_simple_grid_world()
    q_learning = QLearning(
        state_space=env.state_space,
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2
    )
    
    # 训练几个episode
    rewards = []
    epsilons = []
    for episode in range(20):
        reward, steps = q_learning.train_episode(env, max_steps=50)
        q_learning.decay_epsilon()
        rewards.append(reward)
        epsilons.append(q_learning.epsilon)
    
    # 创建算法测试图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 学习曲线
    episodes = range(1, len(rewards) + 1)
    axes[0].plot(episodes, rewards, 'o-', linewidth=2, markersize=4, color='#2E86AB')
    axes[0].set_title('Q-Learning学习曲线', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('训练轮次', fontsize=12)
    axes[0].set_ylabel('累积奖励', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Epsilon衰减
    axes[1].plot(episodes, epsilons, 's-', linewidth=2, markersize=4, color='#E74C3C')
    axes[1].set_title('探索概率衰减过程', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('训练轮次', fontsize=12)
    axes[1].set_ylabel('探索概率 (ε)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 性能统计
    stats = ['平均奖励', '最大奖励', '最小奖励', '标准差']
    values = [np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards)]
    
    bars = axes[2].bar(stats, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[2].set_title('训练性能统计', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('数值', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('logs/algorithm_components_chinese_test.png', dpi=300, bbox_inches='tight')
    print("📊 算法组件中文显示测试图片已保存到: logs/algorithm_components_chinese_test.png")
    plt.show()


def test_learning_tracker():
    """测试学习跟踪器的中文显示"""
    print("\n🧪 测试学习跟踪器中文显示...")
    
    from progress.learning_tracker import LearningTracker
    
    # 创建学习跟踪器
    tracker = LearningTracker()
    
    # 模拟一些学习数据
    tracker.start_week(1)
    tracker.complete_topic(1, "RL基本概念")
    tracker.complete_topic(1, "MDP理论")
    tracker.complete_exercise(1, "基础概念测验")
    tracker.add_study_session(1, 2.5, "理论学习", "学习了RL基本概念")
    tracker.add_study_session(1, 1.5, "实践练习", "完成了基础概念测验")
    
    # 创建学习跟踪图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 学习进度
    weeks = ['第1周', '第2周', '第3周', '第4周']
    progress = [40, 0, 0, 0]  # 模拟进度
    
    bars = axes[0, 0].bar(weeks, progress, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('每周学习进度', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('完成百分比 (%)', fontsize=12)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 学习时间分布
    activities = ['理论学习', '实践练习', '项目开发', '复习总结']
    time_spent = [2.5, 1.5, 0, 0]  # 模拟时间
    
    axes[0, 1].pie(time_spent, labels=activities, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('学习时间分布', fontsize=14, fontweight='bold')
    
    # 成就统计
    achievements = ['基础概念', '算法实现', '项目完成', '考试通过']
    counts = [1, 0, 0, 0]  # 模拟成就
    
    axes[1, 0].bar(achievements, counts, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
    axes[1, 0].set_title('学习成就统计', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('成就数量', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习趋势
    days = range(1, 8)
    daily_time = [0.5, 1.0, 0.8, 1.2, 0.6, 0.9, 1.1]  # 模拟每日学习时间
    
    axes[1, 1].plot(days, daily_time, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    axes[1, 1].set_title('每日学习时间趋势', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('天数', fontsize=12)
    axes[1, 1].set_ylabel('学习时间 (小时)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/learning_tracker_chinese_test.png', dpi=300, bbox_inches='tight')
    print("📊 学习跟踪器中文显示测试图片已保存到: logs/learning_tracker_chinese_test.png")
    plt.show()


def main():
    """主函数"""
    print("🔧 强化学习项目中文字体显示测试")
    print("=" * 60)
    
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 测试基础中文显示
    font = test_basic_chinese_display()
    
    # 测试算法组件
    test_algorithm_components()
    
    # 测试学习跟踪器
    test_learning_tracker()
    
    print("\n" + "=" * 60)
    print("🎉 中文字体显示测试完成！")
    print(f"✅ 使用的字体: {font}")
    print("📁 生成的测试图片:")
    print("  • logs/chinese_display_comprehensive_test.png")
    print("  • logs/algorithm_components_chinese_test.png")
    print("  • logs/learning_tracker_chinese_test.png")
    print("\n💡 如果中文显示正常，说明字体配置成功！")
    print("如果仍有乱码，请检查系统字体安装情况。")


if __name__ == "__main__":
    main()
