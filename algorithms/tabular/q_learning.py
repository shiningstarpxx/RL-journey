"""
Q-Learning算法实现
这是强化学习中最经典的算法之一
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import random
from tqdm import tqdm

# 设置中文字体
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.font_config import setup_chinese_font
    setup_chinese_font()
except ImportError:
    # 如果无法导入字体配置，使用默认设置
    plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


class QLearning:
    """
    Q-Learning算法
    
    核心思想:
    - 学习状态-动作价值函数Q(s,a)
    - 使用ε-贪婪策略平衡探索与利用
    - 通过时序差分学习更新Q值
    """
    
    def __init__(self, state_space: int, action_space: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化Q-Learning算法
        
        Args:
            state_space: 状态空间大小
            action_space: 动作空间大小
            learning_rate: 学习率 (α)
            discount_factor: 折扣因子 (γ)
            epsilon: 探索概率
            epsilon_decay: epsilon衰减率
            epsilon_min: epsilon最小值
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表
        self.Q = np.zeros((state_space, action_space))
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'q_table_changes': []
        }
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        
    def get_action(self, state: int, valid_actions: Optional[List[int]] = None) -> int:
        """
        选择动作 (ε-贪婪策略)
        
        Args:
            state: 当前状态
            valid_actions: 有效动作列表
            
        Returns:
            选择的动作
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_space))
        
        # ε-贪婪策略
        if random.random() < self.epsilon:
            # 探索: 随机选择有效动作
            return random.choice(valid_actions)
        else:
            # 利用: 选择Q值最大的动作
            q_values = self.Q[state]
            # 只考虑有效动作
            valid_q_values = q_values[valid_actions]
            best_valid_action_idx = np.argmax(valid_q_values)
            return valid_actions[best_valid_action_idx]
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool, next_valid_actions: Optional[List[int]] = None):
        """
        更新Q值 (Q-Learning更新规则)
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            next_valid_actions: 下一个状态的有效动作
        """
        if next_valid_actions is None:
            next_valid_actions = list(range(self.action_space))
        
        # 计算目标Q值
        if done:
            target = reward
        else:
            # 只考虑有效动作的最大Q值
            next_q_values = self.Q[next_state][next_valid_actions]
            max_next_q = np.max(next_q_values)
            target = reward + self.discount_factor * max_next_q
        
        # Q-Learning更新
        current_q = self.Q[state, action]
        self.Q[state, action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """衰减探索概率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """
        训练一个episode
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            (总奖励, 步数)
        """
        state = env.reset()
        if hasattr(env, 'get_state_index'):
            state_idx = env.get_state_index(state)
        else:
            state_idx = state
        
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # 获取有效动作
            if hasattr(env, 'get_valid_actions'):
                valid_actions = env.get_valid_actions(state)
            else:
                valid_actions = None
            
            # 选择动作
            action = self.get_action(state_idx, valid_actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            if hasattr(env, 'get_state_index'):
                next_state_idx = env.get_state_index(next_state)
            else:
                next_state_idx = next_state
            
            # 获取下一个状态的有效动作
            if hasattr(env, 'get_valid_actions'):
                next_valid_actions = env.get_valid_actions(next_state)
            else:
                next_valid_actions = None
            
            # 更新Q值
            self.update(state_idx, action, reward, next_state_idx, done, next_valid_actions)
            
            total_reward += reward
            steps += 1
            state = next_state
            state_idx = next_state_idx
            
            if done:
                break
        
        # 更新统计信息
        self.episode_count += 1
        self.total_steps += steps
        
        # 记录历史
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(steps)
        self.training_history['epsilon_history'].append(self.epsilon)
        
        return total_reward, steps
    
    def train(self, env, num_episodes: int = 1000, max_steps_per_episode: int = 1000,
              render_every: Optional[int] = None, verbose: bool = True) -> Dict[str, List]:
        """
        训练算法
        
        Args:
            env: 环境
            num_episodes: 训练episode数量
            max_steps_per_episode: 每个episode的最大步数
            render_every: 每隔多少个episode渲染一次
            verbose: 是否显示进度条
            
        Returns:
            训练历史
        """
        if verbose:
            pbar = tqdm(range(num_episodes), desc="Training Q-Learning")
        else:
            pbar = range(num_episodes)
        
        for episode in pbar:
            # 训练一个episode
            reward, steps = self.train_episode(env, max_steps_per_episode)
            
            # 衰减epsilon
            self.decay_epsilon()
            
            # 渲染
            if render_every and episode % render_every == 0:
                self.render_policy(env, episode)
            
            # 更新进度条
            if verbose:
                pbar.set_postfix({
                    'Episode': episode + 1,
                    'Reward': f'{reward:.2f}',
                    'Steps': steps,
                    'Epsilon': f'{self.epsilon:.3f}'
                })
        
        return self.training_history
    
    def get_policy(self) -> np.ndarray:
        """
        获取最优策略
        
        Returns:
            策略数组，每个状态对应最优动作
        """
        return np.argmax(self.Q, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """
        获取价值函数
        
        Returns:
            价值函数数组，每个状态对应最大Q值
        """
        return np.max(self.Q, axis=1)
    
    def render_policy(self, env, episode: int = None):
        """
        渲染策略
        
        Args:
            env: 环境
            episode: episode编号
        """
        if hasattr(env, 'render'):
            policy = self.get_policy()
            
            # 创建策略映射
            action_names = ['↑', '→', '↓', '←']  # 上右下左
            
            print(f"\n策略可视化 (Episode {episode if episode else 'Final'}):")
            print("=" * 50)
            
            for row in range(env.size):
                for col in range(env.size):
                    state = (row, col)
                    if hasattr(env, 'get_state_index'):
                        state_idx = env.get_state_index(state)
                    else:
                        state_idx = state
                    
                    if state == env.start:
                        print(" S ", end="")
                    elif state == env.goal:
                        print(" G ", end="")
                    elif state in env.obstacles:
                        print(" X ", end="")
                    else:
                        action = policy[state_idx]
                        print(f" {action_names[action]} ", end="")
                print()
            
            print("\n图例: S=起始, G=目标, X=障碍物, 箭头=最优动作")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        绘制训练历史
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # 步数曲线
        axes[0, 1].plot(self.training_history['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Epsilon曲线
        axes[1, 0].plot(self.training_history['epsilon_history'])
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # 平均奖励曲线
        window_size = 100
        if len(self.training_history['episode_rewards']) >= window_size:
            rewards = np.array(self.training_history['episode_rewards'])
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f'Moving Average Reward (window={window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """保存模型"""
        np.save(filepath, self.Q)
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.Q = np.load(filepath)
    
    def evaluate(self, env, num_episodes: int = 100, max_steps: int = 1000) -> Dict[str, float]:
        """
        评估算法性能
        
        Args:
            env: 环境
            num_episodes: 评估episode数量
            max_steps: 最大步数
            
        Returns:
            评估结果
        """
        rewards = []
        steps = []
        successes = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            if hasattr(env, 'get_state_index'):
                state_idx = env.get_state_index(state)
            else:
                state_idx = state
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                # 获取有效动作
                if hasattr(env, 'get_valid_actions'):
                    valid_actions = env.get_valid_actions(state)
                else:
                    valid_actions = None
                
                # 使用贪婪策略（无探索）
                if valid_actions:
                    q_values = self.Q[state_idx][valid_actions]
                    best_action_idx = np.argmax(q_values)
                    action = valid_actions[best_action_idx]
                else:
                    action = np.argmax(self.Q[state_idx])
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                if hasattr(env, 'get_state_index'):
                    state_idx = env.get_state_index(next_state)
                else:
                    state_idx = next_state
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done:
                    if state == env.goal:
                        successes += 1
                    break
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps),
            'success_rate': successes / num_episodes,
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }


def test_q_learning():
    """测试Q-Learning算法"""
    from environments.grid_world import create_simple_grid_world
    
    # 创建环境
    env = create_simple_grid_world()
    
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
    
    print("Q-Learning算法测试")
    print("=" * 50)
    print(f"状态空间: {env.state_space}")
    print(f"动作空间: {env.action_space}")
    print(f"学习率: {q_learning.learning_rate}")
    print(f"折扣因子: {q_learning.discount_factor}")
    print(f"初始Epsilon: {q_learning.epsilon}")
    
    # 训练
    print("\n开始训练...")
    history = q_learning.train(env, num_episodes=500, verbose=True)
    
    # 评估
    print("\n评估结果:")
    eval_results = q_learning.evaluate(env, num_episodes=100)
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # 显示最终策略
    q_learning.render_policy(env, episode="Final")
    
    # 绘制训练历史
    q_learning.plot_training_history()
    
    return q_learning, env


if __name__ == "__main__":
    test_q_learning()
