"""
网格世界环境
这是学习强化学习的最佳起点环境
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Any, Optional
import random


class GridWorld:
    """
    网格世界环境
    
    特点:
    - 离散状态空间
    - 离散动作空间
    - 确定性转移
    - 简单奖励结构
    """
    
    def __init__(self, size: int = 5, start: Tuple[int, int] = (0, 0), 
                 goal: Tuple[int, int] = (4, 4), obstacles: list = None):
        """
        初始化网格世界
        
        Args:
            size: 网格大小 (size x size)
            start: 起始位置 (row, col)
            goal: 目标位置 (row, col)
            obstacles: 障碍物位置列表 [(row, col), ...]
        """
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or [(2, 2), (2, 3), (3, 2)]
        
        # 动作空间: 上(0), 右(1), 下(2), 左(3)
        self.action_space = 4
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
        
        # 状态空间
        self.state_space = size * size
        
        # 当前状态
        self.current_state = start
        
        # 奖励设置
        self.goal_reward = 1.0
        self.obstacle_reward = -1.0
        self.step_reward = -0.01
        
        # 是否结束
        self.done = False
        
        # 步数统计
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self) -> Tuple[int, int]:
        """重置环境"""
        self.current_state = self.start
        self.done = False
        self.step_count = 0
        return self.current_state
    
    def  step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作 (0: 上, 1: 右, 2: 下, 3: 左)
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.done:
            return self.current_state, 0.0, True, {}
        
        # 获取动作对应的移动
        dr, dc = self.actions[action]
        new_row = self.current_state[0] + dr
        new_col = self.current_state[1] + dc
        
        # 检查边界
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            # 撞墙，保持原位置
            reward = self.step_reward
        else:
            new_state = (new_row, new_col)
            
            # 检查是否撞到障碍物
            if new_state in self.obstacles:
                reward = self.obstacle_reward
            else:
                self.current_state = new_state
                
                # 检查是否到达目标
                if self.current_state == self.goal:
                    reward = self.goal_reward
                    self.done = True
                else:
                    reward = self.step_reward
        
        self.step_count += 1
        
        # 检查是否超时
        if self.step_count >= self.max_steps:
            self.done = True
        
        info = {
            'step_count': self.step_count,
            'action': action
        }
        
        return self.current_state, reward, self.done, info
    
    def get_state_index(self, state: Tuple[int, int]) -> int:
        """将状态坐标转换为索引"""
        return state[0] * self.size + state[1]
    
    def get_state_coord(self, index: int) -> Tuple[int, int]:
        """将状态索引转换为坐标"""
        return index // self.size, index % self.size
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """检查状态是否有效"""
        row, col = state
        return (0 <= row < self.size and 
                0 <= col < self.size and 
                state not in self.obstacles)
    
    def get_valid_actions(self, state: Tuple[int, int]) -> list:
        """获取在给定状态下的有效动作"""
        valid_actions = []
        for action, (dr, dc) in enumerate(self.actions):
            new_row = state[0] + dr
            new_col = state[1] + dc
            new_state = (new_row, new_col)
            if self.is_valid_state(new_state):
                valid_actions.append(action)
        return valid_actions
    
    def render(self, mode: str = 'human', highlight_state: Optional[Tuple[int, int]] = None):
        """
        渲染网格世界
        
        Args:
            mode: 渲染模式 ('human' 或 'rgb_array')
            highlight_state: 要高亮显示的状态
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        for i in range(self.size + 1):
            ax.axhline(y=i, color='black', linewidth=1)
            ax.axvline(x=i, color='black', linewidth=1)
        
        # 绘制单元格
        for row in range(self.size):
            for col in range(self.size):
                state = (row, col)
                x, y = col, self.size - 1 - row  # 翻转y轴
                
                if state == self.start:
                    # 起始位置 - 绿色
                    rect = patches.Rectangle((x, y), 1, 1, 
                                           facecolor='lightgreen', 
                                           edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, 'S', ha='center', va='center', 
                           fontsize=16, fontweight='bold')
                
                elif state == self.goal:
                    # 目标位置 - 蓝色
                    rect = patches.Rectangle((x, y), 1, 1, 
                                           facecolor='lightblue', 
                                           edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, 'G', ha='center', va='center', 
                           fontsize=16, fontweight='bold')
                
                elif state in self.obstacles:
                    # 障碍物 - 红色
                    rect = patches.Rectangle((x, y), 1, 1, 
                                           facecolor='red', 
                                           edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, 'X', ha='center', va='center', 
                           fontsize=16, fontweight='bold')
                
                elif state == highlight_state:
                    # 高亮状态 - 黄色
                    rect = patches.Rectangle((x, y), 1, 1, 
                                           facecolor='yellow', 
                                           edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, 'A', ha='center', va='center', 
                           fontsize=16, fontweight='bold')
                
                else:
                    # 普通单元格 - 白色
                    rect = patches.Rectangle((x, y), 1, 1, 
                                           facecolor='white', 
                                           edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
        
        # 设置坐标轴
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.set_title('Grid World Environment', fontsize=16, fontweight='bold')
        
        # 添加图例
        legend_elements = [
            patches.Patch(color='lightgreen', label='Start (S)'),
            patches.Patch(color='lightblue', label='Goal (G)'),
            patches.Patch(color='red', label='Obstacle (X)'),
            patches.Patch(color='yellow', label='Agent (A)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if mode == 'human':
            plt.show()
        else:
            # 返回RGB数组
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return img
    
    def get_transition_probabilities(self) -> Dict[Tuple[int, int, int], float]:
        """
        获取转移概率（确定性环境）
        
        Returns:
            转移概率字典 {(state, action, next_state): probability}
        """
        transitions = {}
        
        for row in range(self.size):
            for col in range(self.size):
                state = (row, col)
                
                for action in range(self.action_space):
                    dr, dc = self.actions[action]
                    new_row = row + dr
                    new_col = col + dc
                    new_state = (new_row, new_col)
                    
                    # 检查边界和障碍物
                    if (0 <= new_row < self.size and 
                        0 <= new_col < self.size and 
                        new_state not in self.obstacles):
                        # 成功转移
                        transitions[(state, action, new_state)] = 1.0
                    else:
                        # 撞墙或撞障碍物，保持原位置
                        transitions[(state, action, state)] = 1.0
        
        return transitions
    
    def get_reward_function(self) -> Dict[Tuple[int, int, int, Tuple[int, int]], float]:
        """
        获取奖励函数
        
        Returns:
            奖励函数字典 {(state, action, next_state): reward}
        """
        rewards = {}
        transitions = self.get_transition_probabilities()
        
        for (state, action, next_state), prob in transitions.items():
            if prob > 0:
                if next_state == self.goal:
                    reward = self.goal_reward
                elif next_state in self.obstacles:
                    reward = self.obstacle_reward
                else:
                    reward = self.step_reward
                
                rewards[(state, action, next_state)] = reward
        
        return rewards


def create_simple_grid_world() -> GridWorld:
    """创建一个简单的网格世界用于测试"""
    return GridWorld(size=4, start=(0, 0), goal=(3, 3), 
                    obstacles=[(1, 1), (2, 2)])


def create_complex_grid_world() -> GridWorld:
    """创建一个复杂的网格世界用于挑战"""
    return GridWorld(size=8, start=(0, 0), goal=(7, 7), 
                    obstacles=[(1, 1), (1, 2), (2, 1), (3, 3), (4, 4), (5, 5), (6, 6)])


if __name__ == "__main__":
    # 测试网格世界环境
    env = create_simple_grid_world()
    
    print("网格世界环境测试")
    print("=" * 50)
    print(f"网格大小: {env.size}x{env.size}")
    print(f"起始位置: {env.start}")
    print(f"目标位置: {env.goal}")
    print(f"障碍物: {env.obstacles}")
    print(f"动作空间: {env.action_space}")
    print(f"状态空间: {env.state_space}")
    
    # 渲染环境
    env.render()
    
    # 测试几步动作
    print("\n测试动作序列: 右->右->下->下")
    state = env.reset()
    print(f"初始状态: {state}")
    
    actions = [1, 1, 2, 2]  # 右右下下
    for i, action in enumerate(actions):
        state, reward, done, info = env.step(action)
        print(f"步骤 {i+1}: 动作={action}, 状态={state}, 奖励={reward:.3f}, 结束={done}")
        
        if done:
            break
    
    print(f"\n最终状态: {state}")
    print(f"是否到达目标: {state == env.goal}")
