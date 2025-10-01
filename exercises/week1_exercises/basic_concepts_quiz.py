#!/usr/bin/env python3
"""
第1周练习: 强化学习基础概念测验
通过这个测验来检验你对RL基础概念的理解
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.grid_world import create_simple_grid_world
import numpy as np


class RLBasicConceptsQuiz:
    """
    强化学习基础概念测验
    """
    
    def __init__(self):
        """初始化测验"""
        self.env = create_simple_grid_world()
        self.score = 0
        self.total_questions = 10
        self.questions = self._create_questions()
    
    def _create_questions(self):
        """创建测验题目"""
        return [
            {
                "question": "强化学习中的智能体(Agent)是什么？",
                "options": [
                    "A. 环境的一部分",
                    "B. 学习和做决策的主体",
                    "C. 奖励函数",
                    "D. 状态空间"
                ],
                "correct": "B",
                "explanation": "智能体是强化学习中的学习和决策主体，它通过与环境交互来学习最优策略。"
            },
            {
                "question": "在网格世界中，状态空间的大小是多少？",
                "options": [
                    f"A. {self.env.size}",
                    f"B. {self.env.size * 2}",
                    f"C. {self.env.state_space}",
                    f"D. {self.env.action_space}"
                ],
                "correct": "C",
                "explanation": f"状态空间大小是网格的总格子数，即 {self.env.size}×{self.env.size} = {self.env.state_space}。"
            },
            {
                "question": "马尔可夫性质的含义是什么？",
                "options": [
                    "A. 未来状态依赖于所有历史状态",
                    "B. 未来状态只依赖于当前状态",
                    "C. 状态转移是确定性的",
                    "D. 奖励函数是线性的"
                ],
                "correct": "B",
                "explanation": "马尔可夫性质是指未来状态只依赖于当前状态，而不依赖于历史状态。"
            },
            {
                "question": "在网格世界中，动作空间包含多少个动作？",
                "options": [
                    f"A. {self.env.action_space}",
                    f"B. {self.env.state_space}",
                    f"C. {self.env.size}",
                    f"D. 2"
                ],
                "correct": "A",
                "explanation": f"动作空间包含4个动作：上、右、下、左，所以大小是 {self.env.action_space}。"
            },
            {
                "question": "价值函数V(s)表示什么？",
                "options": [
                    "A. 状态s的即时奖励",
                    "B. 从状态s开始的期望累积奖励",
                    "C. 状态s的转移概率",
                    "D. 状态s的动作数量"
                ],
                "correct": "B",
                "explanation": "价值函数V(s)表示从状态s开始，遵循某个策略的期望累积奖励。"
            },
            {
                "question": "ε-贪婪策略中，ε=0.1意味着什么？",
                "options": [
                    "A. 10%的时间进行探索，90%的时间进行利用",
                    "B. 90%的时间进行探索，10%的时间进行利用",
                    "C. 总是选择最优动作",
                    "D. 总是随机选择动作"
                ],
                "correct": "A",
                "explanation": "ε=0.1意味着10%的时间随机选择动作（探索），90%的时间选择最优动作（利用）。"
            },
            {
                "question": "折扣因子γ的作用是什么？",
                "options": [
                    "A. 增加即时奖励的重要性",
                    "B. 减少未来奖励的重要性",
                    "C. 平衡即时奖励和未来奖励",
                    "D. 决定动作的选择"
                ],
                "correct": "C",
                "explanation": "折扣因子γ用于平衡即时奖励和未来奖励的重要性，通常0<γ<1。"
            },
            {
                "question": "在网格世界中，目标状态的奖励是多少？",
                "options": [
                    f"A. {self.env.goal_reward}",
                    f"B. {self.env.obstacle_reward}",
                    f"C. {self.env.step_reward}",
                    f"D. 0"
                ],
                "correct": "A",
                "explanation": f"目标状态的奖励是 {self.env.goal_reward}，这是一个正奖励。"
            },
            {
                "question": "贝尔曼方程描述了什么关系？",
                "options": [
                    "A. 状态和动作的关系",
                    "B. 价值函数之间的关系",
                    "C. 奖励和惩罚的关系",
                    "D. 探索和利用的关系"
                ],
                "correct": "B",
                "explanation": "贝尔曼方程描述了价值函数之间的关系，是强化学习的核心方程。"
            },
            {
                "question": "策略π(a|s)表示什么？",
                "options": [
                    "A. 状态s的价值",
                    "B. 动作a的价值",
                    "C. 在状态s下选择动作a的概率",
                    "D. 状态s的转移概率"
                ],
                "correct": "C",
                "explanation": "策略π(a|s)表示在状态s下选择动作a的概率，定义了智能体的行为规则。"
            }
        ]
    
    def run_quiz(self):
        """运行测验"""
        print("🧠 强化学习基础概念测验")
        print("=" * 50)
        print(f"总题数: {self.total_questions}")
        print("每题10分，满分100分")
        print("=" * 50)
        
        for i, q in enumerate(self.questions, 1):
            print(f"\n📝 第{i}题:")
            print(q["question"])
            print()
            
            for option in q["options"]:
                print(f"  {option}")
            
            while True:
                answer = input("\n请输入你的答案 (A/B/C/D): ").upper().strip()
                if answer in ['A', 'B', 'C', 'D']:
                    break
                print("❌ 请输入有效的选项 (A/B/C/D)")
            
            if answer == q["correct"]:
                print("✅ 正确！")
                self.score += 10
            else:
                print(f"❌ 错误！正确答案是 {q['correct']}")
            
            print(f"💡 解释: {q['explanation']}")
            print("-" * 50)
        
        self._show_results()
    
    def _show_results(self):
        """显示测验结果"""
        print("\n" + "=" * 50)
        print("📊 测验结果")
        print("=" * 50)
        print(f"总分: {self.score}/{self.total_questions * 10}")
        print(f"正确率: {self.score / (self.total_questions * 10) * 100:.1f}%")
        
        if self.score >= 90:
            print("🎉 优秀！你对RL基础概念有很好的理解！")
            grade = "A"
        elif self.score >= 80:
            print("👍 良好！你对RL基础概念有较好的理解！")
            grade = "B"
        elif self.score >= 70:
            print("👌 及格！建议复习一下基础概念。")
            grade = "C"
        else:
            print("📚 需要加强学习！建议重新学习RL基础概念。")
            grade = "D"
        
        print(f"等级: {grade}")
        
        # 给出学习建议
        print("\n💡 学习建议:")
        if self.score < 70:
            print("- 重新阅读《强化学习导论》第1-3章")
            print("- 观看David Silver的RL课程第1-2讲")
            print("- 完成网格世界环境的分析练习")
        elif self.score < 90:
            print("- 复习MDP和贝尔曼方程")
            print("- 完成价值函数计算练习")
            print("- 开始学习Q-Learning算法")
        else:
            print("- 你已经掌握了RL基础概念！")
            print("- 可以开始学习Q-Learning算法")
            print("- 尝试实现简单的RL算法")
        
        return grade
    
    def practice_mode(self):
        """练习模式 - 可以重复练习"""
        print("🔄 练习模式")
        print("=" * 50)
        print("在这个模式下，你可以重复练习题目")
        print("输入 'quit' 退出练习模式")
        print("=" * 50)
        
        while True:
            print("\n选择练习类型:")
            print("1. 随机题目练习")
            print("2. 特定题目练习")
            print("3. 退出练习模式")
            
            choice = input("请选择 (1/2/3): ").strip()
            
            if choice == "1":
                self._random_practice()
            elif choice == "2":
                self._specific_practice()
            elif choice == "3":
                break
            else:
                print("❌ 请输入有效选项")
    
    def _random_practice(self):
        """随机题目练习"""
        import random
        
        print("\n🎲 随机题目练习")
        print("-" * 30)
        
        # 随机选择5道题
        random_questions = random.sample(self.questions, min(5, len(self.questions)))
        
        for i, q in enumerate(random_questions, 1):
            print(f"\n📝 第{i}题:")
            print(q["question"])
            print()
            
            for option in q["options"]:
                print(f"  {option}")
            
            while True:
                answer = input("\n请输入你的答案 (A/B/C/D): ").upper().strip()
                if answer in ['A', 'B', 'C', 'D']:
                    break
                print("❌ 请输入有效的选项 (A/B/C/D)")
            
            if answer == q["correct"]:
                print("✅ 正确！")
            else:
                print(f"❌ 错误！正确答案是 {q['correct']}")
            
            print(f"💡 解释: {q['explanation']}")
            print("-" * 30)
    
    def _specific_practice(self):
        """特定题目练习"""
        print("\n🎯 特定题目练习")
        print("-" * 30)
        
        # 显示所有题目
        for i, q in enumerate(self.questions, 1):
            print(f"{i}. {q['question'][:50]}...")
        
        while True:
            try:
                choice = int(input(f"\n请选择题目编号 (1-{len(self.questions)}): "))
                if 1 <= choice <= len(self.questions):
                    break
                else:
                    print(f"❌ 请输入1-{len(self.questions)}之间的数字")
            except ValueError:
                print("❌ 请输入有效的数字")
        
        q = self.questions[choice - 1]
        print(f"\n📝 第{choice}题:")
        print(q["question"])
        print()
        
        for option in q["options"]:
            print(f"  {option}")
        
        while True:
            answer = input("\n请输入你的答案 (A/B/C/D): ").upper().strip()
            if answer in ['A', 'B', 'C', 'D']:
                break
            print("❌ 请输入有效的选项 (A/B/C/D)")
        
        if answer == q["correct"]:
            print("✅ 正确！")
        else:
            print(f"❌ 错误！正确答案是 {q['correct']}")
        
        print(f"💡 解释: {q['explanation']}")


def main():
    """主函数"""
    quiz = RLBasicConceptsQuiz()
    
    print("🎯 强化学习基础概念测验系统")
    print("=" * 50)
    print("1. 正式测验")
    print("2. 练习模式")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择模式 (1/2/3): ").strip()
        
        if choice == "1":
            quiz.run_quiz()
            break
        elif choice == "2":
            quiz.practice_mode()
        elif choice == "3":
            print("👋 再见！")
            break
        else:
            print("❌ 请输入有效选项 (1/2/3)")


if __name__ == "__main__":
    main()
