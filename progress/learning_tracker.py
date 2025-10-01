#!/usr/bin/env python3
"""
强化学习学习进度跟踪器
用于跟踪学习进度、记录成就和管理学习计划
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.font_config import setup_chinese_font
    setup_chinese_font()
except ImportError:
    # 如果无法导入字体配置，使用默认设置
    plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


class LearningTracker:
    """
    学习进度跟踪器
    
    功能:
    - 跟踪学习进度
    - 记录学习成就
    - 管理学习计划
    - 生成学习报告
    """
    
    def __init__(self, data_file: str = "progress/learning_data.json"):
        """
        初始化学习跟踪器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.data = self._load_data()
        
        # 学习路径定义
        self.learning_path = {
            "week1": {
                "name": "强化学习基础",
                "topics": [
                    "RL基本概念",
                    "MDP理论",
                    "价值函数",
                    "探索与利用",
                    "贝尔曼方程"
                ],
                "exercises": [
                    "基础概念测验",
                    "环境理解练习",
                    "价值函数计算",
                    "探索策略分析"
                ],
                "projects": [
                    "网格世界环境分析"
                ]
            },
            "week2": {
                "name": "Q-Learning算法",
                "topics": [
                    "Q-Learning原理",
                    "时序差分学习",
                    "ε-贪婪策略",
                    "参数调优",
                    "收敛性分析"
                ],
                "exercises": [
                    "Q-Learning实现",
                    "参数实验",
                    "复杂环境测试",
                    "奖励崩溃分析"
                ],
                "projects": [
                    "Q-Learning vs SARSA对比"
                ]
            },
            "week3": {
                "name": "策略梯度方法",
                "topics": [
                    "策略梯度定理",
                    "REINFORCE算法",
                    "基线方法",
                    "连续动作空间",
                    "方差分析"
                ],
                "exercises": [
                    "REINFORCE实现",
                    "CartPole环境测试",
                    "策略网络设计",
                    "梯度估计分析"
                ],
                "projects": [
                    "策略梯度算法比较"
                ]
            },
            "week4": {
                "name": "Actor-Critic方法",
                "topics": [
                    "Actor-Critic架构",
                    "A2C算法",
                    "优势估计",
                    "训练稳定性",
                    "超参数调优"
                ],
                "exercises": [
                    "A2C实现",
                    "优势估计比较",
                    "连续控制测试",
                    "性能分析"
                ],
                "projects": [
                    "Actor-Critic算法优化"
                ]
            },
            "week5_6": {
                "name": "深度强化学习",
                "topics": [
                    "DQN算法",
                    "经验回放",
                    "目标网络",
                    "Double DQN",
                    "Dueling DQN"
                ],
                "exercises": [
                    "DQN实现",
                    "Atari游戏测试",
                    "改进方法实现",
                    "性能比较"
                ],
                "projects": [
                    "深度RL算法集成"
                ]
            },
            "week7_8": {
                "name": "现代算法",
                "topics": [
                    "PPO算法",
                    "SAC算法",
                    "最大熵RL",
                    "信任区域方法",
                    "算法比较"
                ],
                "exercises": [
                    "PPO实现",
                    "SAC实现",
                    "复杂环境测试",
                    "算法分析"
                ],
                "projects": [
                    "最终项目"
                ]
            }
        }
    
    def _load_data(self) -> Dict:
        """加载学习数据"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "start_date": datetime.now().isoformat(),
                "current_week": 1,
                "progress": {},
                "achievements": [],
                "study_sessions": [],
                "notes": []
            }
    
    def _save_data(self):
        """保存学习数据"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def start_week(self, week: int):
        """开始新一周的学习"""
        week_key = f"week{week}" if week <= 4 else f"week{week}_{week+1}" if week <= 6 else f"week{week}_{week+1}"
        
        if week_key not in self.learning_path:
            print(f"❌ 无效的周数: {week}")
            return
        
        self.data["current_week"] = week
        self.data["progress"][week_key] = {
            "start_date": datetime.now().isoformat(),
            "topics_completed": [],
            "exercises_completed": [],
            "projects_completed": [],
            "study_time": 0,
            "notes": []
        }
        self._save_data()
        
        print(f"🎯 开始第{week}周学习: {self.learning_path[week_key]['name']}")
        self._print_week_plan(week_key)
    
    def complete_topic(self, week: int, topic: str):
        """完成一个学习主题"""
        week_key = f"week{week}" if week <= 4 else f"week{week}_{week+1}" if week <= 6 else f"week{week}_{week+1}"
        
        if week_key not in self.data["progress"]:
            self.start_week(week)
        
        if topic not in self.data["progress"][week_key]["topics_completed"]:
            self.data["progress"][week_key]["topics_completed"].append(topic)
            self._save_data()
            print(f"✅ 完成主题: {topic}")
            self._check_achievements()
        else:
            print(f"ℹ️ 主题已完成: {topic}")
    
    def complete_exercise(self, week: int, exercise: str):
        """完成一个练习"""
        week_key = f"week{week}" if week <= 4 else f"week{week}_{week+1}" if week <= 6 else f"week{week}_{week+1}"
        
        if week_key not in self.data["progress"]:
            self.start_week(week)
        
        if exercise not in self.data["progress"][week_key]["exercises_completed"]:
            self.data["progress"][week_key]["exercises_completed"].append(exercise)
            self._save_data()
            print(f"✅ 完成练习: {exercise}")
            self._check_achievements()
        else:
            print(f"ℹ️ 练习已完成: {exercise}")
    
    def complete_project(self, week: int, project: str):
        """完成一个项目"""
        week_key = f"week{week}" if week <= 4 else f"week{week}_{week+1}" if week <= 6 else f"week{week}_{week+1}"
        
        if week_key not in self.data["progress"]:
            self.start_week(week)
        
        if project not in self.data["progress"][week_key]["projects_completed"]:
            self.data["progress"][week_key]["projects_completed"].append(project)
            self._save_data()
            print(f"🎉 完成项目: {project}")
            self._check_achievements()
        else:
            print(f"ℹ️ 项目已完成: {project}")
    
    def add_study_session(self, week: int, duration: float, activity: str, notes: str = ""):
        """添加学习会话"""
        session = {
            "date": datetime.now().isoformat(),
            "week": week,
            "duration": duration,
            "activity": activity,
            "notes": notes
        }
        
        self.data["study_sessions"].append(session)
        
        # 更新周学习时间
        week_key = f"week{week}" if week <= 4 else f"week{week}_{week+1}" if week <= 6 else f"week{week}_{week+1}"
        if week_key in self.data["progress"]:
            self.data["progress"][week_key]["study_time"] += duration
        
        self._save_data()
        print(f"📚 记录学习会话: {activity} ({duration:.1f}小时)")
    
    def add_note(self, week: int, note: str):
        """添加学习笔记"""
        week_key = f"week{week}" if week <= 4 else f"week{week}_{week+1}" if week <= 6 else f"week{week}_{week+1}"
        
        if week_key not in self.data["progress"]:
            self.start_week(week)
        
        note_entry = {
            "date": datetime.now().isoformat(),
            "note": note
        }
        
        self.data["progress"][week_key]["notes"].append(note_entry)
        self._save_data()
        print(f"📝 添加笔记: {note[:50]}...")
    
    def _check_achievements(self):
        """检查成就"""
        achievements = []
        
        # 计算总体进度
        total_topics = sum(len(week_data["topics"]) for week_data in self.learning_path.values())
        total_exercises = sum(len(week_data["exercises"]) for week_data in self.learning_path.values())
        total_projects = sum(len(week_data["projects"]) for week_data in self.learning_path.values())
        
        completed_topics = sum(len(week_data.get("topics_completed", [])) 
                              for week_data in self.data["progress"].values())
        completed_exercises = sum(len(week_data.get("exercises_completed", [])) 
                                 for week_data in self.data["progress"].values())
        completed_projects = sum(len(week_data.get("projects_completed", [])) 
                                for week_data in self.data["progress"].values())
        
        # 成就检查
        if completed_topics >= total_topics * 0.1 and "first_topic" not in self.data["achievements"]:
            achievements.append("first_topic")
        
        if completed_exercises >= total_exercises * 0.1 and "first_exercise" not in self.data["achievements"]:
            achievements.append("first_exercise")
        
        if completed_projects >= total_projects * 0.1 and "first_project" not in self.data["achievements"]:
            achievements.append("first_project")
        
        if completed_topics >= total_topics * 0.5 and "half_topics" not in self.data["achievements"]:
            achievements.append("half_topics")
        
        if completed_exercises >= total_exercises * 0.5 and "half_exercises" not in self.data["achievements"]:
            achievements.append("half_exercises")
        
        if completed_topics >= total_topics and "all_topics" not in self.data["achievements"]:
            achievements.append("all_topics")
        
        if completed_exercises >= total_exercises and "all_exercises" not in self.data["achievements"]:
            achievements.append("all_exercises")
        
        if completed_projects >= total_projects and "all_projects" not in self.data["achievements"]:
            achievements.append("all_projects")
        
        # 添加新成就
        for achievement in achievements:
            if achievement not in self.data["achievements"]:
                self.data["achievements"].append(achievement)
                self._show_achievement(achievement)
        
        if achievements:
            self._save_data()
    
    def _show_achievement(self, achievement: str):
        """显示成就"""
        achievement_messages = {
            "first_topic": "🎉 恭喜！你完成了第一个学习主题！",
            "first_exercise": "🎉 恭喜！你完成了第一个练习！",
            "first_project": "🎉 恭喜！你完成了第一个项目！",
            "half_topics": "🎉 恭喜！你完成了50%的学习主题！",
            "half_exercises": "🎉 恭喜！你完成了50%的练习！",
            "all_topics": "🎉 恭喜！你完成了所有学习主题！",
            "all_exercises": "🎉 恭喜！你完成了所有练习！",
            "all_projects": "🎉 恭喜！你完成了所有项目！"
        }
        
        print(achievement_messages.get(achievement, f"🎉 新成就: {achievement}"))
    
    def _print_week_plan(self, week_key: str):
        """打印周学习计划"""
        week_data = self.learning_path[week_key]
        print(f"\n📋 第{week_key}周学习计划:")
        print(f"主题: {week_data['name']}")
        
        print("\n📚 学习主题:")
        for i, topic in enumerate(week_data["topics"], 1):
            print(f"  {i}. {topic}")
        
        print("\n🛠️ 练习任务:")
        for i, exercise in enumerate(week_data["exercises"], 1):
            print(f"  {i}. {exercise}")
        
        print("\n🎯 项目任务:")
        for i, project in enumerate(week_data["projects"], 1):
            print(f"  {i}. {project}")
    
    def show_progress(self):
        """显示学习进度"""
        print("📊 学习进度总览")
        print("=" * 50)
        
        # 总体统计
        total_topics = sum(len(week_data["topics"]) for week_data in self.learning_path.values())
        total_exercises = sum(len(week_data["exercises"]) for week_data in self.learning_path.values())
        total_projects = sum(len(week_data["projects"]) for week_data in self.learning_path.values())
        
        completed_topics = sum(len(week_data.get("topics_completed", [])) 
                              for week_data in self.data["progress"].values())
        completed_exercises = sum(len(week_data.get("exercises_completed", [])) 
                                 for week_data in self.data["progress"].values())
        completed_projects = sum(len(week_data.get("projects_completed", [])) 
                                for week_data in self.data["progress"].values())
        
        print(f"📚 学习主题: {completed_topics}/{total_topics} ({completed_topics/total_topics*100:.1f}%)")
        print(f"🛠️ 练习任务: {completed_exercises}/{total_exercises} ({completed_exercises/total_exercises*100:.1f}%)")
        print(f"🎯 项目任务: {completed_projects}/{total_projects} ({completed_projects/total_projects*100:.1f}%)")
        
        # 周进度
        print(f"\n📅 当前周: 第{self.data['current_week']}周")
        
        for week_key, week_data in self.learning_path.items():
            if week_key in self.data["progress"]:
                progress = self.data["progress"][week_key]
                topics_progress = len(progress.get("topics_completed", [])) / len(week_data["topics"]) * 100
                exercises_progress = len(progress.get("exercises_completed", [])) / len(week_data["exercises"]) * 100
                projects_progress = len(progress.get("projects_completed", [])) / len(week_data["projects"]) * 100
                
                print(f"\n{week_key}: {week_data['name']}")
                print(f"  主题: {topics_progress:.1f}%")
                print(f"  练习: {exercises_progress:.1f}%")
                print(f"  项目: {projects_progress:.1f}%")
        
        # 成就
        print(f"\n🏆 成就: {len(self.data['achievements'])}个")
        for achievement in self.data["achievements"]:
            print(f"  ✅ {achievement}")
        
        # 学习时间
        total_study_time = sum(session["duration"] for session in self.data["study_sessions"])
        print(f"\n⏰ 总学习时间: {total_study_time:.1f}小时")
    
    def plot_progress(self, save_path: Optional[str] = None):
        """绘制学习进度图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总体进度
        total_topics = sum(len(week_data["topics"]) for week_data in self.learning_path.values())
        total_exercises = sum(len(week_data["exercises"]) for week_data in self.learning_path.values())
        total_projects = sum(len(week_data["projects"]) for week_data in self.learning_path.values())
        
        completed_topics = sum(len(week_data.get("topics_completed", [])) 
                              for week_data in self.data["progress"].values())
        completed_exercises = sum(len(week_data.get("exercises_completed", [])) 
                                 for week_data in self.data["progress"].values())
        completed_projects = sum(len(week_data.get("projects_completed", [])) 
                                for week_data in self.data["progress"].values())
        
        categories = ['主题', '练习', '项目']
        completed = [completed_topics, completed_exercises, completed_projects]
        total = [total_topics, total_exercises, total_projects]
        
        axes[0, 0].bar(categories, [c/t*100 for c, t in zip(completed, total)], 
                       color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('总体学习进度')
        axes[0, 0].set_ylabel('完成百分比 (%)')
        axes[0, 0].set_ylim(0, 100)
        
        # 周进度
        weeks = list(self.learning_path.keys())
        week_progress = []
        
        for week_key in weeks:
            if week_key in self.data["progress"]:
                progress = self.data["progress"][week_key]
                topics_progress = len(progress.get("topics_completed", [])) / len(self.learning_path[week_key]["topics"]) * 100
                week_progress.append(topics_progress)
            else:
                week_progress.append(0)
        
        axes[0, 1].bar(weeks, week_progress, color='lightblue')
        axes[0, 1].set_title('每周主题完成进度')
        axes[0, 1].set_ylabel('完成百分比 (%)')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 学习时间趋势
        if self.data["study_sessions"]:
            dates = [datetime.fromisoformat(session["date"]) for session in self.data["study_sessions"]]
            durations = [session["duration"] for session in self.data["study_sessions"]]
            
            # 按日期分组
            daily_time = {}
            for date, duration in zip(dates, durations):
                day = date.date()
                daily_time[day] = daily_time.get(day, 0) + duration
            
            sorted_dates = sorted(daily_time.keys())
            sorted_durations = [daily_time[date] for date in sorted_dates]
            
            axes[1, 0].plot(sorted_dates, sorted_durations, marker='o')
            axes[1, 0].set_title('每日学习时间')
            axes[1, 0].set_ylabel('学习时间 (小时)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, '暂无学习时间数据', ha='center', va='center')
            axes[1, 0].set_title('每日学习时间')
        
        # 成就进度
        achievement_categories = ['主题', '练习', '项目']
        achievement_counts = [0, 0, 0]
        
        for achievement in self.data["achievements"]:
            if "topic" in achievement:
                achievement_counts[0] += 1
            elif "exercise" in achievement:
                achievement_counts[1] += 1
            elif "project" in achievement:
                achievement_counts[2] += 1
        
        axes[1, 1].bar(achievement_categories, achievement_counts, color='gold')
        axes[1, 1].set_title('成就统计')
        axes[1, 1].set_ylabel('成就数量')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """生成学习报告"""
        report = []
        report.append("# 📊 强化学习学习报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 总体统计
        total_topics = sum(len(week_data["topics"]) for week_data in self.learning_path.values())
        total_exercises = sum(len(week_data["exercises"]) for week_data in self.learning_path.values())
        total_projects = sum(len(week_data["projects"]) for week_data in self.learning_path.values())
        
        completed_topics = sum(len(week_data.get("topics_completed", [])) 
                              for week_data in self.data["progress"].values())
        completed_exercises = sum(len(week_data.get("exercises_completed", [])) 
                                 for week_data in self.data["progress"].values())
        completed_projects = sum(len(week_data.get("projects_completed", [])) 
                                for week_data in self.data["progress"].values())
        
        report.append("## 📈 总体进度")
        report.append(f"- 学习主题: {completed_topics}/{total_topics} ({completed_topics/total_topics*100:.1f}%)")
        report.append(f"- 练习任务: {completed_exercises}/{total_exercises} ({completed_exercises/total_exercises*100:.1f}%)")
        report.append(f"- 项目任务: {completed_projects}/{total_projects} ({completed_projects/total_projects*100:.1f}%)")
        report.append("")
        
        # 周进度详情
        report.append("## 📅 每周进度详情")
        for week_key, week_data in self.learning_path.items():
            if week_key in self.data["progress"]:
                progress = self.data["progress"][week_key]
                report.append(f"### {week_key}: {week_data['name']}")
                
                # 完成的主题
                completed_topics_week = progress.get("topics_completed", [])
                report.append(f"**完成的主题 ({len(completed_topics_week)}/{len(week_data['topics'])}):**")
                for topic in completed_topics_week:
                    report.append(f"- ✅ {topic}")
                
                # 未完成的主题
                remaining_topics = [t for t in week_data["topics"] if t not in completed_topics_week]
                if remaining_topics:
                    report.append(f"**待完成的主题 ({len(remaining_topics)}):**")
                    for topic in remaining_topics:
                        report.append(f"- ⏳ {topic}")
                
                report.append("")
        
        # 成就
        report.append("## 🏆 成就列表")
        for achievement in self.data["achievements"]:
            report.append(f"- ✅ {achievement}")
        report.append("")
        
        # 学习时间统计
        total_study_time = sum(session["duration"] for session in self.data["study_sessions"])
        report.append("## ⏰ 学习时间统计")
        report.append(f"- 总学习时间: {total_study_time:.1f}小时")
        report.append(f"- 学习会话数: {len(self.data['study_sessions'])}")
        if self.data["study_sessions"]:
            avg_session_time = total_study_time / len(self.data["study_sessions"])
            report.append(f"- 平均会话时长: {avg_session_time:.1f}小时")
        report.append("")
        
        # 建议
        report.append("## 💡 学习建议")
        if completed_topics < total_topics * 0.3:
            report.append("- 建议增加理论学习时间，打好基础")
        elif completed_topics < total_topics * 0.7:
            report.append("- 建议加强实践练习，巩固理论知识")
        else:
            report.append("- 建议专注于项目实践，提升应用能力")
        
        if len(self.data["study_sessions"]) < 10:
            report.append("- 建议增加学习频率，保持学习连续性")
        
        return "\n".join(report)


def main():
    """主函数 - 演示学习跟踪器"""
    tracker = LearningTracker()
    
    print("🎯 强化学习学习跟踪器")
    print("=" * 50)
    
    # 显示当前进度
    tracker.show_progress()
    
    # 开始第1周学习
    print("\n" + "="*50)
    tracker.start_week(1)
    
    # 模拟完成一些任务
    print("\n" + "="*50)
    tracker.complete_topic(1, "RL基本概念")
    tracker.complete_topic(1, "MDP理论")
    tracker.complete_exercise(1, "基础概念测验")
    
    # 添加学习会话
    tracker.add_study_session(1, 2.5, "理论学习", "学习了RL基本概念和MDP")
    tracker.add_study_session(1, 1.5, "实践练习", "完成了基础概念测验")
    
    # 添加笔记
    tracker.add_note(1, "RL的核心是智能体与环境的交互")
    tracker.add_note(1, "MDP是RL的数学基础")
    
    # 显示更新后的进度
    print("\n" + "="*50)
    tracker.show_progress()
    
    # 绘制进度图
    print("\n" + "="*50)
    tracker.plot_progress()
    
    # 生成报告
    print("\n" + "="*50)
    report = tracker.generate_report()
    print(report)
    
    # 保存报告
    with open("progress/learning_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n📄 学习报告已保存到 progress/learning_report.md")


if __name__ == "__main__":
    main()
