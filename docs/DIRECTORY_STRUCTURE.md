# 📁 项目目录结构说明

## 🎯 项目概览

```
RL-journey/
├── 📚 docs/                    # 项目文档
├── 🧠 algorithms/              # 强化学习算法实现
├── 🌍 environments/            # 强化学习环境
├── 🧪 experiments/             # 实验脚本和结果
├── 📝 exercises/               # 练习和作业
├── 📓 notebooks/               # Jupyter笔记本
├── 📊 progress/                # 学习进度跟踪
├── 🛠️ utils/                   # 工具和配置
├── 📁 data/                    # 数据文件
├── 🎯 models/                  # 训练好的模型
├── ⚙️ configs/                 # 配置文件
├── 🎬 scripts/                 # 脚本文件
├── 🎨 assets/                  # 资源文件
├── 📖 theory/                  # 理论资料
└── 🐍 src/                     # 源代码
```

## 📚 docs/ - 项目文档

包含所有项目相关的文档：

- `CHINESE_FONT_GUIDE.md` - 中文字体配置指南
- `PROJECT_SUMMARY.md` - 项目总结
- `QUICK_START_GUIDE.md` - 快速开始指南
- `RL_LEARNING_PATH.md` - 强化学习学习路径
- `RL_vs_RLHF_Analysis.md` - RL vs RLHF 分析
- `DIRECTORY_STRUCTURE.md` - 本文件，目录结构说明

## 🧠 algorithms/ - 强化学习算法

按算法类型组织的实现：

```
algorithms/
├── tabular/                    # 表格型算法
│   ├── q_learning.py          # Q-Learning算法
│   └── sarsa.py               # SARSA算法
├── policy_gradient/            # 策略梯度算法
├── actor_critic/               # Actor-Critic算法
├── deep_rl/                    # 深度强化学习
└── modern/                     # 现代算法
```

## 🌍 environments/ - 强化学习环境

```
environments/
└── grid_world.py              # 网格世界环境
```

## 🧪 experiments/ - 实验脚本

按学习周次组织的实验：

```
experiments/
├── week1_basics/              # 第1周：基础概念实验
├── week2_q_learning/          # 第2周：Q-Learning实验
│   └── experiment_1_q_learning.py
├── week3_policy_gradient/     # 第3周：策略梯度实验
├── week4_actor_critic/        # 第4周：Actor-Critic实验
├── week5_6_dqn/               # 第5-6周：深度强化学习实验
└── week7_8_modern/            # 第7-8周：现代算法实验
```

## 📝 exercises/ - 练习和作业

按学习周次组织的练习：

```
exercises/
├── week1_exercises/           # 第1周练习
│   └── basic_concepts_quiz.py
├── week2_exercises/           # 第2周练习
├── week3_exercises/           # 第3周练习
├── week4_exercises/           # 第4周练习
├── week5_6_exercises/         # 第5-6周练习
└── week7_8_exercises/         # 第7-8周练习
```

## 📓 notebooks/ - Jupyter笔记本

```
notebooks/
├── 01_rl_basics.ipynb        # 强化学习基础概念
├── RLHF_Demo.ipynb           # RLHF演示
├── README.md                 # 笔记本使用指南
└── setup_notebook.py         # 笔记本环境设置
```

## 📊 progress/ - 学习进度跟踪

```
progress/
├── learning_tracker.py       # 学习进度跟踪器
├── learning_data.json        # 学习数据
└── learning_report.md        # 学习报告
```

## 🛠️ utils/ - 工具和配置

```
utils/
└── font_config.py            # 字体配置工具
```

## 📁 其他目录

- `data/` - 存储数据文件
- `models/` - 存储训练好的模型
- `configs/` - 配置文件
- `scripts/` - 脚本文件
- `assets/` - 资源文件（如演示脚本）
- `theory/` - 理论资料（论文、笔记、幻灯片）
- `src/` - 源代码（按功能模块组织）
- `logs/` - 日志文件
- `rlhf_env/` - Python虚拟环境

## 🎯 使用建议

### 1. 学习路径
按照 `docs/RL_LEARNING_PATH.md` 中的8周学习计划进行学习。

### 2. 实验顺序
1. 先阅读 `notebooks/01_rl_basics.ipynb` 了解基础概念
2. 运行 `experiments/week1_basics/` 中的基础实验
3. 完成 `exercises/week1_exercises/` 中的练习
4. 逐步进行后续周次的学习

### 3. 文档查阅
- 快速开始：`docs/QUICK_START_GUIDE.md`
- 学习路径：`docs/RL_LEARNING_PATH.md`
- 字体问题：`docs/CHINESE_FONT_GUIDE.md`
- 项目总结：`docs/PROJECT_SUMMARY.md`

### 4. 工具使用
- 学习跟踪：`python progress/learning_tracker.py`
- 错误检查：`python scripts/error_check.py`
- 字体测试：`python scripts/test_chinese_font.py`

## 🔧 维护说明

### 添加新算法
1. 在 `algorithms/` 下创建对应的子目录
2. 实现算法代码
3. 添加相应的实验脚本
4. 更新文档

### 添加新环境
1. 在 `environments/` 下添加环境实现
2. 确保与现有接口兼容
3. 添加测试用例

### 添加新实验
1. 在 `experiments/` 下按周次组织
2. 确保实验可重现
3. 添加结果分析

---

**保持目录结构清晰，有助于项目的长期维护和扩展！** 🚀
