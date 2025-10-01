# 📚 Jupyter笔记本使用指南

## 🚀 快速开始

### 1. 启动Jupyter
```bash
# 在项目根目录下启动
cd /Users/peixingxin/Code/DL-journey/RL-journey
source rlhf_env/bin/activate
jupyter lab
```

### 2. 打开笔记本
在Jupyter Lab中打开 `01_rl_basics.ipynb`

### 3. 运行第一个Cell
运行第一个代码cell来设置环境和导入模块。这个cell会自动：
- 设置正确的项目路径
- 导入所有必要的模块
- 配置中文字体显示

## 🔧 导入问题解决

如果遇到 `ModuleNotFoundError: No module named 'environments'` 错误：

### 方法1: 重启内核
1. 在Jupyter中点击 `Kernel` → `Restart Kernel`
2. 重新运行第一个cell

### 方法2: 检查工作目录
确保Jupyter的工作目录是项目根目录：
```python
import os
print("当前工作目录:", os.getcwd())
# 应该显示: /Users/peixingxin/Code/DL-journey/RL-journey
```

### 方法3: 手动设置路径
如果自动路径设置失败，可以手动设置：
```python
import sys
import os
sys.path.insert(0, '/Users/peixingxin/Code/DL-journey/RL-journey')
```

## 📝 笔记本内容

### 01_rl_basics.ipynb
- **目标**: 学习强化学习基础概念
- **内容**: 
  - 环境设置和模块导入
  - 网格世界环境介绍
  - 基本概念演示
  - 交互式练习

## 🎯 学习建议

1. **按顺序运行**: 按照cell的顺序依次运行
2. **理解输出**: 仔细阅读每个cell的输出信息
3. **修改参数**: 尝试修改代码中的参数，观察结果变化
4. **做笔记**: 在markdown cell中记录你的学习心得

## 🆘 常见问题

### Q: 中文显示为方框？
A: 运行第一个cell会自动设置中文字体。如果仍有问题，检查系统是否安装了中文字体。

### Q: 导入模块失败？
A: 确保Jupyter的工作目录是项目根目录，然后重启内核。

### Q: 图表不显示？
A: 确保运行了 `%matplotlib inline` 或 `plt.show()`

## 📞 获取帮助

如果遇到问题：
1. 查看控制台输出信息
2. 检查错误消息
3. 参考项目根目录的 `CHINESE_FONT_GUIDE.md`
4. 运行 `python error_check.py` 进行诊断

---

**开始你的强化学习之旅吧！** 🚀
