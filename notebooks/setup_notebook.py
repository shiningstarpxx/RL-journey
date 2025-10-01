#!/usr/bin/env python3
"""
Jupyter笔记本环境设置脚本
解决导入路径问题
"""

import sys
import os

def setup_notebook_environment():
    """设置Jupyter笔记本环境"""
    print("🔧 设置Jupyter笔记本环境...")
    
    # 获取项目根目录
    current_dir = os.getcwd()
    if 'notebooks' in current_dir:
        # 如果在notebooks目录中，向上两级到项目根目录
        project_root = os.path.dirname(os.path.dirname(current_dir))
    else:
        # 如果在项目根目录中
        project_root = current_dir
    
    print(f"📁 当前目录: {current_dir}")
    print(f"📁 项目根目录: {project_root}")
    
    # 添加项目根目录到Python路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ 已添加项目根目录到Python路径: {project_root}")
    
    # 验证关键模块
    modules_to_check = [
        'environments.grid_world',
        'algorithms.tabular.q_learning',
        'algorithms.tabular.sarsa',
        'progress.learning_tracker',
        'utils.font_config'
    ]
    
    print("\n🔍 检查模块导入...")
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module}: 导入成功")
        except ImportError as e:
            print(f"❌ {module}: 导入失败 - {e}")
    
    # 设置中文字体
    try:
        from utils.font_config import setup_chinese_font
        font = setup_chinese_font()
        print(f"\n✅ 中文字体设置成功: {font}")
    except ImportError:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("\n⚠️ 使用默认中文字体设置")
    
    print("\n🎉 环境设置完成！")
    return project_root

if __name__ == "__main__":
    setup_notebook_environment()
