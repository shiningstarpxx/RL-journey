#!/usr/bin/env python3
"""
字体配置工具
解决matplotlib中文显示乱码问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os


def setup_chinese_font():
    """
    设置中文字体，解决matplotlib中文显示乱码问题
    """
    system = platform.system()
    
    # 根据操作系统选择合适的中文字体
    if system == "Darwin":  # macOS
        chinese_fonts = [
            'PingFang SC',  # 苹方-简
            'PingFang HK',  # 苹方-繁
            'STHeiti',      # 华文黑体
            'Arial Unicode MS',  # Arial Unicode MS
            'Hiragino Sans GB',  # 冬青黑体简体中文
            'STSong',       # 华文宋体
            'STKaiti'       # 华文楷体
        ]
    elif system == "Windows":
        chinese_fonts = [
            'Microsoft YaHei',     # 微软雅黑
            'SimHei',              # 黑体
            'SimSun',              # 宋体
            'KaiTi',               # 楷体
            'Microsoft Sans Serif' # 微软无衬线字体
        ]
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # 思源黑体
            'Source Han Sans SC',   # 思源黑体
            'DejaVu Sans'           # DejaVu Sans
        ]
    
    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = None
    
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        print(f"✅ 中文字体设置成功: {selected_font}")
        return selected_font
    else:
        # 如果没有找到中文字体，使用默认设置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("⚠️ 未找到合适的中文字体，使用默认字体")
        print("建议安装中文字体包")
        return None


def test_chinese_display():
    """
    测试中文显示效果
    """
    import numpy as np
    
    # 设置中文字体
    font = setup_chinese_font()
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    x = np.arange(5)
    y = [0.8, 0.85, 0.9, 0.87, 0.92]
    labels = ['Q-Learning', 'SARSA', 'Policy Gradient', 'Actor-Critic', 'PPO']
    
    # 绘制柱状图
    bars = ax.bar(x, y, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    # 设置标题和标签
    ax.set_title('强化学习算法性能比较', fontsize=16, fontweight='bold')
    ax.set_xlabel('算法类型', fontsize=12)
    ax.set_ylabel('成功率', fontsize=12)
    
    # 设置x轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # 添加数值标签
    for bar, value in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 设置网格
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/chinese_font_test.png', dpi=300, bbox_inches='tight')
    print("📊 中文显示测试图片已保存到: logs/chinese_font_test.png")
    
    # 显示图片
    plt.show()
    
    return font


def get_font_info():
    """
    获取当前字体信息
    """
    print("📋 当前字体配置:")
    print(f"  字体族: {plt.rcParams['font.sans-serif']}")
    print(f"  负号显示: {plt.rcParams['axes.unicode_minus']}")
    
    # 获取当前使用的字体
    current_font = plt.rcParams['font.sans-serif'][0]
    print(f"  当前字体: {current_font}")


def install_chinese_fonts_macos():
    """
    在macOS上安装中文字体的说明
    """
    print("🍎 macOS中文字体安装说明:")
    print("1. 系统自带字体:")
    print("   - PingFang SC (苹方-简)")
    print("   - STHeiti (华文黑体)")
    print("   - Arial Unicode MS")
    print()
    print("2. 如果字体显示异常，可以尝试:")
    print("   - 重启Python环境")
    print("   - 清除matplotlib缓存: rm -rf ~/.matplotlib")
    print("   - 重新安装matplotlib: pip install --upgrade matplotlib")


def install_chinese_fonts_windows():
    """
    在Windows上安装中文字体的说明
    """
    print("🪟 Windows中文字体安装说明:")
    print("1. 系统自带字体:")
    print("   - Microsoft YaHei (微软雅黑)")
    print("   - SimHei (黑体)")
    print("   - SimSun (宋体)")
    print()
    print("2. 如果字体显示异常，可以尝试:")
    print("   - 重启Python环境")
    print("   - 清除matplotlib缓存")
    print("   - 重新安装matplotlib")


def install_chinese_fonts_linux():
    """
    在Linux上安装中文字体的说明
    """
    print("🐧 Linux中文字体安装说明:")
    print("1. Ubuntu/Debian:")
    print("   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
    print()
    print("2. CentOS/RHEL:")
    print("   sudo yum install wqy-microhei-fonts wqy-zenhei-fonts")
    print()
    print("3. 或者安装思源字体:")
    print("   sudo apt-get install fonts-noto-cjk")


def main():
    """
    主函数 - 字体配置和测试
    """
    print("🔧 强化学习项目字体配置工具")
    print("=" * 50)
    
    # 设置中文字体
    font = setup_chinese_font()
    
    # 显示字体信息
    get_font_info()
    
    # 测试中文显示
    print("\n🧪 测试中文显示效果...")
    test_font = test_chinese_display()
    
    # 根据操作系统显示安装说明
    system = platform.system()
    print(f"\n💡 {system}系统字体安装说明:")
    if system == "Darwin":
        install_chinese_fonts_macos()
    elif system == "Windows":
        install_chinese_fonts_windows()
    else:
        install_chinese_fonts_linux()
    
    print("\n✅ 字体配置完成！")
    print("现在可以在项目中使用中文字体了。")


if __name__ == "__main__":
    main()
