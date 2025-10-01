#!/usr/bin/env python3
"""
项目清理脚本
清理临时文件、缓存文件和重复文件
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_pycache():
    """清理Python缓存文件"""
    print("🧹 清理Python缓存文件...")
    
    # 查找所有__pycache__目录
    pycache_dirs = []
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_dirs.append(os.path.join(root, dir_name))
    
    # 删除__pycache__目录
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"  ✅ 删除: {pycache_dir}")
        except Exception as e:
            print(f"  ❌ 删除失败: {pycache_dir} - {e}")
    
    # 查找并删除.pyc文件
    pyc_files = glob.glob('**/*.pyc', recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            print(f"  ✅ 删除: {pyc_file}")
        except Exception as e:
            print(f"  ❌ 删除失败: {pyc_file} - {e}")

def cleanup_temp_files():
    """清理临时文件"""
    print("\n🧹 清理临时文件...")
    
    temp_patterns = [
        '**/*.tmp',
        '**/*.temp',
        '**/*.log',
        '**/*.bak',
        '**/*.swp',
        '**/*.swo',
        '**/*~',
        '**/.DS_Store',
        '**/Thumbs.db'
    ]
    
    for pattern in temp_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"  ✅ 删除: {file_path}")
            except Exception as e:
                print(f"  ❌ 删除失败: {file_path} - {e}")

def cleanup_empty_dirs():
    """清理空目录"""
    print("\n🧹 清理空目录...")
    
    empty_dirs = []
    for root, dirs, files in os.walk('.', topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # 目录为空
                    empty_dirs.append(dir_path)
            except OSError:
                pass
    
    for empty_dir in empty_dirs:
        try:
            os.rmdir(empty_dir)
            print(f"  ✅ 删除空目录: {empty_dir}")
        except Exception as e:
            print(f"  ❌ 删除失败: {empty_dir} - {e}")

def cleanup_duplicate_files():
    """清理重复文件"""
    print("\n🧹 检查重复文件...")
    
    # 检查是否有重复的测试文件
    test_files = [
        'test_setup.py',
        'test_notebook_import.py',
        'test_chinese_font.py'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"  ⚠️ 发现重复文件: {test_file}")
            # 这些文件已经在scripts目录中，可以删除根目录的副本
            if test_file != 'test_chinese_font.py':  # 保留这个文件
                try:
                    os.remove(test_file)
                    print(f"  ✅ 删除重复文件: {test_file}")
                except Exception as e:
                    print(f"  ❌ 删除失败: {test_file} - {e}")

def organize_logs():
    """整理日志文件"""
    print("\n📁 整理日志文件...")
    
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 移动日志文件到logs目录
    log_patterns = [
        '*.log',
        '*.out',
        '*.err'
    ]
    
    for pattern in log_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            if not file_path.startswith('logs/'):
                try:
                    shutil.move(file_path, 'logs/')
                    print(f"  ✅ 移动: {file_path} -> logs/")
                except Exception as e:
                    print(f"  ❌ 移动失败: {file_path} - {e}")

def check_project_structure():
    """检查项目结构"""
    print("\n🔍 检查项目结构...")
    
    required_dirs = [
        'docs',
        'algorithms',
        'environments',
        'experiments',
        'exercises',
        'notebooks',
        'progress',
        'utils',
        'scripts',
        'data',
        'models',
        'configs',
        'assets',
        'theory',
        'src',
        'logs'
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ 缺失: {dir_name}/")

def generate_cleanup_report():
    """生成清理报告"""
    print("\n📊 生成清理报告...")
    
    report = []
    report.append("# 项目清理报告")
    report.append("=" * 50)
    report.append(f"清理时间: {os.popen('date').read().strip()}")
    report.append("")
    
    # 统计文件数量
    total_files = 0
    total_dirs = 0
    
    for root, dirs, files in os.walk('.'):
        total_dirs += len(dirs)
        total_files += len(files)
    
    report.append(f"总文件数: {total_files}")
    report.append(f"总目录数: {total_dirs}")
    report.append("")
    
    # 统计各目录大小
    report.append("## 目录大小统计")
    for root, dirs, files in os.walk('.'):
        if root == '.':
            continue
        if root.startswith('./.'):
            continue
        
        dir_size = 0
        for file_path in files:
            try:
                dir_size += os.path.getsize(os.path.join(root, file_path))
            except OSError:
                pass
        
        if dir_size > 0:
            size_mb = dir_size / (1024 * 1024)
            report.append(f"- {root}: {size_mb:.2f} MB")
    
    # 保存报告
    with open('logs/cleanup_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("  ✅ 清理报告已保存到: logs/cleanup_report.txt")

def main():
    """主函数"""
    print("🧹 强化学习项目清理工具")
    print("=" * 50)
    
    # 切换到项目根目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')
    
    print(f"📁 当前工作目录: {os.getcwd()}")
    
    # 执行清理操作
    cleanup_pycache()
    cleanup_temp_files()
    cleanup_duplicate_files()
    organize_logs()
    cleanup_empty_dirs()
    
    # 检查项目结构
    check_project_structure()
    
    # 生成报告
    generate_cleanup_report()
    
    print("\n🎉 项目清理完成！")
    print("项目现在更加整洁和有序了。")

if __name__ == "__main__":
    main()
