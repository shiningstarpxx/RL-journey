#!/usr/bin/env python3
"""
项目错误检查和诊断脚本
全面检查项目中的潜在问题和错误
"""

import sys
import os
import traceback
import warnings
from typing import List, Dict, Any

def check_imports() -> Dict[str, Any]:
    """检查所有模块的导入"""
    print("🔍 检查模块导入...")
    results = {}
    
    modules = [
        'numpy',
        'matplotlib',
        'tqdm',
        'environments.grid_world',
        'algorithms.tabular.q_learning',
        'algorithms.tabular.sarsa',
        'progress.learning_tracker',
        'utils.font_config'
    ]
    
    for module in modules:
        try:
            __import__(module)
            results[module] = {'status': 'success', 'error': None}
            print(f"✅ {module}: 导入成功")
        except Exception as e:
            results[module] = {'status': 'error', 'error': str(e)}
            print(f"❌ {module}: 导入失败 - {e}")
    
    return results

def check_file_structure() -> Dict[str, Any]:
    """检查文件结构"""
    print("\n📁 检查文件结构...")
    results = {}
    
    required_files = [
        'algorithms/tabular/q_learning.py',
        'algorithms/tabular/sarsa.py',
        'environments/grid_world.py',
        'progress/learning_tracker.py',
        'utils/font_config.py',
        'experiments/week2_q_learning/experiment_1_q_learning.py',
        'requirements.txt',
        'RL_LEARNING_PATH.md',
        'QUICK_START_GUIDE.md'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            results[file_path] = {'status': 'exists', 'size': os.path.getsize(file_path)}
            print(f"✅ {file_path}: 存在 ({results[file_path]['size']} bytes)")
        else:
            results[file_path] = {'status': 'missing', 'size': 0}
            print(f"❌ {file_path}: 不存在")
    
    return results

def check_syntax() -> Dict[str, Any]:
    """检查Python文件语法"""
    print("\n🔧 检查Python文件语法...")
    results = {}
    
    python_files = [
        'algorithms/tabular/q_learning.py',
        'algorithms/tabular/sarsa.py',
        'environments/grid_world.py',
        'progress/learning_tracker.py',
        'utils/font_config.py',
        'experiments/week2_q_learning/experiment_1_q_learning.py'
    ]
    
    for file_path in python_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), file_path, 'exec')
                results[file_path] = {'status': 'valid', 'error': None}
                print(f"✅ {file_path}: 语法正确")
            except SyntaxError as e:
                results[file_path] = {'status': 'syntax_error', 'error': str(e)}
                print(f"❌ {file_path}: 语法错误 - {e}")
            except Exception as e:
                results[file_path] = {'status': 'error', 'error': str(e)}
                print(f"❌ {file_path}: 检查失败 - {e}")
        else:
            results[file_path] = {'status': 'missing', 'error': 'File not found'}
            print(f"❌ {file_path}: 文件不存在")
    
    return results

def check_functionality() -> Dict[str, Any]:
    """检查核心功能"""
    print("\n🧪 检查核心功能...")
    results = {}
    
    try:
        # 测试环境创建
        from environments.grid_world import create_simple_grid_world
        env = create_simple_grid_world()
        results['environment_creation'] = {'status': 'success', 'error': None}
        print("✅ 环境创建: 成功")
        
        # 测试Q-Learning
        from algorithms.tabular.q_learning import QLearning
        q_learning = QLearning(env.state_space, env.action_space)
        results['q_learning_creation'] = {'status': 'success', 'error': None}
        print("✅ Q-Learning创建: 成功")
        
        # 测试训练
        reward, steps = q_learning.train_episode(env, max_steps=5)
        results['training'] = {'status': 'success', 'reward': reward, 'steps': steps}
        print(f"✅ 训练测试: 成功 (奖励: {reward:.2f}, 步数: {steps})")
        
        # 测试学习跟踪器
        from progress.learning_tracker import LearningTracker
        tracker = LearningTracker()
        results['learning_tracker'] = {'status': 'success', 'error': None}
        print("✅ 学习跟踪器: 成功")
        
        # 测试字体配置
        from utils.font_config import setup_chinese_font
        font = setup_chinese_font()
        results['font_config'] = {'status': 'success', 'font': font}
        print(f"✅ 字体配置: 成功 (字体: {font})")
        
    except Exception as e:
        results['functionality'] = {'status': 'error', 'error': str(e)}
        print(f"❌ 功能测试失败: {e}")
        traceback.print_exc()
    
    return results

def check_dependencies() -> Dict[str, Any]:
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    results = {}
    
    required_packages = [
        'numpy',
        'matplotlib',
        'tqdm'
    ]
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            results[package] = {'status': 'installed', 'version': version}
            print(f"✅ {package}: 已安装 (版本: {version})")
        except ImportError:
            results[package] = {'status': 'missing', 'version': None}
            print(f"❌ {package}: 未安装")
    
    return results

def check_warnings() -> Dict[str, Any]:
    """检查警告"""
    print("\n⚠️ 检查警告...")
    results = {}
    
    # 捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            from algorithms.tabular.q_learning import QLearning
            from environments.grid_world import create_simple_grid_world
            from progress.learning_tracker import LearningTracker
            from utils.font_config import setup_chinese_font
            
            if w:
                results['warnings'] = [str(warning.message) for warning in w]
                print(f"⚠️ 发现 {len(w)} 个警告:")
                for warning in w:
                    print(f"  - {warning.message}")
            else:
                results['warnings'] = []
                print("✅ 无警告")
                
        except Exception as e:
            results['warnings'] = [f"Error during warning check: {e}"]
            print(f"❌ 警告检查失败: {e}")
    
    return results

def generate_report(all_results: Dict[str, Any]) -> str:
    """生成检查报告"""
    report = []
    report.append("📊 项目错误检查报告")
    report.append("=" * 50)
    
    # 统计信息
    total_checks = 0
    passed_checks = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            for key, result in results.items():
                total_checks += 1
                if isinstance(result, dict) and result.get('status') in ['success', 'exists', 'valid', 'installed']:
                    passed_checks += 1
    
    report.append(f"总检查项: {total_checks}")
    report.append(f"通过检查: {passed_checks}")
    report.append(f"失败检查: {total_checks - passed_checks}")
    report.append(f"通过率: {passed_checks/total_checks*100:.1f}%")
    report.append("")
    
    # 详细结果
    for category, results in all_results.items():
        report.append(f"## {category}")
        if isinstance(results, dict):
            for key, result in results.items():
                if isinstance(result, dict):
                    status = result.get('status', 'unknown')
                    if status in ['success', 'exists', 'valid', 'installed']:
                        report.append(f"✅ {key}: {status}")
                    else:
                        report.append(f"❌ {key}: {status} - {result.get('error', 'No error message')}")
                else:
                    report.append(f"ℹ️ {key}: {result}")
        report.append("")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("🔧 强化学习项目错误检查工具")
    print("=" * 60)
    
    all_results = {}
    
    # 执行所有检查
    all_results['imports'] = check_imports()
    all_results['file_structure'] = check_file_structure()
    all_results['syntax'] = check_syntax()
    all_results['functionality'] = check_functionality()
    all_results['dependencies'] = check_dependencies()
    all_results['warnings'] = check_warnings()
    
    # 生成报告
    print("\n" + "=" * 60)
    report = generate_report(all_results)
    print(report)
    
    # 保存报告
    os.makedirs('logs', exist_ok=True)
    with open('logs/error_check_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 详细报告已保存到: logs/error_check_report.txt")
    
    # 总结
    total_checks = 0
    passed_checks = 0
    
    for results in all_results.values():
        if isinstance(results, dict):
            for result in results.values():
                total_checks += 1
                if isinstance(result, dict) and result.get('status') in ['success', 'exists', 'valid', 'installed']:
                    passed_checks += 1
                elif isinstance(result, list) and len(result) == 0:  # 空警告列表表示无警告
                    passed_checks += 1
    
    if passed_checks == total_checks:
        print("🎉 所有检查都通过了！项目状态良好。")
    else:
        print(f"⚠️ 发现 {total_checks - passed_checks} 个问题，请查看详细报告。")

if __name__ == "__main__":
    main()
