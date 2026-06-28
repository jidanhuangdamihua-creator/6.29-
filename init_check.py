#!/usr/bin/env python3
"""
初始化检查脚本 (Initialization Check Script)

这个脚本验证配置系统是否正确安装和配置。
可以通过运行此脚本来快速诊断问题。

用法:
    python init_check.py
"""

import sys
import os
from pathlib import Path
import importlib.util

# 确保当前目录在 Python 路径中，用于直接运行脚本
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 预加载 config 和 environment 模块，使用 importlib 避免相对导入问题
def _load_modules():
    """加载 config 和 environment 模块"""
    config_path = os.path.join(current_dir, 'config.py')
    env_path = os.path.join(current_dir, 'environment.py')
    
    try:
        config_spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        sys.modules['config'] = config_module
        
        env_spec = importlib.util.spec_from_file_location("environment", env_path)
        env_module = importlib.util.module_from_spec(env_spec)
        env_spec.loader.exec_module(env_module)
        sys.modules['environment'] = env_module
        
        return True
    except Exception as e:
        return False

_load_modules()

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def print_success(text):
    """打印成功信息"""
    print(f"✓ {text}")

def print_error(text):
    """打印错误信息"""
    print(f"✗ {text}")

def print_warning(text):
    """打印警告信息"""
    print(f"⚠ {text}")

def check_files():
    """检查所需的文件是否存在"""
    print_header("步骤1: 检查文件")
    
    required_files = [
        'config.py',
        'environment.py',
        '__init__.py',
        'config.yaml',
        'supply_chain.yaml'
    ]
    
    all_exist = True
    for filename in required_files:
        if Path(filename).exists():
            print_success(f"找到: {filename}")
        else:
            print_error(f"缺失: {filename}")
            all_exist = False
    
    return all_exist

def check_dependencies():
    """检查Python依赖"""
    print_header("步骤2: 检查Python依赖")
    
    required = {
        'yaml': 'pyyaml',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn'
    }
    
    optional = {
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        'matplotlib': 'matplotlib'
    }
    
    all_required_met = True
    
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            version = getattr(__import__(module_name), '__version__', 'unknown')
            print_success(f"{package_name}: {version}")
        except ImportError:
            print_error(f"{package_name}: 未安装")
            all_required_met = False
    
    print("\n可选依赖:")
    for module_name, package_name in optional.items():
        try:
            __import__(module_name)
            version = getattr(__import__(module_name), '__version__', 'unknown')
            print_success(f"{package_name}: {version}")
        except ImportError:
            print_warning(f"{package_name}: 未安装（可选）")
    
    return all_required_met

def check_imports():
    """检查能否导入模块"""
    print_header("步骤3: 检查模块导入")
    
    try:
        from config import Config
        print_success("成功导入: Config")
    except Exception as e:
        print_error(f"导入Config失败: {e}")
        return False
    
    try:
        from environment import quick_init, setup_environment
        print_success("成功导入: quick_init, setup_environment")
    except Exception as e:
        print_error(f"导入environment模块失败: {e}")
        return False
    
    return True

def check_config_loading():
    """检查配置是否可以加载"""
    print_header("步骤4: 检查配置加载")
    
    try:
        from config import Config
        config = Config(verbose=False)
        
        # 检查各个配置是否成功加载
        checks = [
            (config.dataset is not None, "数据集配置"),
            (config.model is not None, "模型配置"),
            (config.training is not None, "训练配置"),
            (config.supply_chain is not None, "供应链配置"),
        ]
        
        all_loaded = True
        for check, name in checks:
            if check:
                print_success(f"加载: {name}")
            else:
                print_error(f"未加载: {name}")
                all_loaded = False
        
        # 输出一些配置值
        if config.dataset:
            print(f"\n  数据集: {config.dataset.name}")
            print(f"  预测步长: {config.dataset.forecast_horizon}")
        
        if config.model:
            print(f"  模型: {config.model.architecture}")
        
        if config.training:
            print(f"  Epochs: {config.training.epochs}")
            print(f"  Batch Size: {config.training.batch_size}")
        
        return all_loaded
    
    except Exception as e:
        print_error(f"配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """检查环境初始化"""
    print_header("步骤5: 检查环境初始化")
    
    try:
        from environment import setup_logging, setup_reproducibility
        
        # 设置日志
        logger = setup_logging(log_level="INFO")
        print_success("日志系统已初始化")
        
        # 设置随机种子
        setup_reproducibility(seed=42)
        print_success("随机种子已设置")
        
        return True
    
    except Exception as e:
        print_error(f"环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_quick_init():
    """检查快速初始化功能"""
    print_header("步骤6: 检查快速初始化")
    
    try:
        from environment import quick_init
        
        print("执行 quick_init()...")
        config, env_info = quick_init()
        
        print_success("快速初始化成功完成")
        
        # 输出初始化结果
        print(f"  依赖检查: {'✓ 通过' if env_info['dependencies']['all_required_met'] else '✗ 失败'}")
        print(f"  计算设备: {env_info['device_config'].get('device', '未知')}")
        print(f"  随机种子: {env_info.get('random_seed')}")
        
        return True
    
    except Exception as e:
        print_error(f"快速初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print_header("实验配置与环境准备模块 - 初始化检查")
    
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    
    # 执行检查
    checks = [
        ("文件检查", check_files),
        ("依赖检查", check_dependencies),
        ("导入检查", check_imports),
        ("配置加载", check_config_loading),
        ("环境初始化", check_environment),
        ("快速初始化", check_quick_init),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"{name}检查异常: {e}")
            results.append((name, False))
    
    # 汇总结果
    print_header("检查结果汇总")
    
    all_passed = True
    for name, result in results:
        if result:
            print_success(f"{name}: 通过")
        else:
            print_error(f"{name}: 失败")
            all_passed = False
    
    # 最终状态
    print_header("最终状态")
    
    if all_passed:
        print_success("所有检查已通过! ✨")
        print("\n现在可以使用以下代码初始化配置:")
        print("""
from environment import quick_init

config, env_info = quick_init()
print(config.dataset.name)
        """)
        return 0
    else:
        print_error("某些检查未通过，请检查错误信息")
        print("\n常见问题:")
        print("  1. 确保所有.py和.yaml文件都在当前目录")
        print("  2. 安装依赖: pip install numpy pandas scikit-learn pyyaml")
        print("  3. 检查Python版本 (需要3.8+)")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n检查被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n检查过程出现意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
