"""
环境初始化模块 (Environment Initialization)
=========================================

这个模块负责：
1. 设置随机种子以确保结果可复现
2. 配置TensorFlow/Keras在Mac M1上的运行
3. 验证依赖库版本
4. 设置日志系统
5. 配置GPU/CPU/Metal计算设备
"""

import os
import sys
import platform
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

try:
    from config import Config
except ImportError:
    from .config import Config


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    配置日志系统（扩展版）
    
    支持：
    - 控制台输出（带颜色和格式）
    - 文件输出（持久化记录）
    - 多个处理器管理
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 日志文件路径（可选）
    
    Returns:
        配置后的logger对象
    """
    logger = logging.getLogger('experiment')
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # 清除既有handler，防止重复添加（新增安全检查）
    logger.handlers = []
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    
    # 控制台日志格式（包含时间戳）
    console_formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        
        # 文件日志格式（更详细）
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"✓ 日志文件已启用: {log_file}")
    
    return logger


def setup_reproducibility(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    logger = logging.getLogger('experiment')
    
    import random
    import numpy as np
    
    # Python随机数生成器
    random.seed(seed)
    
    # NumPy随机数生成器
    np.random.seed(seed)
    
    # NumPy配置
    np.set_printoptions(precision=4, suppress=True)
    
    logger.info(f"✓ Random seed set to {seed}")
    
    # TensorFlow随机种子
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        logger.info(f"✓ TensorFlow random seed set to {seed}")
    except ImportError:
        logger.debug("TensorFlow not imported, skipping TensorFlow seed setup")
    except Exception as e:
        logger.warning(f"Failed to set TensorFlow seed: {e}")


def setup_mac_m1_environment(use_cpu: bool = True, 
                            use_metal: bool = False,
                            max_threads: int = 4) -> Dict[str, Any]:
    """
    配置Mac M1特定的TensorFlow/计算环境
    
    Args:
        use_cpu: 是否强制使用CPU
        use_metal: 是否启用Apple Metal加速（实验性）
        max_threads: CPU最大线程数
    
    Returns:
        环境配置信息字典
    """
    logger = logging.getLogger('experiment')
    config_info = {'device': 'unknown', 'status': 'not_configured'}
    
    current_platform = platform.system()
    machine = platform.machine()
    
    logger.info(f"系统检测: {current_platform} {machine}")
    
    # 检查是否为Mac M1
    is_mac_m1 = current_platform == 'Darwin' and machine == 'arm64'
    
    if not is_mac_m1:
        logger.warning(f"当前系统不是Mac M1 (arm64)，Mac M1特定配置可能不适用")
        config_info['device'] = machine
        config_info['status'] = 'not_mac_m1'
        return config_info
    
    logger.info("检测到Mac M1芯片，应用M1优化配置...")
    
    try:
        import tensorflow as tf
        
        # 设置环境变量
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志
        
        if use_cpu:
            # 禁用GPU
            tf.config.set_visible_devices([], 'GPU')
            logger.info("✓ TensorFlow GPU已禁用，使用CPU")
            config_info['device'] = 'CPU'
        else:
            # 列出可用设备
            devices = tf.config.list_physical_devices()
            logger.info(f"可用物理设备: {devices}")
            
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                logger.info(f"检测到GPU设备: {gpu_devices}")
                # 启用Metal加速
                if use_metal:
                    logger.info("✓ Apple Metal加速已启用（实验性）")
                    config_info['device'] = 'Metal'
                else:
                    config_info['device'] = 'GPU'
            else:
                logger.info("未检测到GPU，使用CPU")
                config_info['device'] = 'CPU'
        
        # 设置线程数
        tf.config.threading.set_inter_op_parallelism_threads(max_threads)
        tf.config.threading.set_intra_op_parallelism_threads(max_threads)
        logger.info(f"✓ TensorFlow线程数设置为: {max_threads}")
        
        config_info['status'] = 'configured'
        config_info['tensorflow_version'] = tf.__version__
        
    except ImportError:
        logger.warning("TensorFlow未安装，跳过TensorFlow配置")
        config_info['status'] = 'tensorflow_not_installed'
    except Exception as e:
        logger.error(f"配置TensorFlow失败: {e}")
        config_info['status'] = 'configuration_failed'
        config_info['error'] = str(e)
    
    return config_info


def check_dependencies() -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
    """
    检查必要的依赖库
    
    Returns:
        (所有依赖都满足, {包名: (是否安装, 版本)})
    """
    logger = logging.getLogger('experiment')
    
    required_packages: Dict[str, str] = {
        'numpy': 'numpy>=1.19.0',
        'pandas': 'pandas>=1.0.0',
        'sklearn': 'scikit-learn>=0.24.0',
        'yaml': 'pyyaml>=5.3',
    }
    
    optional_packages: Dict[str, str] = {
        'tensorflow': 'tensorflow>=2.7.0',
        'keras': 'keras>=2.4.0',
        'matplotlib': 'matplotlib>=3.3.0',
        'seaborn': 'seaborn>=0.11.0',
        'statsmodels': 'statsmodels>=0.12.0',
    }
    
    dependency_status: Dict[str, Tuple[bool, str]] = {}
    all_required_met = True
    
    logger.info("=" * 70)
    logger.info("检查依赖库 (Checking Dependencies)")
    logger.info("=" * 70)
    
    # 检查必要的包
    logger.info("\n必要的包 (Required Packages):")
    for package_name, requirement in required_packages.items():
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            dependency_status[package_name] = (True, version)
            logger.info(f"  ✓ {package_name}: {version}")
        except ImportError:
            dependency_status[package_name] = (False, 'not installed')
            logger.error(f"  ✗ {package_name}: NOT INSTALLED (required: {requirement})")
            all_required_met = False
    
    # 检查可选的包
    logger.info("\n可选的包 (Optional Packages):")
    for package_name, requirement in optional_packages.items():
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            dependency_status[package_name] = (True, version)
            logger.info(f"  ✓ {package_name}: {version}")
        except ImportError:
            dependency_status[package_name] = (False, 'not installed')
            logger.warning(f"  ⚠ {package_name}: NOT INSTALLED (optional: {requirement})")
    
    logger.info("=" * 70)
    
    if not all_required_met:
        logger.error("缺少必要的依赖库! 请安装所有必要的包。")
        logger.info("\n推荐安装命令:")
        logger.info("pip install numpy pandas scikit-learn pyyaml")
        logger.info("pip install tensorflow keras matplotlib seaborn statsmodels  # 可选")
    
    return all_required_met, dependency_status


def setup_environment(config: Config, 
                     verbose: bool = True) -> Dict[str, Any]:
    """
    完整的环境设置流程
    
    这个函数协调所有的环境初始化步骤：
    1. 设置日志系统
    2. 检查依赖库
    3. 设置随机种子（确保可复现性）
    4. 配置Mac M1特定设置
    5. 创建输出目录
    
    Args:
        config: Config对象
        verbose: 是否输出详细日志
    
    Returns:
        环境配置信息字典
    """
    # 设置日志系统
    log_level = config.logging.get('level', 'INFO') if config.logging else 'INFO'
    log_file = config.logging.get('log_file') if config.logging else None
    logger = setup_logging(log_level=log_level, log_file=log_file)
    
    logger.info("\n" + "=" * 70)
    logger.info("开始环境初始化 (Starting Environment Initialization)")
    logger.info("=" * 70)
    
    environment_info: Dict[str, Any] = {}
    
    # 1. 检查依赖库
    logger.info("\n步骤1: 检查依赖库")
    all_required_met, dependency_status = check_dependencies()
    environment_info['dependencies'] = {
        'all_required_met': all_required_met,
        'status': dependency_status
    }
    
    if not all_required_met:
        logger.error("缺少必要的依赖库，无法继续!")
        return environment_info
    
    # 2. 设置随机种子
    logger.info("\n步骤2: 设置随机种子")
    seed = config.experiment.get('seed', 42) if config.experiment else 42
    setup_reproducibility(seed=seed)
    environment_info['random_seed'] = seed
    
    # 3. 配置Mac M1环境
    logger.info("\n步骤3: 配置计算设备")
    if config.mac_m1:
        use_cpu = config.mac_m1.get('tensorflow_force_cpu', True)
        use_metal = config.mac_m1.get('use_metal_acceleration', False)
        max_threads = config.mac_m1.get('max_threads', 4)
    else:
        use_cpu, use_metal, max_threads = True, False, 4
    
    device_config = setup_mac_m1_environment(
        use_cpu=use_cpu,
        use_metal=use_metal,
        max_threads=max_threads
    )
    environment_info['device_config'] = device_config
    
    # 4. 创建输出目录
    logger.info("\n步骤4: 创建输出目录")
    output_config = config.output if config.output else {}
    necessary_dirs = [
        output_config.get('results_dir', './results/'),
        output_config.get('models_dir', './models/'),
        output_config.get('figures_dir', './figures/'),
        log_file and str(Path(log_file).parent) or None
    ]
    
    for dir_path in necessary_dirs:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✓ 目录已创建/确认: {dir_path}")
    
    # 5. 打印配置摘要
    if verbose:
        logger.info("\n步骤5: 配置摘要")
        config.print_summary()
    
    logger.info("\n" + "=" * 70)
    logger.info("环境初始化完成! (Environment Initialization Complete)")
    logger.info("=" * 70 + "\n")
    
    return environment_info


# 便捷函数：快速初始化
def quick_init(config_file: str = "config.yaml",
               supply_chain_file: str = "supply_chain.yaml") -> Tuple[Config, Dict[str, Any]]:
    """
    快速初始化配置和环境
    
    Args:
        config_file: 配置文件路径
        supply_chain_file: 供应链配置文件路径
    
    Returns:
        (Config对象, 环境信息字典)
    """
    # 加载配置
    config = Config(config_file=config_file, 
                   supply_chain_file=supply_chain_file,
                   verbose=True)
    
    # 设置环境
    environment_info = setup_environment(config, verbose=True)
    
    return config, environment_info


# ============================================================================
# 新增函数：环境快照和 dry-run 检查（增量扩展）
# ============================================================================

def save_environment_snapshot(config: 'Config', env_info: Dict[str, Any], 
                             snapshot_dir: Optional[str] = None) -> str:
    """
    保存环境快照到文件（新增）
    
    记录：
    - Python版本和路径
    - 已安装包及版本
    - TensorFlow配置
    - 计算设备信息
    - 系统信息
    
    Args:
        config: Config对象
        env_info: 环境信息字典
        snapshot_dir: 快照目录，默认使用config中的设置
    
    Returns:
        快照文件路径
    """
    import json
    import platform
    from datetime import datetime
    
    logger = logging.getLogger('experiment')
    
    if snapshot_dir is None:
        snapshot_dir = './outputs/env_snapshots/'
    
    # 创建快照目录
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成快照数据
    # 安全地提取依赖信息
    deps_dict = {}
    dep_check = env_info.get('dependencies', {})
    if isinstance(dep_check, dict):
        # 从 dependencies 检查结果中提取（perform_dry_run 返回的格式）
        if 'required_packages' in dep_check:
            deps_dict.update(dep_check.get('required_packages', {}))
        if 'optional_packages' in dep_check:
            deps_dict.update(dep_check.get('optional_packages', {}))
    
    env_snapshot = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable,
        },
        'tensorflow': env_info.get('device_config', {}),
        'dependencies': deps_dict,
        'random_seed': env_info.get('random_seed'),
    }
    
    # 生成快照文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_filename = f"env_snapshot_{timestamp}.json"
    snapshot_path = Path(snapshot_dir) / snapshot_filename
    
    # 保存快照
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(env_snapshot, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Environment snapshot saved: {snapshot_path}")
    return str(snapshot_path)


def perform_dry_run(config: Optional['Config'] = None, 
                   config_file: str = "config.yaml",
                   supply_chain_file: str = "supply_chain.yaml") -> Dict[str, Any]:
    """
    执行完整的 dry-run 检查（新增）
    
    验证项：
    1. 配置文件完整性
    2. 环境依赖可用性
    3. 配置逻辑正确性
    4. 输出目录可写性
    5. TensorFlow/GPU可用性
    6. 平台兼容性
    
    Args:
        config: Config对象（如为None则自动加载）
        config_file: 配置文件路径
        supply_chain_file: 供应链配置文件路径
    
    Returns:
        检查结果字典 {'status': 'pass/warn/fail', 'checks': {...}, 'errors': [...]}
    """
    logger = logging.getLogger('experiment')
    
    logger.info("\n" + "=" * 70)
    logger.info("执行 DRY-RUN 完整检查")
    logger.info("=" * 70)
    
    dry_run_result = {
        'status': 'pass',
        'checks': {},
        'warnings': [],
        'errors': [],
        'timestamp': None
    }
    
    import datetime
    import platform
    dry_run_result['timestamp'] = datetime.datetime.now().isoformat()
    
    # 检查1: 配置文件存在性
    logger.info("\n【检查1】配置文件完整性...")
    config_files = {
        config_file: "主配置文件",
        supply_chain_file: "供应链配置文件"
    }
    
    config_check = {'status': 'pass', 'details': {}}
    for cfg_file, desc in config_files.items():
        if Path(cfg_file).exists():
            logger.info(f"  ✓ {desc}: {cfg_file}")
            config_check['details'][cfg_file] = 'found'
        else:
            logger.warning(f"  ⚠ {desc} 未找到: {cfg_file}")
            config_check['status'] = 'warn'
            config_check['details'][cfg_file] = 'missing'
    
    dry_run_result['checks']['config_files'] = config_check
    
    # 检查2: 加载和验证配置
    logger.info("\n【检查2】配置加载和验证...")
    try:
        if config is None:
            config = Config(config_file=config_file, 
                          supply_chain_file=supply_chain_file,
                          verbose=False)
        
        # 执行跨字段验证
        validation_success = config.validate_and_report()
        
        logger.info("  ✓ 配置加载成功")
        if validation_success:
            logger.info("  ✓ 配置验证通过")
            dry_run_result['checks']['config_validation'] = {'status': 'pass'}
        else:
            logger.warning("  ⚠ 配置验证有警告")
            dry_run_result['checks']['config_validation'] = {
                'status': 'warn',
                'errors': config._runtime_state.get('validation_errors', [])
            }
    except Exception as e:
        logger.error(f"  ✗ 配置加载失败: {e}")
        dry_run_result['checks']['config_validation'] = {'status': 'fail', 'error': str(e)}
        dry_run_result['status'] = 'fail'
        dry_run_result['errors'].append(str(e))
    
    # 检查3: 依赖库可用性
    logger.info("\n【检查3】依赖库可用性...")
    all_required_met, dep_status = check_dependencies()
    
    dep_check = {
        'status': 'pass' if all_required_met else 'fail',
        'required_packages': {},
        'optional_packages': {}
    }
    
    for pkg, (installed, version) in dep_status.items():
        if installed:
            dep_check['required_packages' if pkg in ['numpy', 'pandas', 'scikit-learn', 'pyyaml'] 
                     else 'optional_packages'][pkg] = version
    
    if not all_required_met:
        dry_run_result['status'] = 'fail'
        dry_run_result['errors'].append("缺少必要的依赖库")
    
    dry_run_result['checks']['dependencies'] = dep_check
    
    # 检查4: 输出目录可写性
    logger.info("\n【检查4】输出目录可写性...")
    output_dirs = [
        config.output.get('results_dir', './results/') if config else './results/',
        config.output.get('models_dir', './models/') if config else './models/',
        config.output.get('figures_dir', './figures/') if config else './figures/',
        './outputs/run_snapshots/',
        './outputs/env_snapshots/',
        './logs/'
    ]
    
    dir_check = {'status': 'pass', 'directories': {}}
    for dir_path in output_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            # 尝试写入测试文件
            test_file = Path(dir_path) / '.dry_run_test.tmp'
            test_file.write_text('dry-run test')
            test_file.unlink()
            logger.info(f"  ✓ {dir_path}")
            dir_check['directories'][dir_path] = 'writable'
        except Exception as e:
            logger.warning(f"  ⚠ {dir_path} 不可写: {e}")
            dir_check['status'] = 'warn'
            dir_check['directories'][dir_path] = f'error: {str(e)}'
            dry_run_result['warnings'].append(f"{dir_path} 不可写")
    
    dry_run_result['checks']['output_directories'] = dir_check
    
    # 检查5: 平台和设备
    logger.info("\n【检查5】平台和设备信息...")
    sys_check = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
    }
    
    logger.info(f"  平台: {sys_check['platform']} ({sys_check['machine']})")
    logger.info(f"  Python: {sys_check['python_version']}")
    
    # 检查Mac M1特定信息
    if sys_check['machine'] == 'arm64' and sys_check['platform'] == 'Darwin':
        logger.info("  ✓ 检测到 Mac M1 芯片")
        sys_check['mac_m1'] = True
        if config and config.mac_m1:
            logger.info(f"    - TensorFlow CPU强制: {config.mac_m1.get('tensorflow_force_cpu')}")
    
    # 检查6: TensorFlow可用性
    logger.info("\n【检查6】TensorFlow和深度学习框架...")
    tf_check = {'status': 'info', 'available': False}
    try:
        import tensorflow as tf
        tf_check['available'] = True
        tf_check['version'] = tf.__version__
        logger.info(f"  ✓ TensorFlow {tf.__version__} 可用")
        
        # 检查GPU/设备
        devices = tf.config.list_physical_devices()
        logger.info(f"    - 物理设备数: {len(devices)}")
        for device in devices:
            logger.info(f"      - {device}")
        
        tf_check['devices'] = [str(d) for d in devices]
    except ImportError:
        logger.warning("  ⚠ TensorFlow 未安装 (可选)")
        tf_check['status'] = 'warn'
    except Exception as e:
        logger.warning(f"  ⚠ TensorFlow 检查失败: {e}")
        tf_check['status'] = 'warn'
    
    dry_run_result['checks']['tensorflow'] = tf_check
    
    # 最终总结
    logger.info("\n" + "=" * 70)
    logger.info("DRY-RUN 检查完成")
    logger.info("=" * 70)
    
    if dry_run_result['status'] == 'pass':
        logger.info("✓ 所有检查均通过，系统已准备好运行实验")
    elif dry_run_result['status'] == 'warn':
        logger.warning("⚠ 检查通过，但存在可能的问题")
        for warning in dry_run_result['warnings']:
            logger.warning(f"  - {warning}")
    else:
        logger.error("✗ 检查失败，请解决以下问题:")
        for error in dry_run_result['errors']:
            logger.error(f"  - {error}")
    
    return dry_run_result


if __name__ == "__main__":
    # 测试脚本
    config, env_info = quick_init()
    print("\n环境信息摘要:")
    print(f"- 随机种子: {env_info.get('random_seed')}")
    print(f"- 计算设备: {env_info['device_config'].get('device')}")
    print(f"- 依赖库检查: {env_info['dependencies']['all_required_met']}")
