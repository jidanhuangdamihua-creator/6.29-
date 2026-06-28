"""
实验配置与环境模块 (Experiment Configuration and Environment Module)

这个包包含：
1. config.py - 配置解析和管理
2. environment.py - 环境初始化和配置

使用方法：
    from config_env import Config, setup_environment, quick_init
    
    # 方法1: 快速初始化
    config, env_info = quick_init()
    
    # 方法2: 分步骤初始化
    config = Config(config_file='config.yaml', supply_chain_file='supply_chain.yaml')
    env_info = setup_environment(config)
"""

from .config import (
    Config,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    SupplyChainConfig
)

from .environment import (
    setup_environment,
    setup_logging,
    setup_reproducibility,
    setup_mac_m1_environment,
    check_dependencies,
    quick_init
)

__all__ = [
    'Config',
    'DatasetConfig',
    'ModelConfig',
    'TrainingConfig',
    'SupplyChainConfig',
    'setup_environment',
    'setup_logging',
    'setup_reproducibility',
    'setup_mac_m1_environment',
    'check_dependencies',
    'quick_init'
]

__version__ = '1.0.0'
