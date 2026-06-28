"""
配置与环境初始化使用示例 (Configuration and Environment Initialization Examples)

本脚本演示如何使用配置与环境模块的各种功能。
"""

import sys
from pathlib import Path
import logging

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from environment import (
    setup_environment,
    setup_logging,
    check_dependencies,
    quick_init
)


def example1_basic_usage():
    """示例1: 基础使用 - 加载配置文件"""
    print("\n" + "=" * 70)
    print("示例1: 基础配置加载")
    print("=" * 70)
    
    # 创建Config对象  - 自动加载config.yaml和supply_chain.yaml
    config = Config(config_file="config.yaml", 
                   supply_chain_file="supply_chain.yaml",
                   verbose=True)
    
    # 访问配置信息
    print("\n访问配置信息:")
    print(f"  数据集名称: {config.dataset.name}")
    print(f"  预测步长: {config.dataset.forecast_horizon}")
    print(f"  模型架构: {config.model.architecture}")
    print(f"  训练epochs: {config.training.epochs}")
    print(f"  目标服务水平: {config.supply_chain.target_service_level}")


def example2_access_patterns():
    """示例2: 配置访问方式"""
    print("\n" + "=" * 70)
    print("示例2: 多种配置访问方式")
    print("=" * 70)
    
    config = Config()
    
    # 方式1: 直接属性访问
    print("\n方式1 - 直接属性访问:")
    print(f"  config.dataset.forecast_horizon = {config.dataset.forecast_horizon}")
    print(f"  config.training.batch_size = {config.training.batch_size}")
    
    # 方式2: 字典式访问（给定默认值）
    print("\n方式2 - get()方法，支持点记法:")
    horizon = config.get('dataset.forecast_horizon')
    batch_size = config.get('training.batch_size', 32)
    print(f"  forecast_horizon = {horizon}")
    print(f"  batch_size = {batch_size}")
    
    # 方式3: 转换为字典
    print("\n方式3 - 转换为字典:")
    config_dict = config.to_dict()
    print(f"  字典键: {list(config_dict.keys())}")


def example3_dynamic_modification():
    """示例3: 动态修改配置"""
    print("\n" + "=" * 70)
    print("示例3: 动态修改配置")
    print("=" * 70)
    
    config = Config()
    
    print(f"\n修改前的batch_size: {config.training.batch_size}")
    
    # 修改配置
    config.set('training.batch_size', 64)
    config.set('training.epochs', 200)
    config.set('dataset.forecast_horizon', 24)
    
    print(f"修改后的batch_size: {config.training.batch_size}")
    print(f"修改后的epochs: {config.training.epochs}")
    print(f"修改后的forecast_horizon: {config.dataset.forecast_horizon}")


def example4_dependency_check():
    """示例4: 依赖库检查"""
    print("\n" + "=" * 70)
    print("示例4: 依赖库检查")
    print("=" * 70)
    
    # 设置日志
    logger = setup_logging(log_level="INFO")
    
    # 检查依赖库
    all_met, status = check_dependencies()
    
    print(f"\n所有必要依赖已安装: {all_met}")
    print("\n已安装的依赖:")
    for package, (installed, version) in status.items():
        if installed:
            print(f"  ✓ {package}: {version}")


def example5_full_initialization():
    """示例5: 完整初始化流程"""
    print("\n" + "=" * 70)
    print("示例5: 完整环境初始化")
    print("=" * 70)
    
    # 快速初始化：一步完成所有配置和环境设置
    config, env_info = quick_init(
        config_file="config.yaml",
        supply_chain_file="supply_chain.yaml"
    )
    
    print("\n初始化完成！环境信息:")
    print(f"  - 随机种子: {env_info.get('random_seed')}")
    print(f"  - 计算设备: {env_info['device_config'].get('device')}")
    print(f"  - TensorFlow版本: {env_info['device_config'].get('tensorflow_version', 'N/A')}")
    print(f"  - 依赖库检查: {'通过' if env_info['dependencies']['all_required_met'] else '失败'}")
    
    return config


def example6_config_for_different_datasets():
    """示例6: 针对不同数据集的配置"""
    print("\n" + "=" * 70)
    print("示例6: 针对不同数据集的配置切换")
    print("=" * 70)
    
    config = Config()
    
    print("\n原始数据集配置:")
    print(f"  名称: {config.dataset.name}")
    print(f"  路径: {config.dataset.path}")
    
    # 修改为不同的数据集
    datasets = [
        {
            'name': 'italian-pasta-demand',
            'path': './hierarchical_sales_data.csv',
            'K': 2,
            'horizon': 7
        },
        {
            'name': 'rossmann-store-sales',
            'path': './rossmann-store-sales (2)/',
            'K': 5,
            'horizon': 14
        }
    ]
    
    print("\n切换到意大利面需求数据集 (Dataset2):")
    config.dataset.name = datasets[0]['name']
    config.dataset.path = datasets[0]['path']
    config.dataset.num_source_sequences = datasets[0]['K']
    config.dataset.forecast_horizon = datasets[0]['horizon']
    print(f"  名称: {config.dataset.name}")
    print(f"  路径: {config.dataset.path}")
    print(f"  源序列数K: {config.dataset.num_source_sequences}")
    print(f"  预测步长: {config.dataset.forecast_horizon}")


def example7_model_architecture_comparison():
    """示例7: 不同模型架构的配置对比"""
    print("\n" + "=" * 70)
    print("示例7: 模型架构配置对比")
    print("=" * 70)
    
    config = Config()
    
    architectures = ['cnn', 'lstm', 'cnn_lstm', 'transformer']
    
    for arch in architectures:
        config.model.architecture = arch
        
        if arch == 'cnn':
            cnn_filters = config.model.cnn.num_filters
            print(f"\n{arch.upper()}:")
            print(f"  - CNN滤波器: {cnn_filters}")
            print(f"  - Dropout: {config.model.cnn.dropout_rate}")
        
        elif arch == 'lstm':
            print(f"\n{arch.upper()}:")
            print(f"  - LSTM单元: {config.model.lstm.units}")
            print(f"  - 层数: {config.model.lstm.num_layers}")
            print(f"  - Dropout: {config.model.lstm.dropout_rate}")
        
        elif arch == 'cnn_lstm':
            print(f"\n{arch.upper()}:")
            print(f"  - CNN滤波器: {config.model.cnn.num_filters}")
            print(f"  - LSTM单元: {config.model.lstm.units}")
            print(f"  - 注意力机制: {config.model.attention.enabled}")
        
        else:
            print(f"\n{arch.upper()}: (配置待定)")


def example8_supply_chain_analysis_setup():
    """示例8: 供应链仿真配置"""
    print("\n" + "=" * 70)
    print("示例8: 供应链仿真配置")
    print("=" * 70)
    
    config = Config()
    sc = config.supply_chain
    
    if sc:
        print("\n初始库存:")
        print(f"  - 数量: {sc.initial_inventory.get('quantity')}")
        print(f"  - 价值: {sc.initial_inventory.get('value')}")
        
        print("\n订货配置:")
        print(f"  - 订货批量Q: {sc.order_quantity_Q}")
        print(f"  - 前置期: {sc.lead_time_days}天")
        
        print("\n成本参数:")
        print(f"  - 订货成本: {sc.ordering_cost_per_order}元/次")
        print(f"  - 采购成本: {sc.unit_acquisition_cost}元/单位")
        print(f"  - 持有成本: {sc.holding_cost_daily}元/单位/天")
        print(f"  - 缺货成本: {sc.stockout_cost_per_unit}元/单位")
        
        print("\n服务水平:")
        print(f"  - 目标服务水平: {sc.target_service_level*100}%")
        print(f"  - 目标订单完成率: {sc.target_fill_rate*100}%")
        
        # 计算年度成本示例
        annual_demand = 365 * 10  # 假设日均需求10单位
        num_orders = annual_demand / sc.order_quantity_Q
        
        ordering_cost = num_orders * sc.ordering_cost_per_order
        acquisition_cost = annual_demand * sc.unit_acquisition_cost
        holding_cost = (sc.order_quantity_Q / 2) * sc.holding_cost_daily * 365
        
        total_cost = ordering_cost + acquisition_cost + holding_cost
        
        print("\n年度成本估算 (假设年需求=3650单位):")
        print(f"  - 订货成本: ¥{ordering_cost:.2f}")
        print(f"  - 采购成本: ¥{acquisition_cost:.2f}")
        print(f"  - 持有成本: ¥{holding_cost:.2f}")
        print(f"  - 总成本: ¥{total_cost:.2f}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("配置与环境初始化 - 完整使用示例")
    print("=" * 70)
    
    try:
        # 运行各个示例
        example1_basic_usage()
        example2_access_patterns()
        example3_dynamic_modification()
        example4_dependency_check()
        example5_full_initialization()
        example6_config_for_different_datasets()
        example7_model_architecture_comparison()
        example8_supply_chain_analysis_setup()
        
        print("\n" + "=" * 70)
        print("所有示例执行完成！")
        print("=" * 70)
    
    except FileNotFoundError as e:
        print(f"\n错误: 找不到配置文件 - {e}")
        print("请确保config.yaml和supply_chain.yaml在当前目录中")
    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
