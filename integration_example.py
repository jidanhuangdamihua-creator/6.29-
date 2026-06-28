#!/usr/bin/env python3
"""
实际项目集成示例 (Real-World Integration Example)

这个脚本演示如何在实际的机器学习项目中集成配置和环境模块。
这是一个"模板"脚本，您可以复制和修改它用于自己的项目。

用法:
    python integration_example.py
"""

import sys
from pathlib import Path

# 步骤1: 导入配置和环境模块
print("=" * 70)
print("实验系统自动化初始化 - 集成示例")
print("=" * 70)

try:
    from environment import quick_init
    from config import Config
    print("\n✓ 模块导入成功")
except ImportError as e:
    print(f"\n✗ 模块导入失败: {e}")
    print("请确保 config.py 和 environment.py 在当前目录")
    sys.exit(1)


def initialize_experiment():
    """初始化实验"""
    print("\n【步骤1】初始化实验配置和环境...")
    
    try:
        # 这一行完成所有初始化！
        config, env_info = quick_init()
        
        print("✓ 初始化完成")
        return config, env_info
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        sys.exit(1)


def setup_custom_parameters(config):
    """根据需要自定义参数"""
    print("\n【步骤2】自定义参数...")
    
    # 例1: 针对快速原型开发的配置
    mode = "prototype"  # 可选: prototype, standard, production
    
    if mode == "prototype":
        print("  选择模式: 快速原型开发")
        config.training.epochs = 10
        config.training.batch_size = 64
        config.dataset.train_ratio = 0.5
        print("  ✓ epochs=10, batch_size=64, 使用50%数据")
    
    elif mode == "standard":
        print("  选择模式: 标准训练")
        config.training.epochs = 100
        config.training.batch_size = 32
        print("  ✓ epochs=100, batch_size=32")
    
    elif mode == "production":
        print("  选择模式: 生产级")
        config.training.epochs = 200
        config.training.batch_size = 16
        print("  ✓ epochs=200, batch_size=16")


def verify_data_availability(config):
    """验证数据集可用性"""
    print("\n【步骤3】检查数据集...")
    
    dataset_path = Path(config.dataset.path)
    
    if dataset_path.exists():
        print(f"✓ 数据集路径存在: {dataset_path}")
        
        # 列出数据集文件
        csv_files = list(dataset_path.glob("*.csv"))
        if csv_files:
            print(f"  找到 {len(csv_files)} 个CSV文件:")
            for csv_file in csv_files:
                print(f"    - {csv_file.name}")
        else:
            print("  ⚠ 未找到CSV文件")
    else:
        print(f"⚠ 数据集路径不存在: {dataset_path}")
        print("  请下载数据或检查配置中的路径")


def print_experiment_summary(config, env_info):
    """打印实验摘要"""
    print("\n【步骤4】实验配置摘要")
    print("-" * 70)
    
    # 数据集信息
    print("\n【数据集配置】")
    print(f"  数据集名称:        {config.dataset.name}")
    print(f"  数据集路径:        {config.dataset.path}")
    print(f"  预测步长:          {config.dataset.forecast_horizon}")
    print(f"  源序列数K:         {config.dataset.num_source_sequences}")
    print(f"  信息共享模式:      {config.dataset.use_information_sharing}")
    
    # 模型信息
    print("\n【模型配置】")
    print(f"  模型架构:          {config.model.architecture}")
    if config.model.architecture in ["cnn", "cnn_lstm"]:
        print(f"  CNN滤波器:         {config.model.cnn.num_filters}")
    if config.model.architecture in ["lstm", "cnn_lstm"]:
        print(f"  LSTM单元:          {config.model.lstm.units}")
    print(f"  注意力机制:        {config.model.attention.enabled}")
    
    # 训练信息
    print("\n【训练配置】")
    print(f"  Epochs:            {config.training.epochs}")
    print(f"  Batch Size:        {config.training.batch_size}")
    print(f"  学习率:            {config.training.learning_rate}")
    print(f"  优化器:            {config.training.optimizer}")
    print(f"  早停:              {config.training.early_stopping_enabled}")
    
    # 环境信息
    print("\n【环境信息】")
    print(f"  Python版本:        {sys.version.split()[0]}")
    print(f"  随机种子:          {env_info.get('random_seed')}")
    print(f"  计算设备:          {env_info['device_config'].get('device')}")
    
    # 依赖检查
    deps = env_info['dependencies']
    status = "✓ 通过" if deps['all_required_met'] else "✗ 失败"
    print(f"  依赖库检查:        {status}")
    
    # 供应链信息（如果有）
    if config.supply_chain:
        print("\n【供应链参数】")
        print(f"  订货批量Q:         {config.supply_chain.order_quantity_Q}")
        print(f"  订货成本:          ¥{config.supply_chain.ordering_cost_per_order}")
        print(f"  采购成本:          ¥{config.supply_chain.unit_acquisition_cost}")
        print(f"  目标服务水平:      {config.supply_chain.target_service_level*100}%")
    
    print("-" * 70)


def demonstrate_flexibility(config):
    """演示配置的灵活性"""
    print("\n【步骤5】演示配置灵活性...")
    
    # 运行多个实验配置
    print("\n  支持的预测步长配置:")
    for horizon in [7, 14, 24]:
        config.dataset.forecast_horizon = horizon
        print(f"    - {horizon} 步: 用于 {horizon} {'天' if horizon <= 30 else '周'} 预测")
    
    print("\n  支持的模型对比:")
    architectures = {
        'cnn': 'CNN - 轻量级结构',
        'lstm': 'LSTM - 序列学习',
        'cnn_lstm': 'CNN+LSTM - 混合架构',
        'transformer': 'Transformer - 最先进'
    }
    for arch, desc in architectures.items():
        print(f"    - {arch:12} - {desc}")
    
    print("\n  支持的数据集:")
    datasets = {
        'demand-forecasting': 'Dataset1 - 需求预测挑战赛',
        'italian-pasta-demand': 'Dataset2 - 意大利面需求',
        'rossmann-store-sales': 'Dataset3 - Rossmann门店'
    }
    for name, desc in datasets.items():
        print(f"    - {name:25} - {desc}")


def show_usage_tips(config):
    """显示使用提示"""
    print("\n【步骤6】使用提示...")
    
    print("\n  访问配置的三种方式:")
    print("    1. 直接属性: config.training.epochs")
    print("    2. get()方法: config.get('training.epochs', 100)")
    print("    3. 转字典:   config.to_dict()")
    
    print("\n  修改配置的两种方式:")
    print("    1. setattr:   config.set('training.epochs', 150)")
    print("    2. 直接赋值: config.training.epochs = 150")
    
    print("\n  常用操作:")
    print("    - 查看摘要:   config.print_summary()")
    print("    - 导出配置:   config_dict = config.to_dict()")
    print("    - 快速初始化: config, env = quick_init()")
    print("    - 依赖检查:   python init_check.py")


def main():
    """主函数 - 集成示例的完整流程"""
    
    try:
        # 初始化
        config, env_info = initialize_experiment()
        
        # 自定义参数
        setup_custom_parameters(config)
        
        # 验证数据
        verify_data_availability(config)
        
        # 打印摘要
        print_experiment_summary(config, env_info)
        
        # 演示灵活性
        demonstrate_flexibility(config)
        
        # 显示提示
        show_usage_tips(config)
        
        # 最终确认
        print("\n" + "=" * 70)
        print("✨ 实验系统初始化完成！")
        print("=" * 70)
        print("\n现在您可以:")
        print("  1. 修改配置参数")
        print("  2. 加载数据集")
        print("  3. 构建和训练模型")
        print("  4. 保存结果和模型权重")
        print("\n祝您实验顺利！🚀")
        print("=" * 70)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n执行被中断")
        return 1
    except Exception as e:
        print(f"\n✗ 出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
# 额外的辅助函数示例 - 可在实际项目中使用
# ============================================================================

def load_data(config):
    """
    根据配置加载数据的示例函数
    
    在实际项目中，这个函数会:
    1. 读取 config.dataset.path 中的数据
    2. 执行数据预处理
    3. 按 config.dataset.train_ratio 等比例划分数据
    4. 返回可用于模型训练的数据集
    """
    import pandas as pd
    from pathlib import Path
    
    print(f"Loading data from {config.dataset.path}...")
    # 例如: df = pd.read_csv(config.dataset.path + "/train.csv")
    # 返回 train_data, val_data, test_data
    
    pass


def build_model(config):
    """
    根据配置构建模型的示例函数
    
    在实际项目中，这个函数会:
    1. 根据 config.model.architecture 选择模型
    2. 使用 config.model 中的参数配置模型
    3. 编译模型（使用 config.training 中的参数）
    4. 返回编译后的模型
    """
    # try:
    #     import tensorflow as tf
    #     
    #     if config.model.architecture == "lstm":
    #         model = tf.keras.Sequential([
    #             tf.keras.layers.LSTM(
    #                 units=config.model.lstm.units,
    #                 input_shape=(config.dataset.lookback_window, 1)
    #             ),
    #             tf.keras.layers.Dense(config.dataset.forecast_horizon)
    #         ])
    #     
    #     model.compile(
    #         optimizer=config.training.optimizer,
    #         loss=config.training.loss,
    #         metrics=config.training.metrics
    #     )
    #     
    #     return model
    # except ImportError:
    #     print("TensorFlow not installed")
    
    pass


def train_model(model, config, train_data, val_data):
    """
    根据配置训练模型的示例函数
    
    在实际项目中，这个函数会:
    1. 使用 config.training 中的参数（epochs, batch_size等）
    2. 使用 config.training.early_stopping_* 进行早停
    3. 使用 config.training.lr_scheduler_* 进行学习率调度
    4. 返回训练历史和训练好的模型
    """
    # callbacks = []
    # 
    # if config.training.early_stopping_enabled:
    #     callbacks.append(EarlyStopping(
    #         monitor='val_loss',
    #         patience=config.training.early_stopping_patience,
    #         restore_best_weights=True
    #     ))
    #
    # history = model.fit(
    #     train_data,
    #     validation_data=val_data,
    #     epochs=config.training.epochs,
    #     batch_size=config.training.batch_size,
    #     callbacks=callbacks
    # )
    #
    # return history
    
    pass


if __name__ == "__main__":
    sys.exit(main())
