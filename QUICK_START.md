# 快速开始指南 (Quick Start Guide)

## 5分钟快速开始

### 步骤1: 安装依赖（1分钟）

```bash
# 确保你在项目目录中
cd /Users/ming/Desktop/全新实验的副本

# 安装核心依赖
pip install numpy pandas scikit-learn pyyaml

# 可选：安装深度学习依赖（Mac M1专用）
pip install tensorflow-macos keras
```

### 步骤2: 使用配置（1分钟）

```python
# 最简单的方式：一行代码
from environment import quick_init

config, env_info = quick_init()

# 现在就可以使用
print(f"数据集: {config.dataset.name}")
print(f"预测步长: {config.dataset.forecast_horizon}")
print(f"模型: {config.model.architecture}")
print(f"训练epochs: {config.training.epochs}")
```

### 步骤3: 修改配置（1分钟）

```python
from config import Config

config = Config()

# 方式1: 直接修改属性
config.training.epochs = 200
config.training.batch_size = 64

# 方式2: 使用set()方法
config.set('dataset.forecast_horizon', 24)
config.set('model.architecture', 'transformer')

# 验证修改
print(config.get('training.epochs'))  # 输出: 200
```

### 步骤4: 完整初始化（1分钟）

```python
from environment import quick_init

# 完整初始化：配置加载 + 环境设置 + 依赖检查 + 随机种子设置
config, env_info = quick_init()

# 检查初始化状态
print(f"依赖检查: {'通过' if env_info['dependencies']['all_required_met'] else '失败'}")
print(f"计算设备: {env_info['device_config']['device']}")
print(f"随机种子: {env_info['random_seed']}")
```

### 步骤5: 运行示例（1分钟）

```bash
# 运行完整示例脚本
python example_usage.py
```

## 常见任务速查表

### 任务1: 加载数据集配置

```python
from config import Config

config = Config()

# 查看当前数据集
print(config.dataset.name)      # 输出: demand-forecasting
print(config.dataset.path)      # 输出: ./demand-forecasting-kernels-only (1)/

# 切换到Rossmann门店数据集（当前统一映射为 Dataset3）
config.dataset.name = "rossmann-store-sales"
config.dataset.path = "./rossmann-store-sales (2)/"
config.dataset.forecast_horizon = 14
```

### 任务2: 配置模型参数

```python
# 查看当前模型
print(config.model.architecture)  # 输出: cnn_lstm

# 修改为LSTM模型
config.model.architecture = "lstm"
config.model.lstm.units = 128
config.model.lstm.num_layers = 3

# 修改为Transformer
config.model.architecture = "transformer"
```

### 任务3: 配置训练参数

```python
# 查看训练配置
print(config.training.epochs)         # 输出: 100
print(config.training.learning_rate)  # 输出: 0.001

# 修改训练参数
config.training.epochs = 150
config.training.batch_size = 16
config.training.learning_rate = 0.0005
config.training.optimizer = "rmsprop"
```

### 任务4: 配置供应链参数

```python
# 查看供应链配置
print(config.supply_chain.order_quantity_Q)      # 输出: 50
print(config.supply_chain.target_service_level)  # 输出: 0.95

# 修改供应链参数（优化库存）
config.supply_chain.order_quantity_Q = 75           # 增加订货量
config.supply_chain.target_service_level = 0.98    # 提高服务水平
config.supply_chain.unit_acquisition_cost = 45     # 降低采购成本
```

### 任务5: 设置随机种子

```python
from environment import setup_reproducibility

# 设置随机种子确保可复现
setup_reproducibility(seed=42)

# 之后所有随机操作将使用相同的种子
```

### 任务6: 进行依赖检查

```python
from environment import check_dependencies

all_met, status = check_dependencies()

if all_met:
    print("所有依赖已安装！")
else:
    print("缺少以下依赖:")
    for pkg, (installed, version) in status.items():
        if not installed:
            print(f"  - {pkg}")
```

### 任务7: 完整环境初始化

```python
from environment import setup_environment
from config import Config

# 加载配置
config = Config()

# 完整初始化
env_info = setup_environment(config, verbose=True)

# 环境初始化后可以安全使用TensorFlow/NumPy/Pandas
import tensorflow as tf
import numpy as np
import pandas as pd
```

### 任务8: 访问配置文件

```python
from config import Config

config = Config()

# 获取完整配置字典（用于保存或输出）
config_dict = config.to_dict()

# 打印配置摘要
config.print_summary()

# 访问嵌套参数（用点记法）
horizon = config.get('dataset.forecast_horizon')           # 12
batch = config.get('training.batch_size', 32)             # 32
missing = config.get('nonexistent.key', 'default')        # 'default'
```

## 配置文件位置

所有配置文件都在项目根目录：

```
/Users/ming/Desktop/全新实验的副本/
├── config.yaml                    # 全局实验配置
├── supply_chain.yaml              # 供应链配置
├── config.py                      # 配置模块
└── environment.py                 # 环境初始化模块
```

## 关键概念

### 数据集参数含义

| 参数 | 含义 |
|------|------|
| `forecast_horizon` | 需要预测多少个时间步入未来 |
| `lookback_window` | 用多少个历史时间步来做预测 |
| `num_source_sequences (K)` | 使用多少个源序列进行迁移学习 |
| `use_information_sharing` | 是否在多源序列间共享信息 |

### 模型参数含义

| 参数 | 含义 |
|------|------|
| `architecture` | 选择的神经网络架构 (cnn/lstm/cnn_lstm等) |
| `num_filters` | CNN各层的卷积滤波器数 |
| `lstm.units` | LSTM隐层神经元数 |
| `attention.enabled` | 是否使用注意力机制 |

### 成本参数含义

| 参数 | 含义 |
|------|------|
| `order_quantity_Q` | 每次订货的数量 |
| `ordering_cost_per_order` | 每次下单的费用 |
| `unit_acquisition_cost` | 每单位产品的采购价格 |
| `holding_cost_daily` | 每单位产品每天的存储费用 |

## Mac M1 特定配置

### 推荐设置

```yaml
mac_m1:
  use_metal_acceleration: false      # 暂不使用（实验性）
  tensorflow_force_cpu: true         # 使用CPU
  max_threads: 4                     # 线程数（可根据自己Mac调整）
```

### 性能优化查看

```python
from environment import setup_mac_m1_environment

device_info = setup_mac_m1_environment(
    use_cpu=True,
    use_metal=False,
    max_threads=4
)

print(f"计算设备: {device_info['device']}")
print(f"TensorFlow版本: {device_info['tensorflow_version']}")
```

## 故障排除

### 问题1: "No module named 'yaml'"

```bash
pip install pyyaml
```

### 问题2: "Config file not found"

确保您在正确的目录中：
```bash
pwd  # 应该显示: /Users/ming/Desktop/全新实验的副本

ls config.yaml  # 应该存在
ls supply_chain.yaml  # 应该存在
```

### 问题3: TensorFlow 相关错误

```bash
# 卸载旧版本
pip uninstall tensorflow -y

# 重新安装（Mac M1）
pip install tensorflow-macos
```

### 问题4: 导入错误

确保你在项目目录中，或添加到Python路径：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
```

## 下一步

1. **阅读完整文档**: [README.md](README.md)
2. **运行所有示例**: `python example_usage.py`
3. **修改配置文件**: 编辑 `config.yaml` 和 `supply_chain.yaml`
4. **集成到你的代码**: 在你的训练脚本中导入 Config 和 setup_environment

## 常用命令集

```bash
# 检查依赖
python -c "from environment import check_dependencies; check_dependencies()"

# 快速初始化
python -c "from environment import quick_init; config, env = quick_init()"

# 运行示例
python example_usage.py

# 查看配置摘要
python -c "from config import Config; Config().print_summary()"
```

---

**提示**: 如果你是第一次使用，建议从 `python example_usage.py` 开始，了解所有的功能。
