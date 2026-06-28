# 实验配置与环境准备模块 - 项目索引 (Project Index)

## 🎯 项目概览

**实验配置与环境准备模块** 是一个生产级的、完整的配置管理和环境初始化系统。适用于：
- 深度学习和机器学习实验
- 时间序列预测任务
- 供应链成本评估
- 任何需要灵活参数管理的复杂实验

## 📂 文件结构与导航

### 🏃 新手入门（从这里开始）

1. **[QUICK_START.md](QUICK_START.md)** ⭐ 推荐首先阅读
   - 5分钟快速开始指南
   - 常见任务速查表
   - Mac M1 配置指南
   - 故障排除常见问题

2. **[README.md](README.md)** 📖 完整功能文档
   - 详细的功能说明和使用指南
   - 所有配置参数详解
   - 8种常见配置场景
   - 1000+行深度文档

### 💻 核心代码文件

| 文件 | 行数 | 用途 |
|------|------|------|
| **[config.py](config.py)** | 400+ | 配置解析和管理模块 |
| **[environment.py](environment.py)** | 350+ | 环境初始化模块 |
| **[__init__.py](__init__.py)** | 20 | 包初始化和API导出 |

### ⚙️ 配置文件

| 文件 | 行数 | 用途 |
|------|------|------|
| **[config.yaml](config.yaml)** | 200+ | 全局实验配置 |
| **[supply_chain.yaml](supply_chain.yaml)** | 150+ | 供应链仿真配置 |

### 📚 示例和工具

| 文件 | 用途 |
|------|------|
| **[example_usage.py](example_usage.py)** | 8个完整的使用示例 |
| **[init_check.py](init_check.py)** | 自动化初始化检查脚本 |
| **[requirements.txt](requirements.txt)** | Python依赖列表 |

### 📋 文档和总结

| 文件 | 用途 |
|------|------|
| **[module_COMPLETION_SUMMARY.md](MODULE_COMPLETION_SUMMARY.md)** | 模块完成总结和检查清单 |
| **[PROJECT_INDEX.md](PROJECT_INDEX.md)** | 本文件 - 项目导航索引 |

---

## 🚀 快速开始（三选一）

### 方式1: 自动检查 (推荐用于首次使用)

```bash
python init_check.py
```

这将自动检查：
- 文件完整性
- Python依赖
- 模块导入
- 配置加载
- 环境初始化
- 快速初始化功能

### 方式2: 一行代码快速开始

```python
from environment import quick_init

config, env_info = quick_init()
print(f"配置已加载: {config.dataset.name}")
```

### 方式3: 分步骤初始化

```python
from config import Config
from environment import setup_environment

# 第1步: 加载配置
config = Config()

# 第2步: 初始化环境
env_info = setup_environment(config)
```

---

## 📖 详细导航指南

### 我想快速了解这个模块...
👉 阅读 [QUICK_START.md](QUICK_START.md) 的前两部分 (~5分钟)

### 我想学习所有功能...
👉 完整阅读 [README.md](README.md) (~30分钟)

### 我想看实际代码示例...
👉 运行 `python example_usage.py` 查看8个示例

### 我想检查模块是否正确安装...
👉 运行 `python init_check.py` 自动验证

### 我想了解配置参数的含义...
👉 查看 [README.md](README.md) 的参数说明部分

### 我想为自己的项目配置...
👉 编辑 [config.yaml](config.yaml) 和 [supply_chain.yaml](supply_chain.yaml)

### 我遇到问题了...
👉 查看 [QUICK_START.md](QUICK_START.md) 的故障排除部分

### 我需要完整的技术细节...
👉 查看 [MODULE_COMPLETION_SUMMARY.md](MODULE_COMPLETION_SUMMARY.md)

---

## 🎯 使用场景导航

### 场景1: 需求预测项目

使用的文件：
- [config.yaml](config.yaml) - 配置数据集为 `demand-forecasting`
- [config.py](config.py) - 加载配置

```python
from config import Config
config = Config()
assert config.dataset.name == "demand-forecasting"
```

参考文档：[README.md](README.md#数据集支持)

### 场景2: Rossmann门店预测（Dataset3）

```python
config.dataset.name = "rossmann-store-sales"
config.dataset.path = "./rossmann-store-sales (2)/"
config.dataset.forecast_horizon = 14
```

说明：当前统一映射中，Rossmann 门店对应 Dataset3。

参考文档：[README.md#数据集支持)](README.md)

### 场景3: 模型架构比较

需要尝试不同的模型：CNN、LSTM、Transformer等

```python
for arch in ["cnn", "lstm", "cnn_lstm", "transformer"]:
    config.model.architecture = arch
    # 训练模型...
```

参考文档：[README.md#模型架构配置](README.md)

### 场景4: 超参数搜索

需要系统地调整学习率、batch_size等

```python
for lr in [0.001, 0.0005, 0.0001]:
    for batch_size in [32, 64, 128]:
        config.training.learning_rate = lr
        config.training.batch_size = batch_size
        # 训练并记录结果...
```

参考文档：[QUICK_START.md#任务8](QUICK_START.md)

### 场景5: 迁移学习实验

使用多源序列做迁移学习

```python
config.dataset.num_source_sequences = 5
config.dataset.use_information_sharing = True
config.model.attention.enabled = True
```

参考文档：[README.md#常见配置场景-场景3-迁移学习实验](README.md)

### 场景6: 供应链成本优化

计算不同参数下的库存成本

```python
config.supply_chain.order_quantity_Q = 50
config.supply_chain.target_service_level = 0.95
# 运行成本仿真...
```

参考文档：[README.md#供应链成本计算](README.md)

### 场景7: Mac M1优化

确保在Apple Silicon上高效运行

配置参考：[config.yaml](config.yaml) 中的 `mac_m1` 部分

文档参考：[README.md#Mac-M1-专属优化](README.md)

---

## 🔍 API 快速参考

### Config 类

```python
from config import Config

config = Config()  # 自动加载 config.yaml 和 supply_chain.yaml

# 访问配置
config.dataset.forecast_horizon
config.model.architecture
config.training.epochs
config.supply_chain.order_quantity_Q

# 使用 get() 方法（支持默认值和点记法）
horizon = config.get('dataset.forecast_horizon', 12)

# 修改配置
config.set('training.epochs', 200)
config.training.batch_size = 64

# 导出配置
config_dict = config.to_dict()
config.print_summary()
```

### 环境初始化函数

```python
from environment import (
    quick_init,                    # 一键初始化
    setup_environment,             # 完整初始化流程
    setup_logging,                 # 日志配置
    setup_reproducibility,         # 随机种子设置
    setup_mac_m1_environment,      # Mac M1优化
    check_dependencies             # 依赖检查
)

# 推荐用法
config, env_info = quick_init()

# 或分步使用
from config import Config
config = Config()
setup_reproducibility(seed=42)
env_info = setup_environment(config)
```

---

## 📊 配置参数一览

### 数据集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `name` | demand-forecasting | 数据集名称 |
| `forecast_horizon` | 12 | 预测多少个时步 |
| `num_source_sequences` | 3 | K值：源序列数 |
| `lookback_window` | 24 | 历史观察窗口 |

详见：[config.yaml](config.yaml) 的 `dataset` 部分

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `architecture` | cnn_lstm | 模型类型 |
| `cnn.num_filters` | [32,64,128] | CNN滤波器 |
| `lstm.units` | 64 | LSTM单元数 |
| `attention.enabled` | true | 注意力机制 |

详见：[config.yaml](config.yaml) 的 `model` 部分

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 100 | 训练轮数 |
| `batch_size` | 32 | 批次大小 |
| `learning_rate` | 0.001 | 学习率 |
| `optimizer` | adam | 优化器 |

详见：[config.yaml](config.yaml) 的 `training` 部分

### 供应链参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `order_quantity_Q` | 50 | 订货批量 |
| `ordering_cost_per_order` | 100 | 订货成本 |
| `target_service_level` | 0.95 | 服务水平 |

详见：[supply_chain.yaml](supply_chain.yaml)

---

## ✅ 检查清单

### 安装前检查
- [ ] 已将文件放在项目目录中
- [ ] Python版本 >= 3.8
- [ ] 网络连接可用（用于安装依赖）

### 安装步骤
```bash
# 1. 安装核心依赖
pip install numpy pandas scikit-learn pyyaml

# 2. 可选：安装深度学习库
pip install tensorflow-macos keras  # for Mac
# 或
pip install tensorflow keras  # for Linux/Windows

# 3. 验证安装
python init_check.py
```

### 使用前检查
- [ ] 运行 `python init_check.py` 通过所有检查
- [ ] 能够导入 Config 和 quick_init
- [ ] config.yaml 和 supply_chain.yaml 存在且有效

### 首次使用
- [ ] 阅读 [QUICK_START.md](QUICK_START.md)
- [ ] 运行 `python example_usage.py`
- [ ] 尝试一个简单的配置修改

---

## 🆘 常见问题

### Q: 报错 "No module named 'yaml'"
A: 安装 PyYAML: `pip install pyyaml`

### Q: 报错 "Config file not found"
A: 确保在正确的目录，config.yaml 和 supply_chain.yaml 应该与脚本在同一目录

### Q: 为什么有这么多配置？能简化吗?
A: 配置文件设计用于最大灵活性。简单使用时直接用默认值即可。

### Q: Mac M1 上 TensorFlow 装不上?
A: 使用 `pip install tensorflow-macos` 而不是标准 tensorflow

### Q: 如何在项目中集成这个模块?
A: 将所有文件复制到你的项目目录，然后：
```python
from environment import quick_init
config, env_info = quick_init()
```

详见：[QUICK_START.md#故障排除](QUICK_START.md)

---

## 📈 推荐学习路径

```
初学者
  │
  ├─→ QUICK_START.md (5 min)
  │   └─→ 运行 init_check.py (2 min)
  │       └─→ 运行 example_usage.py (3 min)
  │
  └─→ README.md (阅读前2章，15 min)
      └─→ 尝试修改 config.yaml (10 min)

中级用户
  │
  ├─→ 完读 README.md (30 min)
  │   └─→ 学习所有8个示例 (example_usage.py)
  │       └─→ 为自己的项目编写初始化脚本
  │
  └─→ 阅读源代码 (config.py, environment.py)
      └─→ 理解参数验证逻辑

高级用户
  │
  ├─→ 阅读 MODULE_COMPLETION_SUMMARY.md
  │   └─→ 了解内部架构设计
  │       └─→ 扩展功能（自定义参数、验证器等）
  │
  └─→ 为特定场景优化配置
```

---

## 📞 获取帮助

### 诊断步骤
1. 运行 `python init_check.py` 获得自动诊断
2. 查看脚本输出的具体错误信息
3. 在 [QUICK_START.md](QUICK_START.md) 中查找类似问题

### 如果问题仍未解决
1. 确认 Python 版本：`python --version`
2. 列出已安装包：`pip list | grep -E "numpy|pandas|tensorflow|pyyaml"`
3. 检查文件存在性：`ls -la *.py *.yaml`

### 常见问题已有答案在：
- [QUICK_START.md#故障排除](QUICK_START.md#故障排除)
- [README.md#错误排查](README.md#错误排查)
- [MODULE_COMPLETION_SUMMARY.md#🔮-扩展方向](MODULE_COMPLETION_SUMMARY.md)

---

## 📝 文件概览总表

| 文件名 | 用途 | 新手 | 中级 | 高级 | 优先级 |
|--------|------|------|------|------|--------|
| QUICK_START.md | 快速入门 | ⭐⭐⭐ | ⭐ | 无 | 1 |
| README.md | 完整文档 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 2 |
| config.py | 配置模块 | 无 | ⭐⭐ | ⭐⭐⭐ | 3 |
| environment.py | 环境模块 | 无 | ⭐⭐ | ⭐⭐⭐ | 3 |
| example_usage.py | 代码示例 | ⭐⭐⭐ | ⭐⭐ | 无 | 4 |
| init_check.py | 检查脚本 | ⭐⭐⭐ | ⭐ | 无 | 4 |
| config.yaml | 配置文件 | ⭐⭐ | ⭐⭐ | ⭐ | 5 |
| supply_chain.yaml | 配置文件 | ⭐ | ⭐⭐ | ⭐ | 5 |
| MODULE_COMPLETION_SUMMARY.md | 技术总结 | 无 | ⭐ | ⭐⭐⭐ | 6 |

---

## 🎓 学习资源总览

### 文档资源
- **入门级**: QUICK_START.md (300行, 5分钟阅读)
- **中级**: README.md (800行, 30分钟阅读)
- **高级**: MODULE_COMPLETION_SUMMARY.md (400行, 技术细节)

### 代码资源
- **入门**: example_usage.py (8个简单示例)
- **中级**: config.py + environment.py (源代码)
- **高级**: 自定义扩展或集成

### 工具资源
- **诊断**: init_check.py (自动检查脚本)
- **测试**: run `python example_usage.py`

---

## 🏁 总结

这个项目提供了：
✅ **灵活的配置管理** - YAML配置 + Python对象  
✅ **完整的环境初始化** - 依赖检查、随机种子、日志、设备配置  
✅ **Mac M1 优先支持** - 针对Apple Silicon的优化  
✅ **详尽的文档** - 1000+行文档 + 8个示例  
✅ **易于使用** - 一行代码快速开始  

**立即开始：** `python init_check.py`

---

**最后更新**: 2026年3月13日  
**模块版本**: 1.0.0  
**状态**: ✅ 完成并可用

欢迎使用！如有问题，请参考相应的文档部分。😊
