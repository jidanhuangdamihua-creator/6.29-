#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速参考卡片 (Quick Reference Card) - 新功能速查
最后更新: 2026-03-13
"""

# ============================================================================
# 📌 command-line 快速命令
# ============================================================================

"""
# 验证系统配置
python verify_bootstrap.py

# 指定配置文件
python verify_bootstrap.py --config config/my_config.yaml

# 静默模式（只输出最终结果）
python verify_bootstrap.py --quiet

# 查看帮助
python verify_bootstrap.py --help
"""

# ============================================================================
# 🐍 Python API 速查
# ============================================================================

"""
# [1] 加载配置
from config import Config
config = Config()

# [2] 配置交叉字段验证
errors = config._validate_cross_fields()  # 返回: List[str]
if errors:
    print(f"验证失败: {errors}")

config.validate_and_report()  # 自动验证+打印报告

# [3] 保存配置快照（用于复现）
snapshot_path = config.save_config_snapshot()
# 输出: ./outputs/run_snapshots/config_snapshot_20260313_153045.json

# [4] 访问实验矩阵
if config.experiment_matrix and config.experiment_matrix.enabled:
    datasets = config.experiment_matrix.datasets_to_run
    methods = config.experiment_matrix.enabled_methods
    
# [5] 访问依赖库版本
from config import Config
config = Config()
core_pkgs = config.dependencies.get_core_packages()
tf_spec = config.dependencies.get_tensorflow_mac_m1_spec()

# [6] 设置日志系统
from environment import setup_logging
setup_logging(
    log_file="logs/experiment.log",
    console_level=logging.INFO,  # 控制台日志级别
    file_level=logging.DEBUG      # 文件日志级别
)

# [7] 执行系统干运行检查
from environment import perform_dry_run
from config import Config

config = Config()
result = perform_dry_run(
    config=config,
    config_file="config/experiment_config.yaml",
    supply_chain_file="config/supply_chain_config.yaml"
)

if result['status'] == 'pass':
    print("✓ 系统就绪")
elif result['status'] == 'warn':
    print(f"⚠ 警告: {result['warnings']}")
else:
    print(f"✗ 错误: {result['errors']}")

# [8] 保存环境快照
from environment import save_environment_snapshot
from config import Config

config = Config()
snapshot_path = save_environment_snapshot(
    config=config,
    env_info={},
    snapshot_dir="./outputs/env_snapshots"
)
# 输出: ./outputs/env_snapshots/ENV_20260313_153045.json

# [9] 编程方式运行验证
from verify_bootstrap import BootstrapVerifier

verifier = BootstrapVerifier(
    config_file="config/experiment_config.yaml",
    quiet=False
)
exit_code = verifier.run()
# 返回: 0 (PASS), 1 (WARN), 2 (FAIL)

print(f"状态: {verifier.results['final_status']}")
print(f"错误: {verifier.results['errors']}")
print(f"快照: {verifier.results['snapshots']}")
"""

# ============================================================================
# 📋 config.yaml 配置示例片段
# ============================================================================

"""
# 实验矩阵配置
experiment_matrix:
  enabled: true
  datasets_to_run:
    - demand-forecasting
    - italian-pasta-demand
  source_counts: [2, 3, 5]
  horizons: [7, 12, 24]
  scenarios:
    standard:
      enabled: true
      epochs: 100
      learning_rate: 0.001
    aggressive:
      enabled: false
      epochs: 150
      learning_rate: 0.01
  enabled_methods:
    - lstm
    - gru
    - transformer
  run_control:
    max_parallel_jobs: 4
    enable_checkpointing: true
    snapshot_dir: ./outputs/run_snapshots

# 依赖库版本说明
dependencies:
  python_version: "3.9"
  core:
    numpy: ">=1.21.0,<1.24.0"
    pandas: ">=1.3.0,<2.0.0"
    scikit-learn: ">=0.24.0,<1.4.0"
    pyyaml: ">=5.3,<6.1"
  deeplearning:
    # 选项 A: 标准 TensorFlow (Linux/Windows)
    # tensorflow: ">=2.10.0,<2.14.0"
    # 选项 B: Mac M1 版本 (推荐 Mac 用户)
    tensorflow-macos: ">=2.10.0,<2.14.0"
  visualization:
    matplotlib: ">=3.5.0,<3.8.0"
    seaborn: ">=0.12.0,<0.14.0"
"""

# ============================================================================
# 📊 常见错误及解决方案
# ============================================================================

"""
错误1: K < 2 且启用了 information_sharing
─────────────────────────────
验证错误: K < 2 但启用了 information_sharing (K=1, need >=2)

解决:
  在 config.yaml 中修改:
  dataset:
    num_source_sequences: 2  # 改成 >= 2
    use_information_sharing: true


错误2: scenario 权重和不等于 1.0
─────────────────────────────
验证错误: scenario 权重和应为 1.0，实际为 0.8

解决:
  在 config.yaml 中调整:
  experiment_matrix:
    scenarios:
      standard:
        enabled: true
        weight: 0.6
      aggressive:
        enabled: true
        weight: 0.4  # Total = 1.0


错误3: TensorFlow 不可用 (Mac M1)
─────────────────────────────
警告: TensorFlow 不可用

解决:
  # 卸载标准版本
  pip uninstall tensorflow
  
  # 安装 Mac 版本
  pip install tensorflow-macos
  
  # 验证
  python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"


错误4: 输出目录不可写
─────────────────────────────
错误: 输出目录不可写: results/

解决:
  mkdir -p results models figures logs outputs
  chmod 755 results models figures logs outputs


错误5: 配置文件不存在
─────────────────────────────
错误: 配置文件不存在: config/experiment_config.yaml

解决:
  # 检查当前目录
  ls -la config/
  
  # 指定正确的路径
  python verify_bootstrap.py --config config/experiment_config.yaml
"""

# ============================================================================
# 🔄 数据结构速查
# ============================================================================

"""
ExperimentMatrixConfig:
  ├─ enabled: bool
  ├─ datasets_to_run: List[str]
  ├─ source_counts: List[int]
  ├─ horizons: List[int]
  ├─ scenarios: Dict[str, ScenarioConfig]
  │  └─ enabled, epochs, learning_rate, ...
  ├─ enabled_methods: List[str]
  └─ run_control: Dict[str, Any]

DependencyConfig:
  ├─ python_version: str
  ├─ core: Dict[str, str]  (numpy, pandas, ...)
  ├─ deeplearning: Dict[str, str]  (tensorflow, keras, ...)
  ├─ visualization: Dict[str, str]  (matplotlib, seaborn, ...)
  ├─ get_core_packages() -> Dict[str, str]
  └─ get_tensorflow_mac_m1_spec() -> str

perform_dry_run() 返回:
  {
    'status': 'pass' | 'warn' | 'fail',
    'checks': {
      'config_files': {...},
      'config_validation': {...},
      'dependencies': {...},
      'output_directories': {...},
      'platform_info': {...},
      'tensorflow': {...},
    },
    'warnings': List[str],
    'errors': List[str],
    'timestamp': str,
  }

save_config_snapshot() 返回: str (路径)
save_environment_snapshot() 返回: str (路径)

Config._validate_cross_fields() 返回: List[str] (错误列表)
"""

# ============================================================================
# 🔍 检查列表 (Checklist)
# ============================================================================

"""
□ 系统初始化
  □ 创建虚拟环境
  □ 安装依赖: pip install -r requirements.txt
  □ 运行验证: python verify_bootstrap.py

□ 配置准备
  □ 创建/修改 config/experiment_config.yaml
  □ 检查交叉字段: config._validate_cross_fields()
  □ 查看配置摘要: config.print_summary()

□ 实验执行前
  □ 执行干运行检查: perform_dry_run()
  □ 保存配置快照: config.save_config_snapshot()
  □ 保存环境快照: save_environment_snapshot()
  □ 设置日志: setup_logging()

□ 实验结束后
  □ 查看配置快照（复现）
  □ 查看环境快照（诊断）
  □ 查看验证结果日志
"""

# ============================================================================
# 📈 性能优化建议
# ============================================================================

"""
Mac M1 优化:
  □ 使用 tensorflow-macos (比标准 tensorflow 快 2-3x)
  □ 检查 Metal 加速是否启用
  □ 使用 conda-forge 安装 numpy (ARM64 native)

配置验证优化:
  □ 在实验开始前调用 validate_and_report()
  □ 快照文件纳入版本控制 (git)
  □ 定期对比环境快照以检测包更新

日志优化:
  □ 设置 console_level=INFO, file_level=DEBUG
  □ 定期压缩 log 文件 (gzip)
  □ 使用 log rotation (每天/每 100MB)
"""

# ============================================================================
# 🎯 常见工作流
# ============================================================================

"""
工作流1: 首次项目设置
───────────────────
1. pip install -r requirements.txt
2. python verify_bootstrap.py
3. 检查所有检查是否通过 (PASS/WARN)
4. 修复任何错误 (FAIL)
5. python verify_bootstrap.py (再次验证)
6. 开始实验

工作流2: 多配置实验
───────────────────
1. 准备 config/exp1.yaml, config/exp2.yaml, ...
2. for config in config/exp_*.yaml:
     python verify_bootstrap.py --config $config
3. 检查所有配置通过
4. 运行实验
5. 对比不同配置的快照

工作流3: CI/CD 集成
───────────────────
1. 在 .github/workflows/test.yml 中添加:
   - run: python verify_bootstrap.py
2. 推送代码前验证
3. PR 自动检查配置有效性

工作流4: 长期运行监控
───────────────────
1. 定期运行: python verify_bootstrap.py
2. 将结果保存到文件
3. 对比多个快照检测包升级问题
4. 提前发现兼容性问题
"""

# ============================================================================
# 📞 快速问题排查
# ============================================================================

"""
Q: 我的项目应该支持哪个 Python 版本？
A: 推荐 Python 3.9 或 3.10
   - 3.8: 功能齐全但可能遇到库兼容性问题
   - 3.9+: 全面支持，推荐
   检查: python --version

Q: Mac M1 用户应该做什么？
A: 三个关键步骤:
   1. 使用 conda 而不是 pip (更好的二进制支持)
   2. 安装 tensorflow-macos 而不是 tensorflow
   3. 运行 verify_bootstrap.py 检查是否检测到 ARM64
   检查: python verify_bootstrap.py | grep -i "arm64\|m1"

Q: 我的配置验证失败了，怎么办？
A: 按以下步骤:
   1. 查看错误信息: config._validate_cross_fields()
   2. 打开 config.yaml 按错误提示修改
   3. 重新加载配置
   4. 再次验证

Q: 如何快速保存/复现实验配置？
A: 使用快照:
   snapshot = config.save_config_snapshot()  # 保存
   # 后续从 JSON 恢复配置

Q: TensorFlow 总是有警告，这是严重问题吗？
A: 不一定。检查 perform_dry_run() 的返回值:
   - status='warn': 可运行但有警告（通常不影响）
   - status='fail': 不可运行，必须修复

Q: 我想在自己的代码中集成验证脚本怎么办？
A: 使用 BootstrapVerifier 类:
   from verify_bootstrap import BootstrapVerifier
   verifier = BootstrapVerifier()
   if verifier.run() == 0:  # PASS
       main()  # 开始实验
   else:
       sys.exit(1)  # 停止
"""

# ============================================================================
# 🎓 关键概念解释
# ============================================================================

"""
1. 静态配置 vs 运行时状态
   ────────────────────────
   静态配置: dataset, model, training (加载后不变)
   运行时状态: 快照路径、加载时间戳、验证错误
   好处: 配置对象轻量，状态分离清晰

2. 交叉字段验证
   ────────────────────────
   不仅检查单个字段有效性，还检查字段间的关系
   例: K=1 且 use_information_sharing=true 是非法的

3. 快照的意义
   ────────────────────────
   配置快照: 记录实验的配置，便于复现
   环境快照: 记录运行环境，便于诊断
   都用 JSON 格式，便于版本化和对比

4. Mac M1 为什么特殊？
   ────────────────────────
   ARM64 架构与 x86_64 二进制不兼容
   许多库需要特殊编译版本 (tensorflow-macos)
   性能差异大 (tensorflow-macos 2-3x 更快)

5. 干运行检查的意义
   ────────────────────────
   在实验开始前检查所有前提条件
   快速失败避免浪费时间/计算资源
   6 大检查类别覆盖所有关键方面
"""

# ============================================================================

print(__doc__)
