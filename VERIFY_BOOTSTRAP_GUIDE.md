# verify_bootstrap.py 快速入门指南

## 🚀 是什么？

`verify_bootstrap.py` 是一个**一条命令验证整个系统**的脚本。它检查：
- ✓ 配置文件是否存在和有效
- ✓ 所有依赖库是否已安装
- ✓ 输出目录是否可写
- ✓ 平台和 Python 版本是否兼容
- ✓ TensorFlow 是否就绪
- 最后生成配置和环境快照供重复使用

## 📋 使用方式

### 基础用法（推荐）

```bash
python verify_bootstrap.py
```

**预期输出**:
```
======================================================================
🚀 实验配置系统引导验证
======================================================================
配置文件: config/experiment_config.yaml
======================================================================

[15:30:45] INFO     - 步骤 1/5: 加载配置文件
...
[15:30:46] INFO     - ✓ 配置加载成功

  ✓ 配置文件加载                           PASS      config/experiment_config.yaml

======================================================================
步骤 5/5: 最终报告
======================================================================

最终状态: PASS
```

### 指定配置文件

```bash
python verify_bootstrap.py --config config/custom_config.yaml
```

### 静默模式（只显示最终结果）

```bash
python verify_bootstrap.py --quiet
```

### 帮助信息

```bash
python verify_bootstrap.py --help
```

---

## 📊 输出解读

### 1. 每步的检查项

脚本分 5 步执行，每步有多个检查：

**步骤 1: 加载配置文件**
- 检查 config_file 是否存在
- 解析 YAML
- 实例化 Config 对象

**步骤 2: 验证配置交叉字段**
- K < 2 且启用 information_sharing → 错误
- scenario 权重和 ≠ 1.0 → 警告
- 模型/训练配置匹配

**步骤 3: 执行系统级检查**
- 配置文件存在性
- 依赖库可用性（numpy, pandas, tensorflow 等）
- 输出目录可写性
- 平台信息（系统、CPU、Python 版本）
- TensorFlow 状态

**步骤 4: 保存快照**
- 配置快照 → `./outputs/run_snapshots/config_snapshot_YYYYMMDD_HHMMSS.json`
- 环境快照 → `./outputs/env_snapshots/ENV_YYYYMMDD_HHMMSS.json`

**步骤 5: 最终报告**
- 汇总所有检查结果
- 给出建议

### 2. 最终状态

- **✓ PASS**: 一切就绪，可以开始实验
- **⚠ WARN**: 有警告但可以运行（建议修复）
- **✗ FAIL**: 有致命错误，无法继续

### 3. 符号含义

| 符号 | 含义 |
|------|------|
| ✓ | 通过/成功 |
| ✗ | 失败/错误 |
| ⚠ | 警告 |
| 📊 | 摘要信息 |
| 🔍 | 检查结果 |
| 💾 | 快照文件 |
| 💡 | 建议 |

---

## 🔍 常见问题排查

### Q: 配置文件加载失败

**症状**: `✗ 配置文件加载 FAIL`

**排查**:
```bash
# 1. 检查文件是否存在
ls -la config/experiment_config.yaml

# 2. 检查 YAML 语法
python3 -c "import yaml; yaml.safe_load(open('config/experiment_config.yaml'))"

# 3. 指定正确的路径
python verify_bootstrap.py --config config/experiment_config.yaml
```

### Q: TensorFlow 不可用

**症状**: `⚠ 检查: tensorflow WARN (可用: False)`

**解决** (Mac M1):
```bash
# 移除标准 tensorflow
pip uninstall tensorflow

# 安装 Mac 版本
pip install tensorflow-macos

# 验证
python -c "import tensorflow as tf; print(tf.__version__)"
```

**解决** (Linux/Windows):
```bash
pip install -U tensorflow
```

### Q: 输出目录权限错误

**症状**: `✗ 检查: output_directories FAIL`

**解决**:
```bash
# 创建必要的目录
mkdir -p results models figures logs outputs/run_snapshots outputs/env_snapshots

# 检查权限
chmod 755 results models figures logs outputs*
```

### Q: 依赖库缺失

**症状**: `⚠ 检查: dependencies WARN - Missing: numpy, pandas, ...`

**解决**:
```bash
# 安装所有依赖
pip install -r requirements.txt

# 或单独安装
pip install numpy pandas scikit-learn pyyaml
```

---

## 💾 输出文件说明

验证完成后会生成以下文件：

### 1. 验证结果日志

**路径**: `./outputs/verify_bootstrap_result.json`  
**内容**:
```json
{
  "timestamp": "2026-03-13T15:30:45.123456",
  "config_file": "config/experiment_config.yaml",
  "checks": {
    "config_files": {...},
    "config_validation": {...},
    ...
  },
  "warnings": [...],
  "errors": [...],
  "final_status": "PASS"
}
```

### 2. 配置快照

**路径**: `./outputs/run_snapshots/config_snapshot_20260313_153045.json`  
**用途**: 记录本次运行的配置，便于复现实验

**内容**:
```json
{
  "dataset": {...},
  "model": {...},
  "training": {...},
  "experiment_matrix": {...},
  "dependencies": {...},
  "load_timestamp": "2026-03-13T15:30:00"
}
```

### 3. 环境快照

**路径**: `./outputs/env_snapshots/ENV_20260313_153045.json`  
**用途**: 记录运行环境，便于诊断环境相关问题

**内容**:
```json
{
  "platform": "macOS",
  "machine": "arm64",
  "python_version": "3.9.18",
  "packages": {
    "numpy": "1.21.6",
    "pandas": "1.3.5",
    "tensorflow": "2.12.0",
    ...
  },
  "tensorflow": {
    "available": true,
    "version": "2.12.0",
    "devices": ["CPU:0"],
    "gpu_available": false
  },
  "timestamp": "2026-03-13T15:30:45.123456"
}
```

---

## 🎯 典型工作流

### 新机器首次设置

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证系统就绪
python verify_bootstrap.py

# 3. 如果有警告，修复
# （例如: 删除旧 tensorflow，安装 tensorflow-macos）

# 4. 重新验证直到通过
python verify_bootstrap.py

# 5. 查看快照记录环境状态
cat outputs/env_snapshots/ENV_*.json | jq '.python_version, .packages.tensorflow'
```

### 多次实验对比环保差异

```bash
# 实验 1
python main.py --config config/exp1.yaml
python verify_bootstrap.py --config config/exp1.yaml
# 快照: ENV_20260313_150000.json

# 实验 2 (几天后)
python main.py --config config/exp2.yaml
python verify_bootstrap.py --config config/exp2.yaml
# 快照: ENV_20260313_160000.json

# 对比两个环境
diff outputs/env_snapshots/ENV_*.json
# 帮助诊断: numpy 版本更新了吗？TensorFlow 兼容吗？
```

### 在持续集成 (CI) 中使用

```yaml
# .github/workflows/verify.yml
name: Verify Config
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python verify_bootstrap.py
        # 如果验证失败，GitHub 会拒绝 merge
```

---

## 📈 脚本流程图

```
开始
  ↓
[步骤 1] 加载配置
  ↓
  ├─ 检查文件存在性
  ├─ 解析 YAML
  └─ 实例化 Config
  ↓
[步骤 2] 验证配置
  ↓
  ├─ 交叉字段检查
  └─ 类型验证
  ↓
[步骤 3] 系统检查
  ↓
  ├─ 依赖库可用性
  ├─ 输出目录权限
  ├─ 平台检测
  └─ TensorFlow 状态
  ↓
[步骤 4] 保存快照
  ↓
  ├─ 配置快照 (JSON)
  └─ 环境快照 (JSON)
  ↓
[步骤 5] 最终报告
  ↓
  ├─ 汇总结果
  ├─ 给出建议
  └─ 返回状态码 (0=PASS, 1=WARN, 2=FAIL)
  ↓
结束
```

---

## 🔧 高级用法

### 以编程方式调用

```python
from verify_bootstrap import BootstrapVerifier

# 创建验证器
verifier = BootstrapVerifier(
    config_file="config/experiment_config.yaml",
    quiet=False
)

# 运行验证
exit_code = verifier.run()

# 访问结果
print(f"最终状态: {verifier.results['final_status']}")
print(f"错误数: {len(verifier.results['errors'])}")
print(f"快照路径: {verifier.results['snapshots']}")

# 返回码: 0=pass, 1=warn, 2=fail
sys.exit(exit_code)
```

### 批量验证多个配置

```bash
for config in config/exp_*.yaml; do
  echo "验证 $config..."
  python verify_bootstrap.py --config "$config"
  if [ $? -ne 0 ]; then
    echo "✗ $config 验证失败"
    exit 1
  fi
done
echo "✓ 所有配置验证通过"
```

### 定期检查环境变化

```bash
# crontab -e
# 每天下午 5 点验证一次
0 17 * * * cd /path/to/project && python verify_bootstrap.py --quiet >> logs/daily_verify.log 2>&1
```

---

## 📞 获得帮助

```bash
# 查看脚本帮助
python verify_bootstrap.py --help

# 查看详细文档
less verify_bootstrap.py  # 脚本顶部的 docstring

# 查看具体错误
python verify_bootstrap.py 2>&1 | tee verify_output.log
cat verify_output.log  # 逐行分析
```

---

**祝你的实验顺利进行！** 🎉

如有问题，检查上述常见问题排查或查看脚本的详细注释。
