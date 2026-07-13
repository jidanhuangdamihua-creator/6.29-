# D4 Target 候选组合扫描脚本使用指南

## 脚本功能

`scan_d4_target_candidates.py` 用于扫描 D4 数据集中所有的 `(store_id, first_category_id)` 组合，并根据预先定义的标准筛选出适合作为 target 的组合。

### 核心功能

1. **全量扫描**: 遍历 D4 数据集中所有的 `(store_id, first_category_id)` 组合
2. **候选数统计**: 计算每个组合的候选商品数（排除该组已有的 target）
3. **30天完整性检查**: 验证候选商品是否有连续30天的完整数据
4. **类目跨度分析**: 统计每个组合跨越的 `second_category` 数量
5. **预设标准筛选**: 应用预先定义的筛选标准，避免"看数据挑组合"的风险

### 筛选标准（预先定义，不可修改）

| 标准 | 阈值 | 说明 |
|------|------|------|
| 最少候选商品数 | ≥10 | 确保有足够的候选池，避免 K=3 刚好卡在边界 |
| 最少30天完整候选数 | ≥6 | 确保候选商品有足够的历史数据质量 |
| 最大second_category跨度 | ≤2 | 避免语义稀释，保持类目相关性 |

## 使用方法

### 方式1：直接运行（推荐）

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe
python scripts/scan_d4_target_candidates.py
```

### 方式2：使用 codex_timeout 保护（如果扫描时间较长）

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe
python tools/protection/codex_timeout.py python scripts/scan_d4_target_candidates.py
```

### 预计运行时间

- 小数据集（<100MB）: 1-3分钟
- 中等数据集（100-500MB）: 3-10分钟
- 大数据集（>500MB）: 10-30分钟

D4 数据集的 `train.parquet` 约 300MB，预计运行时间在 **5-15分钟**。

## 输出文件

扫描完成后，会在 `outputs/dataset_audit/` 目录下生成以下文件：

### 1. `d4_target_candidates_summary.md` （主报告）

- 扫描结果汇总
- 满足条件的组合列表（按候选数排序）
- 不满足条件的组合及失败原因
- 统计分布（候选数、完整性、类目跨度）
- 选择建议

**用途**: 人工阅读，决策选择哪些组合作为 target

### 2. `d4_target_candidates_qualified.json` （合格组合详情）

包含所有满足条件的组合的详细信息，包括：
- 完整的候选商品ID列表
- 完整的30天合格候选ID列表
- 类目跨度详情
- 日期范围

**用途**: 程序读取，用于后续实验配置

### 3. `d4_target_candidates_all.json` （全部组合汇总）

包含所有扫描到的组合（满足和不满足条件的），但不包含详细的商品列表（节省空间）。

**用途**: 审计追溯，证明扫描的完整性和客观性

### 4. `d4_target_candidates_qualified.csv` （Excel友好格式）

满足条件的组合，CSV格式，UTF-8 with BOM编码，可直接用Excel打开。

**用途**: Excel中查看、排序、筛选

## 输出示例

### 控制台输出示例

```
================================================================================
D4 数据集 Target 候选组合扫描
================================================================================

筛选标准:
  - min_candidate_count: 10
  - min_complete_30day_count: 6
  - max_second_category_span: 2

[1/6] 加载D4数据集...
  - 文件路径: .../Dataset 4叮咚数据集/data/train.parquet
  - 总行数: 2,547,833
  - Row groups: 25
  - 读取列: ['store_id', 'product_id', 'first_category_id', ...]
  ✓ 加载完成: 2,547,833 行
  - 日期范围: 2024-05-01 ~ 2025-04-30
  - 唯一store数: 167
  - 唯一product数: 12,345

[2/6] 分析商品时间跨度和完整性...
  ✓ 完成分析: 12,345 个商品
  - 满足30天完整性: 8,234 (66.7%)

[3/6] 扫描所有 (store_id, first_category_id) 组合...
  ✓ 完成扫描: 1,234 个组合
  - 满足筛选条件: 156 (12.6%)

[4/6] 生成汇总报告...
  ✓ Markdown报告: outputs/dataset_audit/d4_target_candidates_summary.md

[5/6] 保存扫描结果...
  ✓ 满足条件的组合JSON: .../d4_target_candidates_qualified.json
  ✓ 所有组合JSON: .../d4_target_candidates_all.json
  ✓ 满足条件的组合CSV: .../d4_target_candidates_qualified.csv

[6/6] 扫描完成！

================================================================================
汇总统计
================================================================================
总组合数: 1,234
满足条件: 156 (12.6%)

满足条件的组合（前10个）:
--------------------------------------------------------------------------------
store_id     first_cat    候选数      完整候选      跨度    
--------------------------------------------------------------------------------
166          25           45          38            2       
42           15           32          28            2       
101          30           28          24            1       
...

================================================================================
输出文件:
================================================================================
  - 报告: outputs/dataset_audit/d4_target_candidates_summary.md
  - 满足条件的组合: .../d4_target_candidates_qualified.json
  - 所有组合: .../d4_target_candidates_all.json
  - CSV格式: .../d4_target_candidates_qualified.csv
```

## 下一步工作

### 1. 查看扫描报告

```bash
# 用文本编辑器打开主报告
open outputs/dataset_audit/d4_target_candidates_summary.md

# 或用命令行查看
cat outputs/dataset_audit/d4_target_candidates_summary.md
```

### 2. 选择 Target 组合

根据报告中的建议，从满足条件的组合中选择 N 个（N=3-5）作为 target 组合。

**推荐方式**:
- 保留原有的 store166 组合（如果满足条件）
- 从满足条件的组合中随机抽取2-4个
- 或选择候选数接近中位数的组合

**避免**:
- ❌ 不要手动挑选"看起来好"的组合
- ❌ 不要只挑候选数最多的组合

### 3. 更新实验配置

根据选定的组合，更新 D4 实验的 KNN 配置文件和实验脚本。

### 4. 提交审计记录

将扫描结果和选择依据提交到版本控制，形成可追溯的审计链：

```bash
git add outputs/dataset_audit/d4_target_candidates_*
git add scripts/scan_d4_target_candidates.py
git commit -m "Add D4 target candidate scanning results"
```

## 调整筛选标准

如果扫描结果显示：
- **满足条件的组合太少（<3个）**: 需要放宽某些标准
- **满足条件的组合太多（>50个）**: 可以收紧标准或增加其他约束

要调整标准，修改脚本中的 `FILTER_CRITERIA` 字典：

```python
FILTER_CRITERIA = {
    "min_candidate_count": 10,          # 可以降低到 8 或 6
    "min_complete_30day_count": 6,      # 可以降低到 4 或 5
    "max_second_category_span": 2,      # 可以放宽到 3
}
```

**重要**: 修改标准后必须：
1. 在 Git 中记录修改原因
2. 重新运行完整扫描
3. 更新审计报告

## 常见问题

### Q1: 扫描时间太长怎么办？

A: 使用 `codex_timeout.py` 包装器运行，或者分批处理：

```bash
# 限制3分钟超时
python tools/protection/codex_timeout.py python scripts/scan_d4_target_candidates.py
```

### Q2: 内存不足怎么办？

A: 脚本已经使用了分chunk读取，如果仍然内存不足，可以：
- 减小 chunk size
- 在更大内存的机器上运行
- 联系开发者获取优化版本

### Q3: 如何验证扫描结果的正确性？

A: 可以：
1. 检查 `d4_target_candidates_all.json` 中的总组合数是否合理
2. 随机抽取几个组合，手动验证候选数和完整性
3. 对比不同运行的结果是否一致（确定性）

### Q4: 可以修改筛选标准吗？

A: 可以，但必须：
1. **先修改标准，再运行扫描**
2. 在 Git 中记录修改原因
3. 重新运行完整扫描
4. 不能"看到结果后反复调整直到满意"

## 技术细节

### 数据流程

```
D4 train.parquet (300MB)
  ↓ (分chunk读取，只读必要列)
商品信息字典 (product_id -> ProductInfo)
  ↓ (按 store_id, first_category_id 分组)
组合统计 (计算候选数、完整性、跨度)
  ↓ (应用筛选标准)
满足条件的组合 + 不满足条件的组合
  ↓ (生成报告和结果文件)
4个输出文件 (.md, .json x2, .csv)
```

### 30天完整性定义

- 取商品数据的最后30天
- 检查这30天内有多少个不同的日期
- 如果 ≥30 天，则认为满足30天完整性

### 候选数计算

- 当前版本：属于该 `(store_id, first_category_id)` 的所有商品
- 实际使用时可能需要：
  - 排除已选为 target 的商品
  - 排除历史数据不足的商品
  - 排除某些特殊类目的商品

## 维护记录

| 日期 | 版本 | 修改内容 |
|------|------|----------|
| 2026-07-12 | 1.0 | 初始版本，实现基础扫描功能 |

## 联系方式

如有问题或需要协助，请联系实验负责人。
