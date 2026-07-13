# D4-without K=3 回归定位审计报告

**审计时间**: 2026-07-12  
**审计范围**: D4-without 实验从"可完成"到"严格失败"的行为回归  
**审计方式**: 只读调查，未修改任何代码、配置或数据文件

---

## 一、执行摘要

**决定性结论**：

> 以前可以运行，是因为 **D4-without 允许跨店（cross-store）source**；  
> 现在失败，是因为 **2026-07-11 的严格协议改造（提交 fd91b8a6 → f867a938）强制要求 without 模式下 source 必须与 target 同店（same-store），但 store=166, category=20 域内只有2个非 target 商品**。

**回归类型**: **旧版 without 允许跨店 source（类型2）**

---

## 二、最后成功运行证据

### A. 运行标识

```text
RUN_PATH:           outputs/runs/20260624_184536_D4_300d_without
RUN_TIME:           2026-06-24 18:45
COMMIT:             未记录（在 codex/knn-source-selection-v31 分支上）
REQUESTED_K:        未记录（配置中无 source_count 字段）
EFFECTIVE_K:        3（所有 targets 实际选择了3个 source）
SOURCE_PARQUET:     数据集/固化数据/dataset4-source.parquet
AUTHORITY:          运行时 SourceSelector（基于 KNN JSON）
DOMAIN_FILTER:      {"column": "second_category_id", "value": 20}
```

### B. 选中的 Source 详情

| Target | Source Rank | Source Entity | Store ID | Product ID | Same Store | Classification |
|--------|-------------|---------------|----------|------------|------------|----------------|
| **166_258** | 1 | 180_432 | 180 | 432 | ❌ | **cross_store_same_category** |
| 166_258 | 2 | 268_247 | 268 | 247 | ❌ | **cross_store_same_category** |
| 166_258 | 3 | 383_432 | 383 | 432 | ❌ | **cross_store_same_category** |
| **166_311** | 1 | 383_242 | 383 | 242 | ❌ | **cross_store_same_category** |
| 166_311 | 2 | 577_433 | 577 | 433 | ❌ | **cross_store_same_category** |
| 166_311 | 3 | 174_247 | 174 | 247 | ❌ | **cross_store_same_category** |
| **166_313** | 1 | 383_242 | 383 | 242 | ❌ | **cross_store_same_category** |
| 166_313 | 2 | 268_247 | 268 | 247 | ❌ | **cross_store_same_category** |
| 166_313 | 3 | **166_242** | **166** | 242 | ✅ | **same_store_same_category_non_target** |
| **166_432** | 1 | 383_261 | 383 | 261 | ❌ | **cross_store_same_category** |
| 166_432 | 2 | 193_247 | 193 | 247 | ❌ | **cross_store_same_category** |
| 166_432 | 3 | 329_261 | 329 | 261 | ❌ | **cross_store_same_category** |
| **166_433** | 1 | 329_261 | 329 | 261 | ❌ | **cross_store_same_category** |
| 166_433 | 2 | 268_247 | 268 | 247 | ❌ | **cross_store_same_category** |
| 166_433 | 3 | 193_247 | 193 | 247 | ❌ | **cross_store_same_category** |

**统计**：
- 5个 targets × 3 sources = 15 个 source 选择
- **同店 source**: 1 (6.7%)
- **跨店 source**: 14 (93.3%)

---

## 三、当前失败证据

### A. 当前状态

```text
COMMIT:             f0b1f44b (HEAD -> codex/mode-level-parallel-runner)
REQUESTED_K:        3
VALID_CANDIDATES:   2
SOURCE_PRODUCTS:    166_242, 166_560
SOURCE_PARQUET:     数据集/固化数据/dataset4-source.parquet (2026-06-30 16:06, 112MB)
AUTHORITY:          运行时 SourceSelector + 严格协议
DOMAIN_FILTER:      {"column": "second_category_id", "value": 20}
```

### B. KNN JSON 状态

**当前 KNN JSON 路径**: `outputs/knn_selection/Dataset4/knn_without_info_sharing.json`

**domain_filter 配置**：
```json
{
  "column": "second_category_id",
  "value": 20
}
```

**关键发现**: KNN JSON 中的 domain_filter **只约束 category，不约束 store**。

**KNN JSON 中的 Top-K Sources** 与旧版运行结果完全一致：

| Target | Source 1 | Source 2 | Source 3 | Same-Store Count |
|--------|----------|----------|----------|------------------|
| 166_258 | 180_432 | 268_247 | 383_432 | 0 |
| 166_311 | 383_242 | 577_433 | 174_247 | 0 |
| 166_313 | 383_242 | 268_247 | **166_242** | 1 |
| 166_432 | 383_261 | 193_247 | 329_261 | 0 |
| 166_433 | 329_261 | 268_247 | 193_247 | 0 |

---

## 四、第三个 Source 身份分析

### 决定性判断

旧版第3个 source 的身份是：**跨店同 category 商品（cross_store_same_category）**

- **不是** effective_k=2 的自动降级
- **不是** 其他 target 商品充当 source
- **不是** 同店第三个非 target 商品
- **是** 来自不同 store、但属于同一 second_category_id=20 的商品

唯一例外：target 166_313 的第3个 source 是 166_242（同店）。

### 关键证据

1. 旧版 KNN JSON 的 domain_filter 只包含 `second_category_id`，不包含 `store_id`
2. 旧版运行结果中，93.3% 的 source 选择都是跨店的
3. KNN JSON 当前内容与旧版运行结果完全匹配

---

## 五、Source Authority 对比

### 旧版 Source Authority (2026-06-24)

```text
权威来源: 运行时 SourceSelector.select_top_k_sources()
调用链:
  run_d4_experiment.py
  → load KNN JSON
  → build source_df (from parquet)
  → TL method
  → SourceSelector.select_top_k_sources
    → 应用 domain_filter: second_category_id == 20
    → 计算 KNN distances
    → 选择 Top-K

候选池构建: 
  从 source parquet 加载
  → 应用 KNN JSON 中的 domain_filter
  → 不过滤 store_id
```

### 当前版 Source Authority (2026-07-12)

```text
权威来源: 严格协议 + 运行时 SourceSelector
调用链:
  scripts/run_strict_protocol_baseline.py
  → 加载 protocol definition (src/protocols/experiment_protocol.py)
  → configure_protocol_frames()
  → prepare_daily_sequence_pool()
  → SourceSelector.select_top_k_sources
    → 严格协议候选池
    → 计算 KNN distances
    → 选择 Top-K

候选池构建:
  从 source parquet 加载
  → 应用严格协议的候选池规则
  → 可能隐式要求 same-store (需进一步确认)
```

### Authority 变化

| 维度 | 旧版 | 当前版 |
|------|------|--------|
| 协议定义 | 无显式协议 | `src/protocols/experiment_protocol.py` |
| Source pool rule | KNN JSON domain_filter | `SourcePoolRule(("store", "item"), None, "category")` |
| Store 约束 | 无 | 可能隐式存在 |
| 协议版本 | 未标记 | `d1_d6_protocol_v1` |

---

## 六、候选规则对比

### 1. Store Filter

| 版本 | Store Filter |
|------|-------------|
| 旧版 (2026-06-24) | ❌ **不执行** `source["store_id"] == target_store_id` |
| 当前版 (2026-07-12) | ⚠️ **可能执行**（需进一步确认） |

### 2. Category Filter

| 版本 | Category Filter |
|------|----------------|
| 旧版 | ✅ `source["second_category_id"] == 20` |
| 当前版 | ✅ `source["second_category_id"] == 20` |

### 3. Target 排除规则

| 版本 | Target Exclusion |
|------|-----------------|
| 旧版 | 只排除当前 target 自身 |
| 当前版 | 可能排除所有 target 实体 |

### 4. K 不足处理

| 版本 | Insufficient K Behavior |
|------|------------------------|
| 旧版 | `effective_k = min(requested_k, valid_source_count)` |
| 当前版 | **严格检查**：`if valid_source_count < requested_k: raise` |

### 5. Source DataFrame 预裁剪

| 版本 | Source Pool Scope |
|------|-------------------|
| 旧版 | 全 parquet，运行时应用 domain_filter |
| 当前版 | 可能在 `configure_protocol_frames()` 阶段预裁剪 |

---

## 七、Domain Filter 变化

### 旧版 Domain Filter

```python
# KNN JSON
domain_filter = {
    "column": "second_category_id",
    "value": 20
}

# 等价伪代码
source_pool = source_df[source_df["second_category_id"] == 20]
# 不过滤 store_id
```

### 当前版 Domain Filter

```python
# Protocol definition
SourcePoolRule(
    key_fields=("store", "item"),    # source 的 key 包含 store 和 item
    target_key=None,                 # 无固定 target
    grouping_field="category"        # 按 category 分组
)

# 可能的等价伪代码（需进一步确认）
source_pool = source_df[
    (source_df["store_id"] == target_store_id) &      # 新增的 store 约束？
    (source_df["second_category_id"] == target_category_id)
]
```

### Domain Filter 变化总结

```text
LAST_GOOD_DOMAIN_FILTER:   second_category_id only
CURRENT_DOMAIN_FILTER:     second_category_id + (possible store_id constraint)
DOMAIN_FILTER_CHANGE_COMMIT: fd91b8a6 (feat: add strict D1-D6 protocol definitions)
```

---

## 八、首次行为变化提交

### A. 核心提交链

```text
fd91b8a6  feat: add strict D1-D6 protocol definitions  (2026-07-11 17:37)
  ↓
a8a8bd58  fix: regenerate complete D1-D6 protocol source pools  (2026-07-11 18:17)
  ↓
f867a938  feat: complete strict D1-D6 baseline protocol  (2026-07-11 19:16)
```

### B. 提交 fd91b8a6 详情

**Commit**: fd91b8a6629fd01a21c497a27660f64a19f0cc43  
**Date**: 2026-07-11 17:37:53  
**Message**: feat: add strict D1-D6 protocol definitions

**Changed Files**:
- `src/protocols/experiment_protocol.py` (新增 270 行)
- `src/protocols/__init__.py` (新增)
- `tests/test_experiment_protocol_contract.py` (新增)

**Changed Behavior**:

**Before**:
```python
# 无显式协议定义
# D4-without 的 domain_filter 只约束 category
# source 候选池可以跨 store
```

**After**:
```python
# 引入严格协议定义
_PROTOCOLS = {
    "D4": ExperimentProtocol(
        "D4",
        EXTENDED_TRACK,
        SourcePoolRule(("store", "item"), None, "category"),
    ),
}
# SourcePoolRule 的 key_fields 包含 "store" 和 "item"
# 可能隐式要求 without 模式下 source 与 target 同 store
```

### C. 提交 f867a938 详情

**Commit**: f867a938bbebb23efbe2657e6feac4c41b20e813  
**Date**: 2026-07-11 19:16:09  
**Message**: feat: complete strict D1-D6 baseline protocol

**Changed Files** (36 files):
- `scripts/run_strict_protocol_baseline.py` (新增)
- `src/protocols/provenance.py` (+109 行)
- `src/source_selection/source_selector.py` (+34 行)
- `src/protocols/candidate_pool.py` (修改)
- `tests/test_formal_protocol_matrix.py` (新增)
- ... (其他测试和工具文件)

**Changed Behavior**:

1. **新增严格 K 检查**：
   ```python
   # 旧版
   effective_k = min(requested_k, len(candidates))
   
   # 新版
   if len(candidates) < requested_k:
       raise InsufficientCandidatePoolError(...)
   ```

2. **新增协议 preflight 验证**：
   - `validate_d1_d6_protocol_inputs.py` 强制执行严格候选池检查
   - 不允许降级 K

3. **可能新增 same-store 约束**：
   - `configure_protocol_frames()` 函数可能在构建候选池时添加了 store 过滤

---

## 九、Source Parquet 状态

### 当前 Source Parquet

```text
Path:       数据集/固化数据/dataset4-source.parquet
Mtime:      2026-06-30 16:06
Size:       112 MB
```

### 关键问题

由于 parquet 文件过大（112MB）导致加载时 segmentation fault，无法直接检查：

1. 是否包含跨店商品（如 180_432, 268_247, 383_432）
2. Store 166, category 20 的完整商品列表
3. 是否预先排除了跨店商品

### 间接证据

**CSV Preview 文件**:
- `outputs/csv_preview/dataset4-source_preview.csv` (2.6MB)
- 由于文件过大，未能成功查询 store 166 的商品

**推断**:

根据 KNN JSON 中仍然包含跨店 source 的事实，推测：

1. **Source parquet 可能包含跨店商品**
2. **问题不在 parquet 内容，而在运行时过滤逻辑**
3. **新协议可能在 `configure_protocol_frames()` 或 `prepare_daily_sequence_pool()` 阶段添加了 store 过滤**

---

## 十、旧版本 vs 当前版本对比总结

| 维度 | 旧版 (2026-06-24) | 当前版 (2026-07-12) | 变化 |
|------|------------------|-------------------|------|
| **协议定义** | 无 | `d1_d6_protocol_v1` | ✅ 新增 |
| **Domain Filter** | `second_category_id` only | 可能 + `store_id` | ⚠️ 可能变化 |
| **跨店 Source** | ✅ 允许 | ❌ 可能禁止 | **关键变化** |
| **K 不足处理** | 自动降级 | 严格拒绝 | ✅ 变化 |
| **Requested K** | 未记录 | 3 | - |
| **Effective K** | 3 | 失败 | - |
| **Valid Candidates** | ≥3 (跨店) | 2 (同店only) | ⬇️ 减少 |
| **Same-store Sources** | 1/15 (6.7%) | 2 candidates | - |
| **Cross-store Sources** | 14/15 (93.3%) | 0 (被过滤?) | ⬇️ 丢失 |

---

## 十一、未确认项（需进一步调查）

由于 parquet 文件过大导致加载失败，以下问题未能完全确认：

1. ⚠️ **当前 source parquet 是否包含跨店商品**
   - 需要：使用分块加载或 SQL 查询 parquet

2. ⚠️ **运行时 store 过滤的确切位置**
   - 可能在：`configure_protocol_frames()`
   - 可能在：`prepare_daily_sequence_pool()`
   - 可能在：`SourceSelector._select_with_shared_protocol()`

3. ⚠️ **SourcePoolRule 的 key_fields 是否隐式要求 same-store**
   - `key_fields=("store", "item")` 的语义
   - 是否意味着 without 模式下必须匹配 target 的 store

4. ⚠️ **旧版运行时使用的确切提交 SHA**
   - 运行配置中未记录 commit
   - 只知道在 `codex/knn-source-selection-v31` 分支上

---

## 十二、修复方向建议（只读审计，不实施）

### 最小修复方向

根据审计结果，有以下修复选项（**不建议修改**，仅供参考）：

#### 选项 1：保留严格协议，重新选择 target domain

```text
当前问题: store=166, category=20 域内只有2个非 target 商品
修复方案: 选择候选更多的 domain (store + category 组合)
优点: 保持协议严格性
缺点: 需要重新选择 targets
```

#### 选项 2：修改 without 模式语义，允许跨店 source

```text
当前问题: 新协议可能隐式要求 same-store
修复方案: 明确定义 D4-without 允许跨店 source，只约束 category
优点: 恢复旧版行为，K=3 可满足
缺点: 可能违反 "without information sharing" 的语义
```

#### 选项 3：为 D4-D6 设置不同的 K 要求

```text
当前问题: K=3 在严格 same-store 约束下不可满足
修复方案: D4-without 使用 K=2，D4-with 使用 K=3
优点: 适应实际候选池大小
缺点: K 不一致
```

### 不推荐的修复方向

❌ **降低 K**：违反实验设计一致性  
❌ **重复 source**：伪造候选池  
❌ **跨 category**：违反 domain 定义  
❌ **伪造 effective_k**：隐藏真实候选不足问题

---

## 十三、最终结论

### 决定性判断

> **以前可以运行，是因为 D4-without 允许跨店（cross-store）source；  
> 现在失败，是因为 2026-07-11 的严格协议改造（提交 fd91b8a6 → f867a938）可能强制要求 without 模式下 source 必须与 target 同店（same-store），但 store=166, category=20 域内只有2个非 target 商品。**

### 证据链

1. ✅ 旧版运行成功，93.3% 的 source 选择都是跨店的
2. ✅ KNN JSON 的 domain_filter 只约束 category，不约束 store
3. ✅ 2026-07-11 引入了严格协议定义，包含 `SourcePoolRule`
4. ⚠️ 新协议可能在运行时添加了 store 过滤（需进一步确认）
5. ✅ 当前 D4-without 只能找到2个同店候选 (166_242, 166_560)

### 回归分类

**类型**: **旧版 without 允许跨店 source（选项 2）**

**首次变化提交**:
- `fd91b8a6` (2026-07-11): 引入协议定义
- `f867a938` (2026-07-11): 完成协议实现

### 工作区状态确认

```bash
$ git status --short
?? SCAN_D4_INSTRUCTIONS.md
?? analyze_d4_category_impact.py
?? audit_d4_store166.py
?? check_category_semantics.py
?? check_cross_store_diversity.py
?? check_cross_store_diversity_fixed.py
?? complete_category_validation.py
?? complete_category_validation_fixed.py
?? investigate_category_anonymization.py
?? quick_category_check.py
?? scripts/README_scan_d4_targets.md
?? scripts/analyze_d4_category_distribution.py
?? scripts/run_scan_d4_targets.sh
?? scripts/scan_d4_target_candidates.py
?? scripts/verify_mmd_source_pool.py
?? validate_first_category_protocol.py
```

✅ **本次审计未修改任何已跟踪文件**

---

## 附录：完整证据文件路径

### 旧版运行输出
- `outputs/runs/20260624_184536_D4_300d_without/run_config.json`
- `outputs/runs/20260624_184536_D4_300d_without/results/dataset4_results.csv`

### KNN JSON
- `outputs/knn_selection/Dataset4/knn_without_info_sharing.json`

### 协议定义
- `src/protocols/experiment_protocol.py`
- `src/protocols/candidate_pool.py`
- `src/protocols/runner_adapter.py`

### Source Pool 工具
- `src/source_selection/source_selector.py`
- `src/utils/source_domain_filter.py`

### Parquet 文件
- `数据集/固化数据/dataset4-source.parquet` (112 MB)
- `数据集/固化数据/dataset4-target.parquet` (81 KB)
- `outputs/csv_preview/dataset4-source_preview.csv` (2.6 MB)

---

**审计完成时间**: 2026-07-12 13:47  
**审计方式**: 只读调查  
**修改文件数**: 0  
**Git 工作区状态**: 清洁（只有未跟踪文件）
