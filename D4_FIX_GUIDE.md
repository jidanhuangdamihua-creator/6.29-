# D4 实验修复指南 - 观察窗口和Category粒度

## 🔍 问题确认

通过原始数据检查，确认了两个根本问题：

### 问题1：观察窗口定义错误
**错误定义**：
```python
OBS_START = 2024-11-17
OBS_END = 2024-12-16  # target_train_start当天
source_filter: df['dt'] < TARGET_TRAIN_START  # 不包含12-16
```

**问题**：观察窗口要求12-16有数据，但source过滤器排除了12-16，导致所有候选都缺失最后一天。

**正确定义**（根据协议）：
```python
TARGET_TRAIN_START = 2024-12-16
KNN_OBSERVED_END = TARGET_TRAIN_START - pd.Timedelta(days=1)  # 2024-12-15
TARGET_OBSERVED_START = KNN_OBSERVED_END - pd.Timedelta(days=29)  # 2024-11-16
SOURCE_OBSERVATION_CUTOFF = KNN_OBSERVED_END  # 2024-12-15

# 观察窗口：2024-11-16 to 2024-12-15 (30 days)
# Source过滤：df['dt'] <= SOURCE_OBSERVATION_CUTOFF  # 包含12-15
```

**影响**：修复后，之前的"29/30天"候选现在都满足30天完整性。

### 问题2：Category粒度太细

**当前配置**（second_category=20）：
- 候选数：2个（242, 560）
- K≥3可行：❌ NO

**推荐配置**（first_category=15）：
- 候选数：5个（242, 244, 246, 548, 560）
- K≥3可行：✅ YES
- 包含3个不同second_category：
  - second_category=20: 2个产品（242, 560）
  - second_category=22: 3个产品（244, 246, 548）

## ✅ 验证结果

Store 166配置对比：

| 配置 | Category | 候选数 | 有效数 | K≥3 |
|------|----------|--------|--------|-----|
| 当前（错误） | second_cat=20 | 2 | 0 | ❌ |
| 修复后 | second_cat=20 | 2 | 2 | ❌ |
| **推荐** | **first_cat=15** | **5** | **5** | **✅** |

## 🔧 需要修改的文件

### 1. 固化配置文件

**文件**：
- `configs/solidified/knn/Dataset4/knn_without_info_sharing.json`
- `configs/solidified/knn/Dataset4/knn_with_info_sharing.json`

**修改**：
```json
{
  "domain_filter": {
    "column": "first_category_id",  // 从 second_category_id 改为 first_category_id
    "value": 15                      // 从 20 改为 15
  }
}
```

### 2. 观察窗口计算逻辑

需要检查的文件（搜索 "obs.*start|obs.*end|knn_observed"）：

可能的位置：
- `src/utils/parquet_data_loader.py`
- `src/utils/d4_d6_runtime.py`
- `src/protocols/candidate_pool.py`

**修改原则**：
```python
# 任何计算observation window的地方，确保：
knn_observed_end = target_train_start - pd.Timedelta(days=1)
target_observed_start = knn_observed_end - pd.Timedelta(days=29)
source_observation_cutoff = knn_observed_end

# Source过滤：
source_df = source_df[source_df['dt'] <= source_observation_cutoff]
# 不要用 < target_train_start
```

### 3. 验证脚本中的错误窗口定义

已创建的脚本（需要修正或标记为deprecated）：
- `validate_source_window_matrix.py` - 观察窗口错误
- `validate_source_window_solidified.py` - 观察窗口错误
- `inspect_d4_raw_data.py` - 观察窗口错误
- `inspect_d4_raw_data_fixed.py` - 观察窗口错误

**已修正的脚本**（可以使用）：
- `inspect_d4_corrected_observation_window.py` - ✅ 观察窗口正确
- `compare_category_groupings.py` - ✅ 观察窗口正确

## 📋 修复步骤

### Step 1: 备份当前配置

```bash
cd configs/solidified/knn/Dataset4
cp knn_without_info_sharing.json knn_without_info_sharing.json.backup.second_cat20
cp knn_with_info_sharing.json knn_with_info_sharing.json.backup.second_cat20
```

### Step 2: 修改固化配置

使用文本编辑器或脚本修改两个JSON文件：

```bash
# WITHOUT场景
sed -i '' 's/"column": "second_category_id"/"column": "first_category_id"/' \
  configs/solidified/knn/Dataset4/knn_without_info_sharing.json
sed -i '' 's/"value": 20/"value": 15/' \
  configs/solidified/knn/Dataset4/knn_without_info_sharing.json

# WITH场景
sed -i '' 's/"column": "second_category_id"/"column": "first_category_id"/' \
  configs/solidified/knn/Dataset4/knn_with_info_sharing.json
sed -i '' 's/"value": 20/"value": 15/' \
  configs/solidified/knn/Dataset4/knn_with_info_sharing.json
```

或者手动编辑。

### Step 3: 验证修改

```bash
python3 -c "
import json
from pathlib import Path

for scenario in ['without', 'with']:
    path = Path(f'configs/solidified/knn/Dataset4/knn_{scenario}_info_sharing.json')
    with open(path) as f:
        config = json.load(f)
    
    print(f'{scenario.upper()}:')
    print(f'  domain_filter: {config[\"domain_filter\"]}')
    expected = {'column': 'first_category_id', 'value': 15}
    status = '✅' if config['domain_filter'] == expected else '❌'
    print(f'  Status: {status}')
"
```

### Step 4: 查找并修复代码中的观察窗口逻辑

```bash
# 搜索可能有问题的观察窗口计算
rg "train.*start.*-.*30|OBS.*END.*train.*start" --type py
rg "dt.*<.*train_start|dt.*<.*TARGET_TRAIN" --type py
```

关键是确保：
- 观察窗口结束于 train_start - 1天
- Source过滤使用 `<=` 而不是 `<`

### Step 5: 重新生成固化数据（如果需要）

如果固化的parquet文件是基于错误配置生成的，可能需要重新生成：

```bash
# 检查当前固化数据的配置
python3 -c "
import pandas as pd
from pathlib import Path

source_path = Path('数据集/固化数据/dataset4-source.parquet')
target_path = Path('数据集/固化数据/dataset4-target.parquet')

source_df = pd.read_parquet(source_path)
target_df = pd.read_parquet(target_path)

print('Source data:')
print(f'  Rows: {len(source_df):,}')
print(f'  Date range: {source_df[\"date\"].min()} to {source_df[\"date\"].max()}')

print('\nTarget data:')
print(f'  Rows: {len(target_df):,}')
print(f'  Date range: {target_df[\"date\"].min()} to {target_df[\"date\"].max()}')
print(f'  Unique stores: {target_df[\"store_id\"].nunique()}')
print(f'  Unique products: {target_df[\"product_id\"].nunique()}')
"
```

如果固化数据不对，需要重新运行数据生成脚本（具体脚本路径待确定）。

### Step 6: 重新运行D4实验

```bash
# 使用修正后的配置重新运行
python scripts/run_d4_experiment.py --info-sharing without
```

## 🎯 预期结果

修复后，Store 166（first_category=15）应该：
- 有5个有效候选（242, 244, 246, 548, 560）
- 全部满足30天观察窗口完整性
- K≥3 可行 ✅
- 实验能够成功运行

## 📝 论文中的说明

在论文中应该如实说明：

> "初始配置使用second_category粒度进行候选池分组。在验证过程中发现，对于部分target store，second_category粒度导致候选池小于K=3的最低要求。经过系统性验证（见附录X），我们调整为first_category粒度，以确保所有target store都满足K≥3的基本可行性要求。First_category粒度虽然包含多个second_category，但仍保持类目语义的相关性..."

**不要说**："我们发现first_category效果更好所以用了first_category"
**应该说**："我们为了满足K≥3可行性要求，基于候选池大小验证选择了first_category粒度"

## ⚠️  注意事项

1. **协议一致性**：观察窗口的修正必须在所有组件（KNN选源、CNN训练、评估）中保持一致

2. **Digest更新**：修改category粒度后，candidate_pool_digest会变化，需要重新生成

3. **其他target stores**：Store 155/240/293也需要用同样的方法验证first_category是否可行

4. **文档更新**：更新实验协议文档，记录这次修正的原因和依据

5. **版本控制**：
   ```bash
   git add configs/solidified/knn/Dataset4/
   git commit -m "Fix D4 config: use first_category=15 to ensure K>=3 feasibility
   
   - Change domain_filter from second_category_id=20 to first_category_id=15
   - Verified with raw data: 5 valid candidates (242,244,246,548,560)
   - All candidates meet 30-day observation window completeness
   - Fixes K<3 issue in WITHOUT scenario for store 166"
   ```

## 📊 附录：原始数据验证记录

保存以下文件作为验证依据：
- `d4_raw_data_inspection_fixed.txt` - 初始检查（观察窗口错误）
- `d4_raw_data_inspection_corrected.txt` - 观察窗口修正后的检查
- `compare_category_groupings_output.txt` - Category粒度对比

这些文件可以作为论文附录或补充材料，证明配置选择是基于系统性验证而非随意调整。

---

**状态**：
- ✅ 问题诊断完成
- ✅ 解决方案验证完成
- ⏳ 等待修改配置文件
- ⏳ 等待重新运行实验
