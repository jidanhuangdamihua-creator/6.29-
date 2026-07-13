# D4 数据集 Target 候选扫描 - 执行指令

## 快速开始（推荐）

### 方式1：使用启动脚本

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe
./scripts/run_scan_d4_targets.sh
```

### 方式2：直接运行Python脚本

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe
python scripts/scan_d4_target_candidates.py
```

### 方式3：使用超时保护（如果担心运行时间过长）

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe
python tools/protection/codex_timeout.py python scripts/scan_d4_target_candidates.py
```

## 预计运行时间

- **D4 数据集大小**: ~300MB (train.parquet)
- **预计扫描时间**: 5-15分钟
- **内存需求**: ~2-4GB

## 输出结果

扫描完成后，会在 `outputs/dataset_audit/` 目录生成以下文件：

```
outputs/dataset_audit/
├── d4_target_candidates_summary.md      # 主报告（人工阅读）
├── d4_target_candidates_qualified.json  # 满足条件的组合（程序使用）
├── d4_target_candidates_all.json        # 所有组合（审计追溯）
└── d4_target_candidates_qualified.csv   # Excel友好格式
```

## 查看结果

### 1. 在终端查看报告

```bash
cat outputs/dataset_audit/d4_target_candidates_summary.md
```

### 2. 在编辑器打开报告

```bash
# macOS
open outputs/dataset_audit/d4_target_candidates_summary.md

# Linux
xdg-open outputs/dataset_audit/d4_target_candidates_summary.md

# 或使用你喜欢的编辑器
vim outputs/dataset_audit/d4_target_candidates_summary.md
code outputs/dataset_audit/d4_target_candidates_summary.md
```

### 3. 查看CSV（Excel）

```bash
# macOS
open outputs/dataset_audit/d4_target_candidates_qualified.csv

# Excel会自动识别UTF-8 with BOM编码，中文显示正常
```

## 筛选标准说明

本次扫描使用以下**预先定义**的标准（不可临时修改）：

| 标准 | 阈值 | 说明 |
|------|------|------|
| 最少候选商品数 | ≥10 | 避免K=3刚好卡在边界 |
| 最少30天完整候选数 | ≥6 | 确保候选有足够历史数据 |
| 最大second_category跨度 | ≤2 | 避免语义稀释 |

**重要原则**：
- ✅ 先定标准，再扫描
- ✅ 标准对所有组合一视同仁
- ❌ 不能看结果后再调标准
- ❌ 不能手动挑"看着好"的组合

## 扫描输出示例

```
================================================================================
D4 数据集 Target 候选组合扫描
================================================================================

筛选标准:
  - min_candidate_count: 10
  - min_complete_30day_count: 6
  - max_second_category_span: 2

[1/6] 加载D4数据集...
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
[5/6] 保存扫描结果...
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
```

## 下一步操作

### 1. 阅读扫描报告

查看 `d4_target_candidates_summary.md`，了解：
- 有多少组合满足条件
- 候选数、完整性、跨度的分布
- 推荐的选择策略

### 2. 选择Target组合

根据报告建议，从满足条件的组合中选择 3-5 个：

**推荐策略**：
1. 保留原有的 store166（如果满足条件）
2. 从其余组合中随机抽取 2-4 个
3. 或选择候选数接近中位数的组合

**示例选择过程**：
```python
import json
import random

# 读取满足条件的组合
with open('outputs/dataset_audit/d4_target_candidates_qualified.json', 'r') as f:
    data = json.load(f)
    qualified = data['qualified_groups']

# 如果 store166 + first_category=25 满足条件，保留它
store166_group = [g for g in qualified if g['store_id'] == 166 and g['first_category_id'] == 25]

# 从其余组合中随机抽取3个
other_groups = [g for g in qualified if not (g['store_id'] == 166 and g['first_category_id'] == 25)]
random.seed(42)  # 固定随机种子以保证可复现
selected_others = random.sample(other_groups, min(3, len(other_groups)))

# 合并
final_selection = store166_group + selected_others

print("选定的Target组合:")
for g in final_selection:
    print(f"  - store_id={g['store_id']}, first_category_id={g['first_category_id']}, "
          f"候选数={g['candidate_products']}, 完整候选={g['complete_30day_candidates']}")
```

### 3. 提交审计记录

```bash
git add outputs/dataset_audit/d4_target_candidates_*
git add scripts/scan_d4_target_candidates.py
git add scripts/README_scan_d4_targets.md
git add SCAN_D4_INSTRUCTIONS.md
git commit -m "Add D4 target candidate scanning: found X qualified groups using pre-defined criteria"
```

### 4. 更新实验配置

根据选定的组合，更新 D4 实验的配置文件。

## 故障排查

### 问题1：文件不存在错误

```
FileNotFoundError: D4数据集不存在: .../Dataset 4叮咚数据集/data/train.parquet
```

**解决方案**：检查数据集路径是否正确
```bash
ls -la "/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 4叮咚数据集/data/"
```

### 问题2：内存不足

```
MemoryError
```

**解决方案**：
1. 关闭其他占用内存的程序
2. 在更大内存的机器上运行
3. 联系开发者获取优化版本

### 问题3：导入错误

```
ImportError: No module named 'pyarrow'
```

**解决方案**：安装依赖
```bash
pip install pyarrow pandas numpy
```

### 问题4：运行时间过长

**解决方案**：使用超时保护
```bash
python tools/protection/codex_timeout.py python scripts/scan_d4_target_candidates.py
```

如果超时，手动在终端运行：
```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe
nohup python scripts/scan_d4_target_candidates.py > scan_d4_output.log 2>&1 &
```

监控进度：
```bash
tail -f scan_d4_output.log
```

## 调整筛选标准（如果需要）

如果扫描结果显示满足条件的组合太少或太多，可以调整标准。

**重要**：调整标准必须遵循以下流程：
1. 在 Git 中创建新的 commit，说明调整原因
2. 修改脚本中的 `FILTER_CRITERIA`
3. 重新运行**完整**扫描
4. 不能反复试探直到"满意"

**修改位置**：`scripts/scan_d4_target_candidates.py` 第34-38行

```python
FILTER_CRITERIA = {
    "min_candidate_count": 10,          # 可调整为 8, 6, ...
    "min_complete_30day_count": 6,      # 可调整为 4, 5, ...
    "max_second_category_span": 2,      # 可调整为 3, 4, ...
}
```

## 技术支持

如有问题，请查阅：
- 详细文档：`scripts/README_scan_d4_targets.md`
- 脚本源码：`scripts/scan_d4_target_candidates.py`

## 版本信息

- **脚本版本**: 1.0
- **创建日期**: 2026-07-12
- **适用数据集**: D4 (叮咚买菜)
- **Python要求**: ≥3.8
- **依赖**: pandas, numpy, pyarrow
