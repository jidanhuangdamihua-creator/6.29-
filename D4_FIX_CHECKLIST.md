# D4 问题修复 - 执行清单

## ✅ 已完成：问题诊断

### 发现的问题
1. **观察窗口定义错误** - 边界计算导致source候选无法满足观察窗口要求
2. **Category粒度太细** - second_category=20只有2个候选，不满足K≥3

### 验证结果
- Store 166 + first_category=15: **5个有效候选** ✅
- 全部满足30天观察窗口完整性 ✅
- K≥3可行 ✅

## 📋 待执行：修复配置

### [ ] Step 1: 修改固化配置文件

**文件位置**：
- `configs/solidified/knn/Dataset4/knn_without_info_sharing.json`
- `configs/solidified/knn/Dataset4/knn_with_info_sharing.json`

**修改内容**：
```json
// 将这个：
"domain_filter": {
  "column": "second_category_id",
  "value": 20
}

// 改为这个：
"domain_filter": {
  "column": "first_category_id",
  "value": 15
}
```

**快速命令**（macOS/Linux）：
```bash
cd configs/solidified/knn/Dataset4

# 备份
cp knn_without_info_sharing.json knn_without_info_sharing.json.backup
cp knn_with_info_sharing.json knn_with_info_sharing.json.backup

# 修改（使用Python脚本更安全）
python3 << 'EOF'
import json
from pathlib import Path

for filename in ['knn_without_info_sharing.json', 'knn_with_info_sharing.json']:
    path = Path(filename)
    with open(path) as f:
        config = json.load(f)
    
    # 修改domain_filter
    config['domain_filter'] = {
        'column': 'first_category_id',
        'value': 15
    }
    
    # 写回
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f'✅ Updated {filename}')
EOF
```

### [ ] Step 2: 验证修改

```bash
python3 -c "
import json
from pathlib import Path

for scenario in ['without', 'with']:
    path = Path(f'configs/solidified/knn/Dataset4/knn_{scenario}_info_sharing.json')
    with open(path) as f:
        config = json.load(f)
    
    domain_filter = config['domain_filter']
    expected = {'column': 'first_category_id', 'value': 15}
    
    if domain_filter == expected:
        print(f'✅ {scenario}: {domain_filter}')
    else:
        print(f'❌ {scenario}: {domain_filter} (expected {expected})')
"
```

### [ ] Step 3: 检查代码中的观察窗口逻辑

搜索可能有问题的地方：

```bash
# 搜索观察窗口相关代码
rg "OBS.*END.*=.*train.*start|knn_observed_end.*=" --type py

# 搜索source过滤逻辑
rg "dt.*<.*train_start|dt.*<.*TARGET_TRAIN" --type py
```

**需要确保**：
```python
# 正确的观察窗口定义
knn_observed_end = target_train_start - pd.Timedelta(days=1)  # 减1天
target_observed_start = knn_observed_end - pd.Timedelta(days=29)
source_observation_cutoff = knn_observed_end

# Source过滤
source_df = source_df[source_df['dt'] <= source_observation_cutoff]  # <=不是<
```

### [ ] Step 4: 验证其他target stores

运行验证脚本检查Store 155/240/293：

```bash
# 创建验证脚本（检查所有4个stores）
python validate_all_d4_stores.py  # 待创建
```

### [ ] Step 5: 重新运行D4实验

```bash
# 使用修正后的配置
python scripts/run_d4_experiment.py --info-sharing without

# 或使用超时保护
python tools/protection/codex_timeout.py --timeout 300 \
  python scripts/run_d4_experiment.py --info-sharing without
```

### [ ] Step 6: 提交修改

```bash
git add configs/solidified/knn/Dataset4/
git add D4_FIX_GUIDE.md
git add D4_FIX_CHECKLIST.md

git commit -m "Fix D4 config: use first_category=15 for K>=3 feasibility

- Change domain_filter from second_category_id=20 to first_category_id=15
- Verified with raw data: store166 has 5 valid candidates
- Fixes K<3 issue by using broader category granularity
- See D4_FIX_GUIDE.md for detailed analysis"
```

## 📊 预期结果

修复后应该看到：
- Store 166: K=3成功，使用候选242/244/246/548/560
- 实验成功完成，不再报K<3错误
- 结果写入outputs目录

## ⚠️  如果还有问题

### 问题A：固化数据是基于错误配置生成的

**症状**：修改JSON后运行仍然失败，或者固化数据中没有first_category=15的数据

**解决**：需要重新生成固化parquet文件
```bash
# 找到数据生成脚本（具体路径待确定）
# python scripts/generate_d4_solidified_data.py
```

### 问题B：其他stores不满足K≥3

**症状**：Store 155/240/293在first_category粒度下仍然K<3

**解决**：
1. 运行`compare_category_groupings.py`检查每个store
2. 根据结果决定：
   - 使用WITH scenario（跨store）
   - 调整source window天数（180/210/300）
   - 或为不同store使用不同配置

### 问题C：观察窗口逻辑修改困难

**症状**：代码中多处涉及观察窗口，不确定都改了哪些

**解决**：
1. 查看`D4_FIX_GUIDE.md`中的"需要修改的文件"部分
2. 搜索关键字：`obs.*end`, `knn_observed`, `train_start - 30`
3. 确保所有观察窗口计算遵循协议：
   ```
   knn_observed_end = target_observed_start + 29 days
   source_observation_cutoff = knn_observed_end
   ```

## 📞 需要帮助？

如果遇到问题：
1. 检查`D4_FIX_GUIDE.md`的详细说明
2. 运行验证脚本确认当前状态
3. 查看错误日志，确定是配置问题还是代码问题

---

**当前状态**：
- ✅ 问题已诊断
- ✅ 解决方案已验证
- ⏳ 等待执行修改

**预计时间**：
- 修改配置：5分钟
- 验证修改：2分钟
- 重新运行实验：根据数据大小，可能5-30分钟
