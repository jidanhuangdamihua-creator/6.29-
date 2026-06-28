# MSML-TL-RFE 内部细节第二轮只读审核

## 审核范围与结论摘要

本报告只读检查根目录 `msml_tl_rfe.py`，并沿调用链辅助检查根目录
`experiment_runner.py`、D1/D4 入口及已有运行记录。未运行训练或实验。

| 维度 | 结论标签 | 核心结论 |
|---|---|---|
| RFE estimator 类型 | MATCH | 实际默认使用新建的 `RandomForestRegressor(random_state=..., n_estimators=10)`；也支持新建的 `LinearRegression()`，不接收外部 estimator 对象。D1、D4 调用链均落到默认 `random_forest`。 |
| `n_features_to_select` | MATCH | 按 `ceil(候选特征数 × keep_ratio)` 动态计算；D1 为 2/4，D4 为 6/12，均为 50%。实际 `keep_ratio=0.5`，在 0.4～0.6 区间内。 |
| `IndexError` fallback | RISK | fallback **有筛选，不是全量保留**：按 estimator 重要性取前 `num_to_select` 个；但捕获条件是 RFE 拟合/支持掩码提取期间传播出的任意 `IndexError`，代码无法进一步证明 sklearn 内部的具体越界原因。 |
| source 序列数据范围 | MATCH | `selected_source_sequences` 保存完整 source 序列；RFE 联合数据只使用 source train；source CNN 函数收到完整投影序列后再按 80%/10%/10% 切分，`model.fit` 仅使用 train 窗口，val 仅作验证。 |

## 1. RFE estimator 类型

**结论标签：MATCH**

### 代码实际行为

- `run_rfe_feature_selection()` 的参数仅接收字符串 `estimator_name`，默认值为
  `"random_forest"`，并不接收一个外部 estimator 实例（`msml_tl_rfe.py:140-148`）。
- 当 `estimator_name.lower() == "random_forest"` 时，在函数体内执行：
  `RandomForestRegressor(random_state=random_state, n_estimators=10)`
  （`msml_tl_rfe.py:213-219`）。
  - 类型：`sklearn.ensemble.RandomForestRegressor`
  - 显式超参数：`random_state`（默认 42）、`n_estimators=10`
  - 未显式传入 `C`、`alpha` 或 `kernel`；其余参数使用 sklearn 默认值。
- 当值为 `"linear_regression"` 时，在函数体内执行 `LinearRegression()`
  （`msml_tl_rfe.py:220-225`），没有显式超参数。
- 其他名称直接抛出 `ValueError`（`msml_tl_rfe.py:226-230`）。
- 新建的 `estimator` 随后传入
  `RFE(estimator=estimator, n_features_to_select=num_to_select, step=1)`
  （`msml_tl_rfe.py:250-253`）。

因此，estimator 是**每次调用 `run_rfe_feature_selection()` 时在函数内部新建**的，
不是从外部传入或跨调用复用的。

### 调用方实际参数

- `run_msml_tl_rfe()` 默认 `estimator_name="random_forest"`
  （`msml_tl_rfe.py:946-961`），并在调用 RFE 时原样传递
  （`msml_tl_rfe.py:1210-1217`）。
- 根目录 `experiment_runner.py` 的 `run_msml_rfe_experiment()` 同样默认
  `estimator_name="random_forest"`，并原样传入 `run_msml_tl_rfe()`
  （`experiment_runner.py:1068-1079, 1104-1115`）。
- D1 完整论文入口显式从实验配置传入 estimator；缺省仍为
  `"random_forest"`（`scripts/run_full_paper_experiments.py:881-888`）。
- D4 的 `run_single_entity_experiment()` 没有给 MSML-TL-RFE 增加
  `estimator_name` 覆盖值（`src/utils/entity_experiment.py:212-228`），所以使用
  `experiment_runner.py` 的默认 `"random_forest"`。

## 2. `n_features_to_select` 的计算逻辑

**结论标签：MATCH**

### 计算公式

候选列先从去重后的 `feature_cols` 中排除目标列和 ID/code 类列
（`msml_tl_rfe.py:182-203`）。选择数不是固定整数，而是：

```text
num_to_select_requested = max(1, ceil(num_candidates × keep_ratio))
num_to_select = min(num_to_select_requested, num_candidates)
```

对应代码为 `msml_tl_rfe.py:232-243`，最终传给 RFE 的位置为
`msml_tl_rfe.py:250-253`。默认及 D1/D4 实际调用值均为
`keep_ratio=0.5`，属于要求核对的 0.4～0.6 区间。

### D1、D4 实际数量

| 数据集 | 进入 RFE 前模型列数（含 `sales`） | `rfe_candidate_cols` 数量 | `keep_ratio` | `n_features_to_select` | 候选特征实际保留比例 | 加回 `sales` 后最终模型列数及比例 |
|---|---:|---:|---:|---:|---:|---:|
| D1 / Dataset1 | 5 | 4 | 0.5 | 2 | 2/4 = **50.00%** | 3；3/5 = 60.00% |
| D4 / Dataset4 | 13 | 12 | 0.5 | 6 | 6/12 = **50.00%** | 7；7/13 ≈ 53.85% |

依据：

- D1 特征入口为 `sales, year, month, week, day`
  （`scripts/run_full_paper_experiments.py:606-609`）。`sales` 作为目标从 RFE
  候选中排除，故候选数为 4。已有 D1 运行日志也直接记录
  `original=5 candidates=4 to_select=2`
  （`outputs/runs/D1_full_rerun_20260624.txt:22773`）。
- D4 先由 `infer_source_selection_feature_columns()` 取得共享数值列并排除名称含
  leakage 关键字的列（`data_preprocessing.py:811-886`），再由
  `filter_model_input_feature_cols()` 排除 `*_id`/identifier 列
  （`msml_tl_rfe.py:99-101, 1194-1203`）。按当前 D4 固化 parquet schema，
  排除 ID、leakage 和目标 `sales` 后，RFE 候选列为：
  `stock_hour6_22_cnt, activity_flag, discount, holiday_flag, precpt,
  avg_temperature, avg_humidity, avg_wind_level, year, month, week, day`，
  共 12 列，因此 `ceil(12 × 0.5) = 6`。D4 入口没有覆盖 `keep_ratio`
  （`src/utils/entity_experiment.py:212-228`），故使用
  `experiment_runner.py:1077-1078` 的 0.5 默认值。

说明：表中“候选特征实际保留比例”严格按 RFE 候选列计算；最后一列单独展示
`use_sales_as_history_input=True` 时把 `sales` 加回后的模型输入比例，二者口径
不能混用。

## 3. `IndexError` fallback 路径的实际行为

**结论标签：RISK**

### (a) 触发条件

`try` 块只包含 `rfe.fit(X, y)` 和从 `rfe.support_` 计算索引的语句
（`msml_tl_rfe.py:252-255`）。因此可由代码确定的触发条件是：

> `rfe.fit(X, y)` 或紧随其后的 `np.where(rfe.support_)[0]` 在执行期间向外传播
> `IndexError`。

代码没有按样本数、特征数或某个具体下标显式判断，也没有检查异常消息。
`msml_tl_rfe.py:256-257` 的注释提到“较大样本组合下触发越界”，但这只是注释，
不足以确定 sklearn 内部的实际根因。若没有异常 traceback，内部越界的精确条件为
**UNKNOWN**，不应猜测。

### (b) fallback 最终如何选列

fallback 先执行同一个 `estimator.fit(X, y)`（`msml_tl_rfe.py:262`），随后：

1. estimator 有 `feature_importances_`：按重要性降序取前 `num_to_select`
   （`msml_tl_rfe.py:263-265`）；
2. 否则有 `coef_`：按系数绝对值降序取前 `num_to_select`
   （`msml_tl_rfe.py:266-268`）；
3. 两者都没有：取原始顺序最前面的 `num_to_select` 个索引
   （`msml_tl_rfe.py:269-270`）。

之后还会过滤非法索引、去重、不足时按原列顺序补齐、过多时截断，最终严格限制到
`num_to_select`（`msml_tl_rfe.py:272-290`）。

**明确结论：fallback 是“有筛选”，不是“全量保留”。** 只有当
`num_to_select == num_candidates` 时结果才会碰巧覆盖全量候选列；D1/D4 的
`keep_ratio=0.5` 不属于这种情况。

风险点是：该路径用一次完整 estimator 拟合后的全局重要性排序替代逐轮递归消除，
选择数量一致，但算法过程不等同于正常 RFE；并且捕获的是上述区域中的任意
`IndexError`，范围较宽。

### (c) 与正常路径的列名格式是否一致

一致。正常路径和 fallback 最终都把整数索引交给同一条语句：

```text
rfe_selected_features = [rfe_candidate_cols[int(i)] for i in selected_indices]
```

见 `msml_tl_rfe.py:272-290`。之后可选地在开头加回字符串列名 `"sales"`，
并去重得到 `final_selected_features`（`msml_tl_rfe.py:294-303`）。
`selected_feature_cols` 也是该字符串列表的别名（`msml_tl_rfe.py:312-329`）。

`apply_selected_features_to_df()` 将其作为列名序列使用、去重并检查缺列
（`msml_tl_rfe.py:432-470`），所以 fallback 与正常路径在列名格式上兼容，
不会因为路径不同而产生索引列表/列名列表类型不一致的问题。

## 4. source 序列的数据范围与 CNN 训练输入

**结论标签：MATCH**

### `selected_source_sequences` 的范围

每个选中 source 通过 `entity_id` 和 `item_id` 从整个 `source_df` 过滤得到
`source_sequence_df`，没有在此处按 train/val/test 截断；随后原样保存到
`selected_source_sequences[source_key]`
（`msml_tl_rfe.py:1149-1166`）。因此这里的结论是：

**`selected_source_sequences` 保存完整 source DataFrame 序列，包含随后可切分出的
train/val/test 全部行。**

与此同时，RFE 联合训练数据走另一条变量链：完整 `source_sequence_df` 先经
`_prepare_source_split()`，只把 `src_train` 加入
`selected_source_train_dfs`（`msml_tl_rfe.py:1168-1171`），再与
`target_train_df` 拼接（`msml_tl_rfe.py:1194-1204`）。所以 RFE 拟合没有使用
source val/test 行。

### RFE 投影和 source CNN 的实际输入

- Step 6 对 `selected_source_sequences` 中的**完整序列**应用所选列，生成
  `selected_source_sequences_rfe[source_key]`
  （`msml_tl_rfe.py:1258-1261`）。
- Step 7 取出变量 `source_sequence_df_rfe`，将它作为参数
  `source_sequence_df` 传给 `train_source_cnn_for_msml_rfe()`
  （`msml_tl_rfe.py:1294-1314`）。
- 该函数内部再次调用 `_prepare_source_split(source_sequence_df)`，按 source
  规则切成 80% train、10% val、10% test
  （`msml_tl_rfe.py:109-119, 535-544`）。
- 训练序列由 `src_train` 生成，实际变量名是 `X_train, y_train`
  （`msml_tl_rfe.py:546-551`）。
- `model.fit()` 的实际主输入为 `X_train, y_train`
  （`msml_tl_rfe.py:584-598`）；`X_val, y_val` 仅作为
  `validation_data`，`src_test` 没有进入 `model.fit()`。

因此需区分两个层次：传给 source CNN 包装函数的是**完整投影序列**，但真正用于
参数拟合的是该函数内部重新切出的**仅 train 部分**。不存在“完整 source 序列全部
直接进入 CNN fit”的行为。

## 最终判定

1. estimator：默认 `RandomForestRegressor(n_estimators=10,
   random_state=42)`，每次 RFE 调用新建；D1/D4 均使用该默认类型。
2. 选择数量：比例动态计算，D1 为 2/4，D4 为 6/12，候选特征保留比例均为 50%。
3. `IndexError` fallback：**有筛选**，按 estimator 重要性/系数（或原顺序）
   选到目标数量，不是无条件全量保留；列名格式与正常路径一致。
4. source 数据：`selected_source_sequences` 是**完整序列**；source CNN
   `model.fit` 则是**仅 train**，实际输入变量为 `X_train, y_train`。
