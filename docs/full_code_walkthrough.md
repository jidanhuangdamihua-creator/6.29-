# Full Code Walkthrough

读图版索引：见 [docs/full_code_walkthrough_visual_index.md](docs/full_code_walkthrough_visual_index.md)

本文档给出当前论文复现仓库从原始 CSV 到最终 RMSE/Accuracy 输出的完整执行路径。

说明：仓库中存在“根目录模块”和 `src/` 下镜像模块两套实现，主运行脚本默认走根目录模块；本文以当前实际运行链路为主，同时给出 `src/` 对应路径。

---

## 1. Project Entry Point

### 调用路径

1. 单实验主入口：`run_main_experiment.py` 或 `scripts/run_main_experiment.py`
2. 论文矩阵入口：`scripts/run_full_paper_experiments.py`
3. Smoke 对齐入口：`scripts/run_paper_alignment_smoke_test.py`

核心调用链（单实验）：

`main()`
→ `run_all_experiments(...)`
→ `results_to_dataframe(...)`
→ `save_results_to_csv(...)`
→ `run_result_visualization(...)`

### 关键代码片段

```python
# run_main_experiment.py
experiment_result = run_all_experiments(...)
results_df = results_to_dataframe(experiment_result)
save_results_to_csv(results_df, str(raw_csv_path))
report = run_result_visualization(csv_path=str(raw_csv_path), output_dir=...)
```

### 输入数据结构

- 配置：`configs/default_config.json`
- 数据路径映射：`dataset_paths`
- 方法列表：`enabled_methods`

### 输出数据结构

- 原始实验结果：`DataFrame(method, rmse, accuracy, prediction_shape)`
- 格式化排名表与图像路径

### 数学含义

入口层不做建模计算，负责把“实验配置空间”映射为“可执行方法调用图”。

---

## 2. Dataset Loading

实际加载函数位于：

- 根目录：`data_preprocessing.py`
- `src` 对应：`src/data_processing/data_preprocessing.py`

### 调用路径

`run_all_experiments(...)`
→ `prepare_base_data_for_experiments(...)`
→ `load_dataset(dataset_name, data_path)`

### 关键代码片段

```python
raw_df = pd.read_csv(data_path)
if name == "dataset1":
    df = _standardize_dataset1(raw_df)
elif name == "dataset2":
    df = _standardize_dataset2(raw_df)
elif name == "dataset3":
    df = _standardize_dataset3(raw_df)
```

### 输入数据结构

- `dataset_name`: `Dataset1`/`Dataset2`/`Dataset3`
- `data_path`: CSV 路径
- 原始列（因数据集而异）

### 输出数据结构

统一输出最少包含：

- `date`
- `entity_id`
- `item_id`
- `sales`

### 数学含义

这是“样本空间标准化”，把不同数据源映射到同构特征域，保证后续距离计算、滑窗与损失计算可比较。

---

## 3. Data Cleaning

### 调用路径

`load_dataset(...)` 内部完成清洗。

### 关键代码片段

```python
df = _ensure_base_columns(df).copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
before_drop = len(df)
df = df.dropna().sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
logger.info("rows_before_dropna=%d rows_after_dropna=%d", before_drop, len(df))
```

### 输入数据结构

- 标准化后的 DataFrame（可能含缺失）

### 输出数据结构

- 去除缺失、日期可解析、按时序和实体排序后的 DataFrame

### 数学含义

清洗后数据保证了时序单调性和可计算性，避免 NaN 传播到距离度量和 RMSE 中导致无定义值。

---

## 4. Datetime Feature Extraction

### 调用路径

`prepare_base_data_for_experiments(...)`
→ `extract_datetime_features(df)`

### 关键代码片段

```python
out["year"] = out["date"].dt.year
out["month"] = out["date"].dt.month
out["week"] = out["date"].dt.isocalendar().week.astype(int)
out["day"] = out["date"].dt.day
```

### 输入数据结构

- 至少含 `date/entity_id/item_id/sales`

### 输出数据结构

- 新增时间特征列：`year/month/week/day`

### 数学含义

把时间索引映射为可学习的离散统计特征，等价于引入季节性与周期位置信号。

---

## 5. Source / Target Split

### 调用路径

`prepare_base_data_for_experiments(...)`
→ `build_source_target_split(df, config)`

### 关键代码片段

```python
source_items, target_items = _infer_source_target_items(sorted_df, config)
source_df = sorted_df[sorted_df["item_id"].isin(source_items)].copy()
target_df = sorted_df[sorted_df["item_id"].isin(target_items)].copy()
```

### 输入数据结构

- 带时间特征的全量样本

### 输出数据结构

- `source_df`: 源域池（完整历史）
- `target_df`: 目标域（最近窗口，默认 train+val 约 30 天，test 约 180 天）

### 数学含义

该步骤定义迁移学习域：

- 源域分布 $P_S(X, y)$
- 目标域分布 $P_T(X, y)$

后续方法本质是在 $P_S \to P_T$ 之间做知识迁移。

---

## 6. Source Selection

### 调用路径

`run_msml_tl(...)` / `run_msml_tl_rfe(...)`
→ `SourceSelector.select_top_k_sources(...)`
→ `compute_euclidean_distances(...)`

### 关键代码片段

```python
distances = np.linalg.norm(src - tgt, axis=1)
sorted_indices = np.argsort(distances)
selected_indices = sorted_indices[:top_k]
selected_weights = self.compute_source_weights(selected_distances, mode=weight_mode)
```

### 输入数据结构

- `target_df`（目标序列）
- `source_df`（源池）
- `feature_cols`
- `k`

### 输出数据结构

```text
{
  meta: {weight_mode, target_signature_dim, feature_cols},
  sources: [{source_key, distance, weight}, ...]
}
```

### 数学含义

1. 先构造签名向量 $z_t, z_i$
2. 计算欧式距离 $d_i = ||z_i - z_t||_2$
3. 选最小的 Top-k
4. 权重（inverse mode）：

$$
w_i = \frac{1/(d_i+\epsilon)}{\sum_j 1/(d_j+\epsilon)}
$$

---

## 7. Feature Selection (RFE)

### 调用路径

`run_msml_tl_rfe(...)`
→ `build_joint_rfe_training_dataframe(...)`
→ `run_rfe_feature_selection(...)`

### 关键代码片段

```python
num_to_select = max(1, int(np.ceil(num_original * keep_ratio)))
rfe = RFE(estimator=estimator, n_features_to_select=num_to_select, step=1)
rfe.fit(X, y)
selected_cols = [cols[i] for i in np.where(rfe.support_)[0]]
```

### 输入数据结构

- 联合训练集：`target_train + selected_sources_train`
- 候选特征列：通常为 `sales/year/month/week/day`

### 输出数据结构

```text
{
  selected_feature_cols,
  num_selected_features,
  num_original_features,
  keep_ratio
}
```

### 数学含义

RFE 通过迭代剔除低贡献特征，近似求解“在固定特征数约束下最小化预测误差”的子集选择问题。

说明：你提到的 `sales/year/day` 是一种可能结果，是否出现取决于当前数据分布与随机森林重要性，不是硬编码固定输出。

---

## 8. Source CNN Training

### 调用路径

`run_msml_tl_rfe(...)`
→ `train_source_cnn_for_msml_rfe(...)`

### 关键代码片段

```python
src_train, src_val, src_test, _, _ = normalize_features(src_train, src_val, src_test)
X_source, y_source = build_tabular_sequence(src_train, horizon=horizon, window_size=window_size)
X_source = to_cnn_tensor(X_source)
model = build_base_cnn(input_shape, learning_rate=learning_rate)
model.fit(X_source, y_source, epochs=source_epochs, batch_size=batch_size, verbose=1)
```

### 输入数据结构

- 单个 source 序列 DataFrame
- `window_size`, `horizon`

### 输出数据结构

```text
{
  model,
  input_shape: (window_size, num_features),
  num_samples,
  source_key
}
```

### 数学含义

通过滑窗构造监督样本，学习函数 $f_\theta: \mathbb{R}^{w \times d} \to \mathbb{R}$。

典型输入 shape 形式为 `(samples, 10, feature_count)`，例如 `(1450, 10, 5)`（具体样本数随数据切分变化）。

---

## 9. Multi-Source Parameter Fusion

### 调用路径

`run_msml_tl_rfe(...)`
→ `fuse_source_models_layerwise(source_models, weights, layer_names)`

### 关键代码片段

```python
for name in layer_names:
    per_source = [all_params[i][name] for i in range(len(source_models))]
    fused[name] = weighted_average_layer_params(per_source, weights)
```

其中 `weighted_average_layer_params`：

```python
acc += float(w[s_idx]) * layer_params_list[s_idx][p_idx].astype(np.float64)
```

### 输入数据结构

- 多个 source CNN
- 对应权重
- 层名（默认 `conv1/conv2/conv3`）

### 输出数据结构

- `fused_params[layer_name] = [avg_kernel, avg_bias, ...]`

### 数学含义

逐层参数融合：

$$
\Theta^{(l)}_{fused} = \sum_{i=1}^{k} w_i \Theta^{(l)}_i
$$

其中 $l \in \{conv1, conv2, conv3\}$。

---

## 10. Target Fine-Tuning

### 调用路径

`run_msml_tl_rfe(...)`
→ `load_fused_params_into_target_model(...)`
→ `freeze_fused_layers(...)`
→ `fine_tune_fused_target_model_rfe(...)`

### 关键代码片段

```python
frozen_layers = freeze_fused_layers(target_model, layer_names)
target_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
history = target_model.fit(X_train, y_train, validation_data=(X_val, y_val), ...)
```

### 输入数据结构

- 融合后 target 模型
- target train/val（已 RFE，后续再归一化）

### 输出数据结构

- 微调后的模型
- 训练历史
- 冻结层名列表

### 数学含义

冻结 `conv1/conv2/conv3` 的目的：保留跨源迁移得到的共享低层表示，仅在高层参数上做目标域适配，降低过拟合与灾难性遗忘。

---

## 11. Prediction Generation

### 调用路径

`evaluate_msml_rfe_model(...)`
内调用：`target_model.predict(X_test, verbose=0)`

### 关键代码片段

```python
y_pred = target_model.predict(X_test, verbose=0)
y_pred_flat = y_pred.flatten()
y_true = y_test.flatten()
```

### 输入数据结构

- `X_test`: `(samples, window_size, num_features)`

### 输出数据结构

- `y_pred`, `y_true`
- `prediction_shape`（如 `(1720, 1)`）

### 数学含义

模型推理产生条件期望近似 $\hat{y}_t = f_\theta(X_{t-w+1:t})$。

---

## 12. Evaluation Metrics

### 调用路径

`run_msml_tl_rfe(...)`
→ `evaluate_msml_rfe_model(...)`

### 关键代码片段

```python
rmse = float(np.sqrt(np.mean((y_pred_flat - y_true) ** 2)))
accuracy = float(1.0 / (rmse + eps))
```

### 输入数据结构

- `y_true`, `y_pred`

### 输出数据结构

```text
{
  rmse,
  accuracy,
  y_pred,
  y_true,
  prediction_shape
}
```

### 数学含义

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

$$
\mathrm{Accuracy} = \frac{1}{\mathrm{RMSE} + \epsilon}
$$

当前实现在归一化后的空间计算误差（normalized scale）。

补充（严格论文模式）：

- 入口脚本会先执行协议预检（metric/split/source）。
- `paper_metric_space` 与 `paper_accuracy_definition` 若未确认，会在报告中保留 `TODO_*`，不伪装为已对齐。

---

## 13. Result Saving

### 调用路径

`run_main_experiment.py`
→ `results_to_dataframe(...)`
→ `save_results_to_csv(...)`

### 关键代码片段

```python
raw_csv_path = experiment_results_dir / f"{dataset_name.lower()}_results.csv"
save_results_to_csv(results_df, str(raw_csv_path))
```

### 输入数据结构

- 实验结果字典 `experiment_results`

### 输出数据结构

- CSV：`outputs/experiment_results/*.csv`
- 列：`method, rmse, accuracy, prediction_shape`

### 数学含义

该阶段不引入新计算，只做结果结构化持久化，保证可复核与可视化复用。

---

## 13.5 Strict Paper Alignment Guardrail

### 调用路径

`scripts/run_main_experiment.py` / `scripts/run_full_paper_experiments.py` / `scripts/run_paper_alignment_smoke_test.py`
→ `resolve_strict_paper_mode(...)`
→ `validate_paper_protocol_config(...)`
→ `run_all_experiments(...)`

### 关键行为

1. 统一协议预检：先检查配置结构是否满足论文轨道硬约束。
2. split 断言：strict 模式下窗口日期跨度必须等于 `target_train_val_days + target_test_days`。
3. source cap 约束：多源 TL 论文轨道最多 5 个预训练模型。

### 独立校验脚本

- `scripts/validate_paper_protocol_strict.py`
- 输出：
  - `outputs/paper_alignment_reports/paper_protocol_strict_validation.csv`
  - `outputs/paper_alignment_reports/paper_protocol_strict_validation.json`

---

## 14. Result Visualization

### 调用路径

`run_main_experiment.py`
→ `run_result_visualization(csv_path, output_dir)`

### 关键代码片段

```python
results_df = load_results_csv(csv_path)
sorted_df = sort_results_by_rmse(results_df)
ranked_df = add_rank_column(sorted_df, metric_col="rmse", ascending=True)
plot_rmse_bar_chart(formatted_df, str(rmse_plot_path))
plot_accuracy_bar_chart(formatted_df, str(accuracy_plot_path))
```

### 输入数据结构

- 结果 CSV（至少包含 `method/rmse/accuracy/prediction_shape`）

### 输出数据结构

- 排名表 CSV：`*_results_formatted.csv`
- RMSE 图：`*_rmse_bar.png`
- Accuracy 图：`*_accuracy_bar.png`

### 数学含义

以 RMSE 升序排序等价于按最小化损失排序；Accuracy 是 RMSE 的单调变换，因此二者排序通常一致。

---

## Appendix: 完整执行调用链

### A. 单实验全方法链

`run_main_experiment.main()`
→ `run_all_experiments()`
→ `prepare_base_data_for_experiments()`
→ `load_dataset()`
→ `extract_datetime_features()`
→ `build_source_target_split()`
→ `run_msml_rfe_experiment()`
→ `run_msml_tl_rfe()`
→ `select_top_k_sources()`
→ `run_rfe_feature_selection()`
→ `train_source_cnn_for_msml_rfe()`
→ `fuse_source_models_layerwise()`
→ `fine_tune_fused_target_model_rfe()`
→ `evaluate_msml_rfe_model()`
→ `results_to_dataframe()`
→ `save_results_to_csv()`
→ `run_result_visualization()`

### B. 你给出的目标链（与当前实现对齐）

`run_experiment()`
→ `run_msml_tl_rfe()`
→ `train_source_cnn_for_msml_rfe()`
→ `fuse_source_models_layerwise()`
→ `fine_tune_fused_target_model_rfe()`
→ `evaluate_msml_rfe_model()`

---

## Quick Reference: 关键文件索引

- 入口：`run_main_experiment.py`
- 论文矩阵调度：`scripts/run_full_paper_experiments.py`
- 统一实验运行器：`experiment_runner.py`
- 数据处理：`data_preprocessing.py`
- 源选择：`source_selector.py`
- MSML：`msml_tl.py`
- MSML-RFE：`msml_tl_rfe.py`
- 可视化：`result_visualizer.py`
