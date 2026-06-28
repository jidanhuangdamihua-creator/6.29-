# `msml_tl_rfe.py` 四维只读审核报告

## 审核范围

- 主要对象：根目录 `msml_tl_rfe.py`
- 调用链上下文：根目录 `experiment_runner.py`
- 为确认实际权重计算路径，辅助阅读：根目录 `source_selector.py`、`msml_tl.py`
- 为确认网络层名称，辅助阅读：`src/models/cnn_model.py`
- 本次审核未运行实验，未修改任何 Python 文件、parquet 文件或 KNN JSON 文件。

## 结论摘要

| 维度 | 行为标签 | 论文对比标签 | 结论摘要 |
|---|---|---|---|
| A. RFE fit 范围 | **CLEAN** | **MATCH** | RFE 仅拟合 `target_train_df` 与各 selected source 的 `src_train` 联合数据，不含 target/source 的 val、test 行。 |
| B. 权重融合公式 | **INVERSE（默认调用路径）** | **MISMATCH** | 默认使用归一化逆距离权重；权重由 `SourceSelector` 在代码内部计算，不从 KNN JSON 读取。代码另支持显式 `raw_distance` 模式，但默认未使用。 |
| C. 冻结层范围 | **MATCH** | **MATCH** | 融合 `conv1`、`conv2`，冻结 `conv1`、`pool1`、`conv2`、`pool2`；`conv3`、`flatten`、`dropout`、`dense_out` 保持可训练。fine-tune 使用 target train，target val 仅作验证，不混入 test。 |
| D. source/target 特征列一致性 | **SAME** | **MATCH** | 全流程只有一次联合 RFE fit；其产出的同一组列名被明确保存并同时应用到 target train/val/test 和所有 selected source。 |

## 第一步：调用链与数据传递

### 相关函数与行号

- `experiment_runner.py:614-675`：`prepare_base_data_for_experiments()`
- `experiment_runner.py:1135-1164`：`run_all_experiments()` 的参数定义
- `experiment_runner.py:1207-1215`：取得 `source_df`、`target_df`
- `experiment_runner.py:1369-1395`：调用 `run_msml_rfe_experiment()`
- `experiment_runner.py:1068-1132`：`run_msml_rfe_experiment()` 调用根目录 `msml_tl_rfe.run_msml_tl_rfe()`
- `msml_tl_rfe.py:946-977`：`run_msml_tl_rfe()` 参数定义
- `msml_tl_rfe.py:1077-1092`、`1140-1147`：target 内部切分

### 实际行为

`prepare_base_data_for_experiments()` 先从预处理数据构造 source domain 与 target domain，返回尚未在该调用点拆成 train/val/test 的 `source_df` 和 `target_df`。`run_all_experiments()` 将这两个 DataFrame 以及候选 `feature_cols`、source 数量、`weight_mode`、RFE 参数和训练参数传给 `run_msml_rfe_experiment()`；后者原样调用根目录的 `run_msml_tl_rfe()`。

默认调用路径中：

- `run_all_experiments()` 的 `weight_mode` 默认值为 `"inverse_distance"`（`experiment_runner.py:1144`）。
- `run_msml_rfe_experiment()` 将 `source_selection_window` 缺省为 `"target_observed_window"`（`experiment_runner.py:1122`），将 `full_target_df` 缺省为 `None`。
- `run_msml_tl_rfe()` 在 source selection 前先通过 `temporal_split_by_ratio_or_dates(target_df)` 得到 `target_train_df`、`target_val_df`、`target_test_df`；默认 source selection 使用 target train+val 组成的 observed window（`msml_tl_rfe.py:1084-1092`）。
- 后续 RFE 使用内部已经切分出的训练部分，而不是直接使用完整 `target_df`。

## 维度 A：RFE fit 范围

### 相关函数与行号

- `msml_tl_rfe.py:109-119`：`_prepare_source_split()`，source 按 0.8/0.1/0.1 切分
- `msml_tl_rfe.py:353-425`：`build_joint_rfe_training_dataframe()`
- `msml_tl_rfe.py:1140-1171`：target 切分及各 selected source 的 `src_train` 提取
- `msml_tl_rfe.py:1194-1218`：构造 `joint_train_df` 并调用 RFE
- `msml_tl_rfe.py:140-346`：`run_rfe_feature_selection()`
- `msml_tl_rfe.py:203-211`：`X = train_df[rfe_candidate_cols]`、`y = train_df[target_col]`
- `msml_tl_rfe.py:251-254`：`rfe.fit(X, y)`
- `data_preprocessing.py:710-774`：`temporal_split_by_ratio_or_dates()`

### 代码实际行为

1. target 在 `run_msml_tl_rfe()` 内先切分为 `target_train_df`、`target_val_df`、`target_test_df`。
2. 每个 selected source 通过 `_prepare_source_split()` 切分，只把返回的 `src_train` 加入 `selected_source_train_dfs`。
3. `build_joint_rfe_training_dataframe()` 只拼接：
   - `target_train_df`
   - 每个 selected source 的 `src_train`
4. `run_rfe_feature_selection(train_df=joint_train_df, ...)` 从该联合训练表构造 `X` 和 `y`，随后执行 `rfe.fit(X, y)`。
5. RFE 的正常路径与 `IndexError` 兜底路径都只拟合相同的联合训练 `X, y`；兜底路径为 `estimator.fit(X, y)`（`msml_tl_rfe.py:255-268`）。

因此，RFE fit 不包含 target val/test，也不包含 selected source 的 val/test。

### 与论文 Algorithm 1 的对比

RFE 特征选择发生在训练部分上，验证与测试行未进入 fit，符合训练阶段特征选择不接触验证/测试数据的边界。

### 结论

- **行为标签：CLEAN**
- **论文对比标签：MATCH**
- 关键 fit 变量：`X`、`y`
- `X` 的来源变量：`joint_train_df`
- `joint_train_df` 的组成：`target_train_df` + `selected_source_train_dfs` 中的各 `src_train`

## 维度 B：权重融合公式

### 相关函数与行号

- `experiment_runner.py:1144`：顶层默认 `weight_mode="inverse_distance"`
- `experiment_runner.py:1370-1379`：将 `weight_mode` 传入 MSML-TL-RFE
- `msml_tl_rfe.py:946-955`：`run_msml_tl_rfe()` 默认 `weight_mode="inverse_distance"`
- `msml_tl_rfe.py:1111-1127`：调用 `SourceSelector.select_top_k_sources(..., weight_mode=weight_mode)`
- `source_selector.py:689-743`：`compute_source_weights()`
- `source_selector.py:1013-1037`：内部计算距离并生成 selected weights
- `source_selector.py:1063-1071`：将 `distance`、`weight` 写入本次 selection result
- `msml_tl_rfe.py:1284-1292`：从本次 `selected_sources` 结果读取 `distance`、`weight`
- `msml_tl_rfe.py:1324-1332`：将各 source 的 `weight` 加入 `source_weights`
- `msml_tl_rfe.py:1357-1361`：把 `source_weights` 传入逐层融合
- `msml_tl.py:287-335`：`weighted_average_layer_params()` 执行参数加权和

### 代码实际行为

权重不是从 KNN JSON 文件读取。当前调用链在运行时由 `SourceSelector`：

1. 计算 source 与 target signature 的欧氏距离；
2. 对选中的 source 距离调用 `compute_source_weights()`；
3. 把计算出的 `weight` 放入内存中的 `selected_sources`；
4. `msml_tl_rfe.py` 读取这些 `weight` 并用于 CNN 层参数融合。

默认模式 `"inverse_distance"` 的公式位于 `source_selector.py:726-732`：

```text
score_i = 1 / (distance_i + eps)
weight_i = score_i / sum(score)
```

融合公式位于 `msml_tl.py:330-333`：

```text
fused_parameter = sum(weight_i * source_parameter_i)
```

代码还支持显式 `"raw_distance"` 模式（`source_selector.py:733-738`），其公式是 `weight_i = distance_i / sum(distance)`；但 `experiment_runner.py` 和 `msml_tl_rfe.py` 的默认值均为 `"inverse_distance"`。

`src/utils/parquet_data_loader.py:61-71` 的 `load_knn_results()` 不在上述 MSML-TL-RFE 权重调用链中。因此，本维度不需要用 KNN JSON 的 `weight` 样例来判定实际融合权重。

### 与论文 Algorithm 1 的对比

论文给出的公式是正距离权重：

```text
weight_i = distance_i / sum(distance)
```

即距离越大，融合权重越大。默认代码路径采用逆距离权重，距离越小，融合权重越大。因此默认执行行为与论文 Algorithm 1 第 13-16 行不一致。仅在调用方显式传入 `weight_mode="raw_distance"` 时，代码提供的另一分支才与该正距离公式一致。

### 结论

- **行为标签：INVERSE（默认调用路径）**
- **论文对比标签：MISMATCH**
- **权重来源：代码内部实时计算，不从 KNN JSON 读取**

## 维度 C：冻结层范围与 fine-tune 数据

### 相关函数与行号

- `msml_tl_rfe.py:63-65`：默认冻结/融合层常量
- `msml_tl_rfe.py:1357-1372`：融合、加载与冻结调用
- `msml_tl.py:215-237`：`get_transferable_layer_names()`
- `msml_tl.py:431-459`：`freeze_fused_layers()`
- `src/models/cnn_model.py:142-155`：基础 CNN 的层顺序与名称
- `msml_tl_rfe.py:634-752`：`fine_tune_fused_target_model_rfe()`
- `msml_tl_rfe.py:676-687`：从 target train/val 构造微调输入
- `msml_tl_rfe.py:717-724`：fine-tune 的 `target_model.fit()`
- `msml_tl_rfe.py:1384-1399`：主流程传入 `target_train_df_rfe`、`target_val_df_rfe`
- `msml_tl_rfe.py:1416-1425`：test 仅用于最终评估

### 代码实际行为

基础 CNN 的具名层顺序为：

| Keras 层索引 | 层名 | fine-tune 时状态 |
|---:|---|---|
| 0 | input layer | 无可训练参数 |
| 1 | `conv1` | 冻结 |
| 2 | `pool1` | 冻结；该层本身无可训练参数 |
| 3 | `conv2` | 冻结 |
| 4 | `pool2` | 冻结；该层本身无可训练参数 |
| 5 | `conv3` | 可训练 |
| 6 | `flatten` | 可训练标志为 True；本身无可训练参数 |
| 7 | `dropout` | 可训练标志为 True；本身无可训练参数 |
| 8 | `dense_out` | 可训练 |

参数实际融合层是 `_DEFAULT_FUSION_LAYERS = ["conv1", "conv2"]`。冻结名单由 `get_transferable_layer_names()` 返回 `["conv1", "pool1", "conv2", "pool2"]`。`freeze_fused_layers()` 对名单内的层设置 `layer.trainable = False`，对其他层设置 `layer.trainable = True`。

fine-tune 的 `model.fit()` 输入为：

- 训练数据：由 `target_train_df_rfe` 构造的 `X_train, y_train`
- 验证数据：由 `target_val_df_rfe` 构造的 `(X_val, y_val)`，通过 `validation_data` 传入
- `target_test_df_rfe` 未传入 fine-tune；只在后续 `evaluate_msml_rfe_model()` 中用于最终评估

### 与论文 Algorithm 1 / 图 7、图 8 的对比

代码将前部已融合的卷积层参数加载到 target model 后冻结 `conv1`、`conv2`；位于其后的 `conv3` 和 `dense_out` 可使用 target 数据更新。`pool1`、`pool2` 同时列入冻结名单，但池化层没有权重，冻结标志不会额外冻结 Dense 参数。Dense 输出层保持可训练。该行为与“前部 Conv 区域冻结、输出侧层使用 target 数据 fine-tune”的描述一致。

### 结论

- **行为标签：MATCH**
- **论文对比标签：MATCH**
- 冻结层：`conv1`、`pool1`、`conv2`、`pool2`
- 可训练的有参数层：`conv3`、`dense_out`
- fine-tune 数据变量：`target_train_df_rfe`；`target_val_df_rfe` 仅作 `validation_data`
- test 数据：`target_test_df_rfe` 仅用于最终评估，未混入 fine-tune

## 维度 D：source/target RFE 特征列一致性

### 相关函数与行号

- `msml_tl_rfe.py:140-346`：唯一的 RFE fit 函数 `run_rfe_feature_selection()`
- `msml_tl_rfe.py:251-254`：唯一的 `rfe.fit(X, y)`
- `msml_tl_rfe.py:290-321`：由 `rfe.support_` 索引生成并保存列名
- `msml_tl_rfe.py:432-479`：`apply_selected_features_to_df()`
- `msml_tl_rfe.py:1208-1232`：执行一次 RFE，并保存 `rfe_selected_feature_cols`、`selected_feature_cols`
- `msml_tl_rfe.py:1243-1261`：同一 `selected_feature_cols` 应用于 target 和所有 selected source
- `msml_tl_rfe.py:1302-1310`：source CNN 使用 `selected_feature_cols`
- `msml_tl_rfe.py:1386-1391`：target fine-tune 使用 `selected_feature_cols`

### 代码实际行为

RFE 并非分别在 source 与 target 上独立 fit，而是对 `joint_train_df` 执行一次联合 fit。正常路径读取 `rfe.support_`，将被选中的索引映射回 `rfe_candidate_cols`，形成 `rfe_selected_features`；最终用于模型的列名保存在 `final_selected_features` / `selected_feature_cols` 中。

代码没有在 DataFrame 应用阶段调用 `rfe.transform()`，也没有把 `rfe` 对象返回到主流程；它明确保存所选列名，然后由 `apply_selected_features_to_df()` 按同一列名列表取列：

- target：`target_train_df`、`target_val_df`、`target_test_df`
- source：`selected_source_sequences` 中的每一个完整 source sequence

未找到 target 单独创建或 fit 另一个 RFE 对象的路径。

### 与论文 Algorithm 1 的对比

source 与 target 均使用同一次 RFE 结果形成的统一特征子集，符合迁移模型输入维度和列语义一致的要求。

### 结论

- **行为标签：SAME**
- **论文对比标签：MATCH**
- 共享结果载体：`selected_feature_cols`
- 共享方式：保存统一列名后按列投影，不是分别 fit，也不是分别生成 support mask
