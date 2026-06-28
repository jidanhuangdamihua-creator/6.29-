# paper_ambiguities_and_engineering_defaults.md

## 1. 论文冲突项

| 冲突项 | 论文位置 | 影响 | 审计处理 |
| ---- | ---- | ---- | ---- |
| Source 权重公式：正文/表格使用 inverse distance，Algorithm 1 使用 distance/sum_distance | Section 3、Section 4.3.1、Table 5/6、Algorithm 1 lines 13–16 | 影响 MSWA prediction aggregation 与 MSML weights/biases fusion | 代码必须记录 `weight_formula`；建议跑 text-aligned 与 pseudo-code-aligned 双版本 |
| Algorithm 1 注释称用 target domain test data 训练新 CNN，但 line 19 实际是 Target_train_data/Target_val_data | Algorithm 1 | 若误用 test 训练会严重泄漏 | 按 line 19 与正文使用 train/val；标记算法注释疑似笔误 |
| Dataset 3 特征：正文提 store type，Fig. 9 显示 Customer/Open/Promotion/Holiday，不显示 store type | Section 5.1/5.2；Fig. 9 | 影响 Dataset 3 特征维度和 KNN/RFE | 代码记录实际字段；store_type 作为可选 paper-text feature，不强行写入 Fig. 9 对齐版本 |
| Table 8 with-info Mean RMSE 表格 0.1937，正文写 0.1973 | Table 8 与正文 | 影响综合结果引用 | 报告表格值与正文值冲突，不自行修正为唯一事实 |
| Table 13 9-source time 表格 3915，正文写 3519 | Table 13 与正文 | 影响 runtime 复现 | 优先记录表格值，同时标记正文冲突 |
| horizon 描述：提到 1/15/30 参数示例，但实验限制为 1–5 days | Section 5.3；Section 6.1 | 影响实验矩阵 | paper-aligned 结果使用 1–5 days；15/30 不作为论文结果默认 |
| RFE 范围：Section 4.3.4 说 mainly selected from similar source data，Algorithm 1 对 target/source 都 RFE | Section 4.3.4；Algorithm 1 | 影响 feature selection 与泄漏风险 | 记录 `rfe_fit_scope` 与 `rfe_transform_scope`；必要时双版本消融 |

## 2. 论文未明确但必须决定的工程默认项

| 默认项 | 推荐记录字段 | 不能写成论文事实的原因 |
| ---- | ---- | ---- |
| random seed | `random_seed`, `repeat_seed_list` | 论文未给 seed。 |
| epochs | `source_epochs`, `target_epochs` | 论文未给训练轮数。 |
| batch size | `batch_size` | 论文未给 batch size。 |
| optimizer | `optimizer` | 论文未给 optimizer。 |
| learning rate | `learning_rate` | 论文未给 LR。 |
| loss function | `loss` | 论文只给 RMSE evaluation，未给 training loss。 |
| early stopping | `early_stopping`, `patience`, `monitor` | 论文未提。 |
| CNN filters | `conv_filters` | 论文未给 filters。 |
| CNN kernel size | `kernel_size` | 论文未给 kernel size。 |
| activation | `activation`, `output_activation` | 论文未给 activation。 |
| RFE estimator | `rfe_estimator` | 论文未给 estimator。 |
| RFE n_features_to_select | `rfe_n_features_to_select`, `feature_keep_ratio` | 论文只说 40%–60%。 |
| dynamic LT 随机分布 | `dynamic_lt_delay_distribution` | 论文只说 3 days + additional 1–2 days delay。 |
| cost model daily event order | `cost_model_step_order` | 论文无成本仿真伪代码。 |
| metric space | `metric_space=normalized/original` | 论文未明确 RMSE 是否全部在 normalized/original 空间。 |

## 3. Paper-aligned 默认项

| 默认项 | 论文依据 | 审计要求 |
| ---- | ---- | ---- |
| target observed window | Section 5.4.1 30 days target sales data for KNN | KNN 不使用 target test。 |
| base source count | Section 5.4.1, Algorithm 1 | `num_sources=3`。 |
| sensitivity source count | Section 6.3, Table 13 | `num_sources=6/9` 仅作为 sensitivity。 |
| window size | Section 5.3, Fig. 10 | reported 1-day/5-day experiments 使用 `window_size=10`。 |
| CNN layer sequence | Section 4.1, Fig. 3 | Conv1D-MaxPool-Conv1D-MaxPool-Conv1D-Flatten-Dense。 |
| frozen layers | Section 4.2, Fig. 4 | 冻结 Conv1D_1, MaxPool_1, Conv1D_2, MaxPool_2。 |
| fixed LT | Section 6.2.1, Table 11 | 5 days。 |
| dynamic LT | Section 6.2.2, Table 12 | 3+2 days；具体随机生成需工程默认。 |
