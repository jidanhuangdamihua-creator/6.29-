# Phase 0 协议审计

状态：`PROTOCOL_AUDIT_FREEZE`。这是协议审计冻结，不是阶段 0 验收通过。本轮没有改仓库模块，也没有启动实验。

核对范围：`消融实验/ABLATION_EXECUTION_SPEC.md` 的 §3、§4、§9、§12、§13。

D4 目标按论文表 2 定为 Store166/SKU258。主协议目标学习率是 D1、D4、D6 的 `3e-5` 和 D5 的 `3e-4`。RevIN 稳定常数定为 `1e-5`。

## 1. 每个数据集的输入、相似性、滞后和可用性

没有名为 `sales_lag_*` 的特征列。序列滞后由滑窗实现：输入是结束于 `end_idx` 的 `window_size` 天，标签是 `end_idx + horizon` 的销量。`window_size` 默认 10，正式清单也是 10。见 `src/data_processing/data_preprocessing.py:1133-1136,1230-1232` 和 `configs/ms_katl_corrected_sita_60cell_manifest.json:25`。预测偏移只有 `1, 3, 5`。见 `configs/ms_katl_corrected_sita_60cell_manifest.json:13`。场景是 `without`。见同文件第 5 行。

比较窗口是闭区间 30 天。起点加 29 天得到终点，源观测截止日等于比较窗口终点。见 `src/protocols/experiment_protocol.py:28,131-146`。源序列在训练前被裁到 `source_observation_cutoff`。见 `src/protocols/runner_adapter.py:536,645,649`。目标测试日必须晚于比较窗口终点。见 `src/protocols/runner_adapter.py:600,648`。候选源必须在这 30 个日历日上有完整、有限的比较变量，缺日不补齐 K。见 `src/protocols/candidate_pool.py:1431-1439`。

### D1

- 模型输入：`sales, year, month, week, day`。`src/protocols/transformation_identity.py:1083`。
- 相似性变量：只有 `sales`。`src/protocols/transformation_identity.py:1104`；`src/protocols/experiment_protocol.py:29,265`。
- 目标键：`store_id=1, item_id=10`。`src/protocols/experiment_protocol.py:263`。
- 比较窗口：终点 `2017-06-30`，向前 30 天，即 `2017-06-01` 至 `2017-06-30`。`src/protocols/experiment_protocol.py:24-28,137-146`；日期合同 `src/protocols/gate1_transformation.py:239`。
- 源历史合同：`2017-01-02` 至 `2017-06-30`。`src/protocols/gate1_transformation.py:239`。`SOURCE_HISTORY_DAYS=180`。`src/constants.py:4`。
- 目标划分合同：训练 `2017-06-01` 至 `2017-06-15`，验证 `2017-06-16` 至 `2017-06-30`，测试 `2017-07-01` 至 `2017-12-27`。`src/protocols/gate1_transformation.py:239`。天数 15/15/180。`configs/ms_katl_corrected_sita_60cell_manifest.json:26-28`。
- without 候选：商店只有 1，商品 1 至 9。`src/protocols/experiment_protocol.py:379-386`。目标商品 10 不在候选里。

### D4

- 模型输入：`sales, stock_hour6_22_cnt, activity_flag, discount, holiday_flag, precpt, avg_temperature, avg_humidity, avg_wind_level, year, month, week, day`。`src/protocols/transformation_identity.py:1088-1091`。
- 相似性变量：只有 `sales`。`src/protocols/transformation_identity.py:1107`；`src/protocols/experiment_protocol.py:298`。
- 消融目标定为 `166/258`（Store166/SKU258）。依据是 `论文文稿相关/4 Experiments and Results.docx` 表 2，以及 `src/transfer_methods/ms_katl_corrected_sita_60cell.py:33`。协议键列表里仍有 `166/311`（`src/protocols/experiment_protocol.py:291-296`；`src/transfer_methods/ms_katl_fto_three_track_robustness.py:124`），消融不使用它。
- 比较窗口合同：`2024-12-16` 至 `2025-01-14`。源历史 `2024-07-19` 至 `2025-01-14`。目标训练 `2024-12-16` 至 `2024-12-30`，验证 `2024-12-31` 至 `2025-01-14`，测试 `2025-01-15` 至 `2025-07-13`。`src/protocols/gate1_transformation.py:242`。
- 加载器上界与合同一致：`train_start=2024-12-16`，`test_end=2025-07-13`。`src/constants.py:278-280`。
- 候选规则：`require_same_group=False`，`domain_filter_scope=target_only`。`src/protocols/experiment_protocol.py:284-289`。`target_only` 不把域过滤作用到源池。`src/utils/source_domain_filter.py:136-140`。

### D5

- 模型输入：`sales, year, month, week, day, class, perishable, cluster, transactions, oil_price, is_holiday`。`src/protocols/transformation_identity.py:1092-1095`。
- 相似性变量：`sales, onpromotion, oil_price`。`onpromotion` 只用于比较，不是模型输入。`src/protocols/transformation_identity.py:1108`；`src/protocols/experiment_protocol.py:31,311`。
- 目标键在两套正式入口里都是 `48/1159414`。`src/transfer_methods/ms_katl_fto_three_track_robustness.py:125`；`src/transfer_methods/ms_katl_corrected_sita_60cell.py:34`。
- 合同比较窗口：`2017-01-17` 至 `2017-02-15`。源历史 `2016-08-20` 至 `2017-02-15`。目标训练 `2017-01-17` 至 `2017-01-31`，验证 `2017-02-01` 至 `2017-02-15`，测试 `2017-02-16` 至 `2017-08-14`。`src/protocols/gate1_transformation.py:243`。
- 正式测试终点是 `2017-08-14`。训练 `2017-01-17` 至 `2017-01-31`，验证 `2017-02-01` 至 `2017-02-15`，测试 `2017-02-16` 至 `2017-08-14`。`src/protocols/gate1_transformation.py:243`；`src/protocols/formal_target_scope.py:29-44`；`dataset5/manifest.json:67`；`formal-proof.json:182-185`。封存文件最大日期仍是 `2017-08-15`。加载时先去掉这一天，再做 15/15/180。`src/constants.py:284` 的上界还是 `2017-08-15`，那条划分不作为正式结果。
- 候选：同一 `family`，`require_same_group` 默认真。`src/protocols/experiment_protocol.py:300-303`。域过滤作用到源池。`src/utils/source_domain_filter.py:136-154`。

### D6

- 模型输入：`sales, year, month, week, day, wm_yr_wk, weekday, is_event_1, is_event_2, snap, sell_price`。`src/protocols/transformation_identity.py:1096-1099`。
- 相似性变量：只有 `sales`。`src/protocols/transformation_identity.py:1109`。
- 目标键在两套正式入口里都是 `CA_1/FOODS_3_080`。`src/transfer_methods/ms_katl_fto_three_track_robustness.py:126`；`src/transfer_methods/ms_katl_corrected_sita_60cell.py:35`。
- 合同：源历史 `2015-05-29` 至 `2015-11-24`；比较窗口 `2015-10-26` 至 `2015-11-24`；训练 `2015-10-26` 至 `2015-11-09`；验证 `2015-11-10` 至 `2015-11-24`；测试 `2015-11-25` 至 `2016-05-22`。`src/protocols/gate1_transformation.py:244`。
- 加载器上界与合同一致：`2015-10-26` 至 `2016-05-22`。`src/constants.py:286-288`。
- 候选：同一 `department`。`src/protocols/experiment_protocol.py:313-316`。

## 2. 源选择

现场训练在 `configure_protocol_frames` 里把选择权威写成 `shared_protocol`，表示写成 `daily_sales_flattened_30d`，缩放写成 `global_minmax_legal_observed_values`，拟合范围写成目标加全部合法候选的比较窗口观测值，对齐方式是比较窗口的精确日期。见 `src/protocols/runner_adapter.py:631-654`。协议版本是 `d1_d6_protocol_v1`。`src/protocols/experiment_protocol.py:10`。默认模式是 `aligned_30d`。`src/protocols/candidate_pool.py` 中的 `SOURCE_SELECTION_MODE_ALIGNED_30D`。

比较向量按声明的相似性列拼接。每一列是比较窗口内的逐日值。D1、D4、D6 是 30 维；D5 是 3 列，因此是 90 维。见 `src/protocols/candidate_pool.py:1425-1429`。

代码里的缩放是全体拼接值上的一个标量最小值和一个标量最大值，再做 `(x-min)/(max-min)`。全距为 0 时向量置零。距离是 `np.linalg.norm(..., axis=1)`。见 `src/protocols/candidate_pool.py:1441-1452`。权重是 `1/(distance+1e-8)` 再归一化。`src/protocols/candidate_pool.py:1499-1500`；`weight_epsilon=1e-8`，`tie_tolerance=1e-12`。`src/protocols/experiment_protocol.py:205-207`。K=3。`configs/ms_katl_corrected_sita_60cell_manifest.json:15`。

消融采用的源选择缩放是逐相似性变量 Min–Max。对变量 \(j\)，\(r_j>0\) 时按 \((x-m_j)/r_j\) 缩放；\(r_j=0\) 时该变量整列记为 0。`930 第三章.md:43`。正式代码把全部变量拼成一个向量后只用一个全局 min/max，并且只在全局全距为 0 时置零。`src/protocols/candidate_pool.py:1441-1451`。D1、D4、D6 只有销量，两种结果相同。D5 的三列不同。旧距离、Top-3 和下游正式结果不能直接继承，须先改实现并重新生成源选择权威文件。

固化 JSON 里的 Top-K 结果标注为诊断，不是训练权威。D6 JSON 头仍写运行时 `mean/std/min/max/last` 和 `scaling: none`。现场入口在选择前改写为共享协议。本轮采用现场入口，不采用 JSON 里的统计量距离。

## 3. 源模型训练

源划分是按唯一日期排序后的比例切分：`train_ratio=0.8`，`val_ratio=0.1`，`test_ratio=0.1`。`src/utils/parquet_data_loader.py:264,317-322`。切点是 `train_end=max(1, int(n*0.8))`，`val_end=max(train_end+1, int(n*0.9))`，并且不超过 `n-1`。`src/data_processing/data_preprocessing.py:690-704`。[SITA v2 匹配窗切分](b8f1d14a-59b0-4dd0-a0a4-a38715d05067) 没有用比例推算日期。它调用 `_prepare_source_split`，对象是排序后的不重复日期行，滑窗在每个切分内部单独构造，窗口不跨切分。范围只包括 D4、目标 `166/258`、without、horizon 1，以及当时 `historical_window_30d` 选出的三个源 `166/492`、`166/277`、`166/184`。三个源都是 `2024-07-19` 至 `2025-01-14` 的 180 个连续日期。实测切分是：训练行 0–143，日期 `2024-07-19` 至 `2024-12-09`；验证行 144–161，日期 `2024-12-10` 至 `2024-12-27`；测试行 162–179，日期 `2024-12-28` 至 `2025-01-14`。horizon 1、窗长 10 时，进入 `model.fit` 的只有训练目标 `2024-07-29` 至 `2024-12-09`，共 134 个样本。这不是消融默认的 `aligned_30d` 选源结果，不能把那三个匹配窗当成消融的 Top-3。

源 CNN 只在训练切片上构造窗口并 `fit`。损失是建模型时的 MSE，优化器 Adam。调用处的学习率是 `1e-4`，轮数是 50。`configs/ms_katl_corrected_sita_60cell_manifest.json:16-18`；`src/utils/experiment_hyperparams.py:5-6`。`model.fit` 没有 `validation_data`。返回值只有模型、输入形状、样本数和源键，没有验证误差。`src/transfer_methods/msml_tl.py:325-367`。验证切片被构造并做了 Min–Max 变换，随后没有被评估。源模型输入缩放只拟合这条源序列的训练日期行，不用目标 scaler，也不用源选择 scaler。切分前数值缺失填 0。`src/utils/source_fillna.py:42-57`；`src/transfer_methods/msml_tl.py:324-328`；`src/data_processing/data_preprocessing.py:1068`。

源侧验证误差定为 `NOT_APPLICABLE`，数值空间同样是 `NOT_APPLICABLE`。源模型只在训练窗口上做 MSE 优化。`src_val` 被切出，但没有进入 `model.fit`，也不参与早停、检查点选择、源模型排序或融合。`src/transfer_methods/msml_tl.py:325-367`。规范 §5.1 的字段在正式结果中不填。

融合只取 `conv1`、`conv2`、`conv3` 的 kernel 和 bias，按逆距离权重加权平均。`src/transfer_methods/msml_tl.py:503-520`。`align_fusion_filters` 不是这条正式路径的默认行为。

## 4. 两个 Min–Max

目标模型缩放是按列 `sklearn.preprocessing.MinMaxScaler`，只在目标训练行上 `fit`，验证和测试只 `transform`。`src/data_processing/data_preprocessing.py:969-982,1058-1071`。目标划分在 `paper_split_protocol` 下被改成最后 210 个唯一日期中的 15/15/180。`src/data_processing/data_preprocessing.py:750-777`。

源选择缩放不是这个 scaler。代码用比较窗口里目标向量和全部合法候选向量的一个全局标量 min/max。`src/protocols/candidate_pool.py:1441-1451`；元数据 `src/protocols/runner_adapter.py:652-653`。论文要求逐变量。`930 第三章.md:43,89`。两者分别拟合，这一点代码和论文一致。

## 5. CNN、RevIN、Huber、EMA、sMAPE

源网络 `build_base_cnn`：

- `conv1`：32 个滤波器，核 3，`padding=same`，ReLU。
- `pool1`：窗口 2。
- `conv2`：64，核 3，`same`，ReLU。
- `pool2`：窗口 2。
- `conv3`：128，核 3，`same`，ReLU。
- Flatten。
- Dropout，比率 `FIXED_DROPOUT=0.1`。
- `dense_out`：Dense(1)。
- 编译为 Adam + MSE。

见 `src/models/cnn_model.py:143-154` 和 `src/utils/experiment_hyperparams.py:8`。

目标预测路径的特征提取停在 Flatten，不走这条 Dropout 和 `dense_out`。目标头是 `MSKATLLite` 的 `head_target` Dense(1)，在展平特征上另有 `feature_dropout=0.1`。`src/transfer_methods/ms_katl_lite.py:97,293,321`。迁移支路关闭时，预测等于 `head_target(z_T)`。

RevIN 代码：

- `eps=1e-5`，仿射开启，目标通道 0。`src/transfer_methods/revin.py:19-21`。
- 沿时间轴求均值和均方，`stop_gradient`。标准差是 `sqrt(variance+eps)`。`src/transfer_methods/revin.py:83-88`。
- 反变换是 `(y-bias)/(weight+eps*eps)*stdev+mean`。`src/transfer_methods/revin.py:125-130`。

冻结值是 `eps = 1e-5`。归一化是 `sqrt(variance + 1e-5)`。仿射反变换分母是 `affine_weight + 1e-10`。`src/transfer_methods/revin.py:19,83-85,127-130`。这个常数不等于融合权重的 `1e-8`。`9.13第三章.md:168` 把两者写成同一个常数，消融不采用那种写法。

Huber：`delta=1.0`，Keras `Huber`。`configs/ms_katl_corrected_sita_60cell_manifest.json:31-32`；`src/transfer_methods/ms_katl_lite.py:106,549`。第三章给出的分段公式与这个 delta 的标准 Huber 一致，但第三章没有给出 1.0 这个数。`930 第三章.md:125`。

目标优化是 Adam。清单写的学习率日程是 cosine。`configs/ms_katl_corrected_sita_60cell_manifest.json:19-24,33-34`。实现是 `CosineDecay(initial_lr, decay_steps=epochs*steps_per_epoch)`。`src/transfer_methods/ms_katl_lite.py:541`。主协议初始学习率：D1、D4、D6 为 `3e-5`，D5 为 `3e-4`。`消融实验/ABLATION_EXECUTION_SPEC.md:88`；`configs/ms_katl_corrected_sita_60cell_manifest.json:19-23`。论文 4.1.2 中 A–E 统一的 `3e-4` 不作为主协议。轮数 50，批大小 16，不早停。`configs/ms_katl_corrected_sita_60cell_manifest.json:29-30,42`。

EMA：动量 0.99。每步 `ema = momentum*ema + (1-momentum)*variable`。循环内验证使用当前 RAW 权重。全部轮数结束后把 EMA 写回变量，测试预测看到的是 EMA。`src/transfer_methods/ms_katl_lite.py:781-782,849-852,900-902`；`formal_weight_variant=EMA`。`configs/ms_katl_corrected_sita_60cell_manifest.json:35-39`。源模型和目标模型是两套权重。

sMAPE：`100 * mean(2*|pred-true| / (|true|+|pred|+1e-8))`。`src/evaluation/metrics.py:59-79`；`src/evaluation/metric_contract.py:15-17`。分母加了 `1e-8`，样本不因为真值和预测同时为 0 而被丢弃。原销量空间先把负数裁成 0，再算指标。`src/evaluation/metrics.py:276-294`。代码没有单独返回“裁剪前分母恰好为 0 的样本数”。

## 6. TCN 参考

本仓库参考路径：

- 结构与 BL7 训练循环：`scripts/baselines/bl7_tcn.py`。
- RevIN 包装：`scripts/baselines/bl7_tcn_revin.py`。
- 可融合参数名单：`src/transfer_methods/sita_tcn.py:43-59`。
- 协议包装：`scripts/sita_tcn_protocol.py`。它不是层定义。

结构：输入通道 1，通道数 16，两层，核 3，膨胀 1 和 2。每块两次因果卷积，左侧填充 `(kernel-1)*dilation`，ReLU，通道变化时用 1×1 残差，否则恒等。只有第 0 块有可训练残差。头是 `Linear(16, 1)`，取最后时刻。`scripts/baselines/bl7_tcn.py:18-23,63-93,187-207`。SITA-TCN 的 RevIN 是 `num_features=1`。`src/transfer_methods/sita_tcn.py:142`。

可融合张量是两块的 `conv1`、`conv2` 权重和偏置，以及 `blocks.0.residual` 的权重和偏置。头和 RevIN 仿射参数不融合。`src/transfer_methods/sita_tcn.py:43-59`。

BL7 自己的拟合循环是 PyTorch、MSE、Adam `1e-4`、最多 50 轮、早停耐心 5。`scripts/baselines/bl7_tcn.py:21-23,214-236`。消融规范要求 TensorFlow、数据集学习率、50 轮、不早停、最终状态用于预测。这些训练项不能从 BL7 循环抄过来。本轮按用户指定的四个文件记录，没有另写 `configs/tcn_frozen.json`。

## 7. 规范与正式代码的差异

1. D4 目标已定为 `166/258`。三轨入口仍锁定 `166/311`（`src/transfer_methods/ms_katl_fto_three_track_robustness.py:124`），论文表 2 和纠正后的正式 SITA 使用 `166/258`。消融采用后者。
2. 源侧验证误差正式不定义、不报告。`source_validation_metric = NOT_APPLICABLE`。`src/transfer_methods/msml_tl.py:325-367`。规范 §5.1 仍列出该字段，正式结果不填。
3. D5 正式终点已定为 `2017-08-14`。封存文件和 `src/constants.py:284` 仍含 `2017-08-15`。消融加载必须先执行 `date <= 2017-08-14`。按 `2017-08-15` 切出的结果不能当作正式合同结果。
4. 源选择的可执行规则是逐变量 Min–Max；某个变量全距为 0 时整列缩放为 0。代码的全局标量是必须先改掉的差异。`930 第三章.md:43`；`src/protocols/candidate_pool.py:1441-1451`。旧 Top-3 不能直接继承。
5. RevIN 稳定常数已定为 `1e-5`，反变换分母加 `1e-10`。`src/transfer_methods/revin.py:19,125-130`。融合权重 ε 仍是 `1e-8`，两者不相等。`9.13第三章.md:168` 的“同一个常数”不采用。
6. 目标学习率日程：消融规范 §3.3 写 Adam 和数据集学习率，正式清单写 cosine。`configs/ms_katl_corrected_sita_60cell_manifest.json:33-34`。第三章没有写 cosine。按规范 §12，未写明的实现取正式 V3，因此记录为 cosine，不记成常值学习率。
7. CNN Dropout：源模型 Dropout 在 Flatten 之后、`dense_out` 之前。目标预测提取器不含这一层，另有特征 Dropout。两处比率都是 0.1。
8. 循环内验证用 RAW，测试用 EMA。规范要求核对最终预测权重；正式清单指定 EMA。
9. D6 固化 JSON 的运行时统计量距离与现场共享协议不一致。现场入口改写元数据，JSON 自标为非训练权威。
10. 协议默认视界是 1 至 5。`src/protocols/experiment_protocol.py:11`。消融只用 1、3、5。
11. BL7 有早停和 MSE。消融目标协议没有早停，损失是 Huber。
12. 仓库里没有 `scripts/run_d1_experiment.py`。D1 特征以协议合同为准。

## 8. 停止项

先前两项缺口已冻结。源侧验证误差是 `NOT_APPLICABLE`。逐变量 Min–Max 在 \(r_j=0\) 时整列缩放为 0。

状态仍是协议审计冻结，不是阶段 0 验收通过。正式选择代码还是全局标量；在它改成逐变量缩放并重新生成源选择权威文件之前，旧距离、Top-3 和下游正式结果不能直接继承。

TCN 结构见 `消融实验/configs/tcn_frozen.json`，与 `protocol_frozen.json` 的 `tcn_reference` 一致。BL7 的早停和 MSE 不进入消融训练协议。
