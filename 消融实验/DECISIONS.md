# Phase 0 决定

状态：`PROTOCOL_AUDIT_FREEZE`。这是协议审计冻结，不是阶段 0 验收通过。

## 已记录、可以沿用的实现

- 现场源选择采用 `d1_d6_protocol_v1` 的对齐 30 日扁平向量、欧氏距离和逆距离权重。固化 JSON 的 Top-K 和 D6 的运行时统计量不是训练权威。证据：`src/protocols/runner_adapter.py:631-654`。
- 目标测试预测使用 EMA，动量 0.99。循环内验证使用 RAW。证据：`src/transfer_methods/ms_katl_lite.py:849-852,900-902`。
- 目标学习率日程采用正式清单里的 cosine，初始学习率仍按数据集。消融规范正文没有写出日程名称，正式 V3 写了。证据：`configs/ms_katl_corrected_sita_60cell_manifest.json:33-34`。
- sMAPE 不删除分母为零的样本，分母加 `1e-8`。原销量空间先把负数裁成 0。证据：`src/evaluation/metrics.py:73-79,276-294`。
- TCN 结构以 `scripts/baselines/bl7_tcn.py` 为准，并写入 `configs/tcn_frozen.json`。BL7 的早停、patience 和 best_state 不进入消融训练协议。TCN 源拟合的损失和学习率以 `tcn_source_fit` 为准，不从 BL7 的早停循环继承。

## 源选择缩放

可执行规则是逐相似性变量 Min–Max。对变量 \(j\)，在目标与全部合法候选的同一比较窗口上取 \(m_j\) 和 \(M_j\)，\(r_j=M_j-m_j\)。\(r_j>0\) 时 \(g_j(x)=(x-m_j)/r_j\)；\(r_j=0\) 时该变量整列缩放为 0。缩放后的矩阵再展开，距离是两个展开向量的欧氏范数。

正式代码不一致：`src/protocols/candidate_pool.py:1441-1451` 把所有变量拼成一个向量后只用一个全局 min/max，`1446-1448` 只处理全局全距为 0。旧选择距离、Top-3、摘要和下游正式结果不能直接继承。正式实现改为逐变量缩放并重新生成源选择权威文件之后，新结果才有效。本轮不改选择代码。

## 已关闭

STOP-1。D4 目标定为 `166/258`，即 Store166/SKU258。依据是 `论文文稿相关/4 Experiments and Results.docx` 表 2，以及 `src/transfer_methods/ms_katl_corrected_sita_60cell.py:33`。三轨入口里的 `166/311`（`src/transfer_methods/ms_katl_fto_three_track_robustness.py:124`）不采用。

主协议目标学习率：D1、D4、D6 为 `3×10⁻⁵`，D5 为 `3×10⁻⁴`。依据是 `消融实验/ABLATION_EXECUTION_SPEC.md:88`、`configs/ms_katl_corrected_sita_60cell_manifest.json:19-23` 和论文表 3。论文 4.1.2 里 A–E 统一的 `3×10⁻⁴` 不替代这套主协议。

STOP-4。RevIN 稳定常数定为 `eps = 1e-5`。依据是 `src/transfer_methods/revin.py:19`。归一化使用 `sqrt(variance + 1e-5)`（`src/transfer_methods/revin.py:83-85`）。仿射反变换分母是 `affine_weight + (1e-5)^2`，即 `affine_weight + 1e-10`（`src/transfer_methods/revin.py:127-130`）。它不等于融合权重的 `1e-8`。

STOP-2。`source_validation_metric = NOT_APPLICABLE`，`source_validation_value_space = NOT_APPLICABLE`。源验证窗口不参与参数更新、早停、检查点选择、源模型排序或融合权重计算，因此不定义、不报告源侧验证误差。代码依据是 `src/transfer_methods/msml_tl.py:325-367`。规范 §5.1 仍列了这个字段（`消融实验/ABLATION_EXECUTION_SPEC.md:181`），正式结果不填它。以后若要记录源验证 RMSE，须另立诊断协议，写明 scaler、预测公式和数值空间，不得补进当前正式结果。源 CNN 的优化损失仍是 MSE，见 SOURCE-CNN-FIT。本条不规定 TCN 源损失。该损失由已关闭的 GAP-TCN-SOURCE-FIT 写成 MSE，作用在源 Min–Max 后的 sales 上。

OPEN-1。`signed_error = mean(pred - true)`。正值表示高估，负值表示低估。`pred` 和 `true` 是指标所用的原始销量，并先按冻结规则把负数裁成 0，再取平均。这与 RMSE 使用同一对裁剪后的向量。无源臂和有源臂都记录这个数。

OPEN-2。`alpha_entropy = -sum(alpha_i * ln(alpha_i))`，`ln` 是自然对数。计算熵之前把每个 `alpha_i` 裁到 `[1e-12, 1]`。裁剪后的项若为 0，约定 `0 * ln(0) = 0`。不把裁剪后的权重重新归一化，也不新增归一化熵。`N_eff = 1/sum(alpha_i^2)` 继续记在已有字段 `effective_source_count`，使用裁剪前的融合权重。无源臂没有 `alpha`，`alpha_entropy` 写 `null`。

STOP-3。`D5_FORMAL_TEST_END = 2017-08-14`。正式日历是训练 `2017-01-17` 至 `2017-01-31`，验证 `2017-02-01` 至 `2017-02-15`，测试 `2017-02-16` 至 `2017-08-14`。依据是 `src/protocols/gate1_transformation.py:243`、`数据集/固化数据/d1_d6_sealed_v1/dataset5/manifest.json:67`、`formal-proof.json:182-185`，以及 `src/protocols/formal_target_scope.py:29-44`。封存文件保持不变。`2017-08-15` 只是文件里的额外日期。**目标**加载时先保留 `date <= 2017-08-14`，再切 15/15/180。`2017-08-14` 只是目标测试终点，不是源候选截止日。现有加载器上界 `src/constants.py:284` 仍是 `2017-08-15`；按该上界得到的 D5 划分不能作为正式合同结果。

STOP-HISTORY。合法源候选必须在比较 scaler 拟合和距离计算之前同时通过四道检查：（1）冻结身份与分组；（2）冻结源历史闭区间上恰好 180 个不重复公历日，策略名 `exact_expected_date_set`；（3）比较窗口每一个日历日都在；（4）比较变量全部有限且可编码。选择 scaler 只在目标与通过这四道检查的候选上拟合。依据是正式 `src/protocols/source_history.py:137-234`、`src/source_selection/source_selector.py:1243-1254`，以及 `src/protocols/candidate_pool.py` 对比较窗口的完整日检查。`ℓ=15` 只缩短比较窗口，不缩短源历史，也不取消 180 日资格。

STOP-D5-CUTOFF。D5 的 `source_observation_cutoff` 是 `2017-02-15`，等于源历史终点和比较窗口终点。源候选身份与源历史资格只使用 `date <= 2017-02-15` 的源行。依据是 `消融实验/configs/features_frozen.json` 的 D5 `source_history`、`src/protocols/gate1_transformation.py:243`、`src/protocols/runner_adapter.py:536-551`。

STOP-D5-RECON。D5 源历史在 180 日资格检查之前，必须先按正式日历重建补齐自然日。消融在 `消融实验/` 内独立实现同等行为，禁止 `import src`，禁止改写封存 `source.parquet` / `target.parquet`，选源输入也不读取 `prepared_180day_source.parquet`。权威 CSV 路径与 SHA256 冻结为 `数据集/固化数据/d1_d6_sealed_v1/dataset5/prepared_180day_source_manifest.json` 的 `auxiliary_csvs`。重建行为（对照 `src/utils/d5_calendar_reconstruction.py:554-809`）：

- 候选实体：先把源行裁到 `[2016-08-20, 2017-02-15]`，再取该窗内出现过的 `(store_nbr, item_nbr)`（经 `entity_id` 聚合）；窗外日期不得进入重建输入。
- 对每个实体补齐期望 180 日中缺失的自然日；已有观测行保持不变（`original_rows_unchanged=True`）。
- 合成行：`sales=0`；若有 `onpromotion` 则填 `0`；`year/month/week/day` 由日期生成；`family/class/perishable` 来自 `items.csv`；`city/state/type/cluster` 来自 `stores.csv`；`transactions` 按 `(store_nbr, date)` 查表，缺失填 `0`；`oil_price` 按日期查权威油价表，缺失则失败；若列中有 `is_holiday`，按 `(store_nbr, date)` 查假期表，缺失填 `0`。
- 油价权威：`oil.csv` 的 `dcoilwtico` 对齐自然日后前向填充，再滞后一天得到 `oil_price`（同文件 `build_authoritative_d5_oil_by_date`，第 125–142 行）。
- 重建后仍须做 `exact_expected_date_set`：行数与唯一日期数均为 180，且无重复实体-日期。
- 实现完成后可用 `prepared_180day_source_manifest.json` 的 `source_history_frame_digest` 做核验，但训练与选源不得把该 parquet 当作输入。

按截止日 `2017-02-15` 重计的 D5 without 阶段数量（本轮只计身份与资格，不算距离）：封存源不重建时，历史窗内键 49868，身份过滤后（store 48 且 `GROCERY I`）1054，恰好 180 日为 0，比较窗口合法 330；按正式重建产物 `prepared_180day_source.parquet` 核验，同一身份过滤后 1054，且全部满足 180 日与比较窗口合法。目标 `48/1159414` 不在源侧。

## 接口核对

TEST-COVERAGE。测试评分覆盖冻结测试区间的每一个日期，每个 horizon 180 条。规范 §3.3 写的是“测试窗口 180 天，逐日滚动一步预测，参数固定”（`消融实验/ABLATION_EXECUTION_SPEC.md:82`）。标签位于输入窗终点再偏移 horizon 天（`消融实验/configs/features_frozen.json` 的 `lag_columns`）。因此首个测试目标日的输入窗落在验证窗内。验证日只做输入，不拟合 scaler，不选源，也不更新参数。只在测试切片内滑窗会留下未评分测试日：horizon 1 为前 10 天，horizon 3 为前 12 天，horizon 5 为前 14 天。`src/protocols/rolling_origin.py:132-159` 把第一个原点固定在验证终点，horizon 3 和 5 仍会漏掉测试起点后的前 `h-1` 天。这两种少样本构造都不采用。日期表写在 `消融实验/INTERFACES.md` 第 4 节。缺测试日时该 cell 失败，不能写完成标记。

FUSION-CACHE。`F3`、`F4`、`P1`、`P3` 共享的 Stage A 缓存只保存入选源、距离、源模型权重和各源 scaler。规范 §4.1 把“融合参数”写在这份共享缓存里（`消融实验/ABLATION_EXECUTION_SPEC.md:129`），同节又规定 `P3` 只把 α 改成 `1/3`（同文件第 127 行）。共享文件里的融合参数按源卷积权重理解。逆距离平均和均匀平均在初始化时分别计算。融合结果若落盘，缓存键包含 `weight_mode`。`F4` 与 `P1` 共用 `inverse_distance`，`P3` 只用 `uniform`，`P2` 只用 Top-1 的 `single`。

SOURCE-CNN-FIT。源 CNN 的训练设置是 Adam、MSE、学习率 `1e-4`、`clipnorm=None`、50 epoch、batch size 16，且 `fit` 不接收验证集。适用对象只是源 CNN，不是 TCN，也不是目标适配。出处：`消融实验/INTERFACES.md:337`（源 CNN 编译为 Adam 与 MSE，Adam 的 `clipnorm` 为 `None`）、`消融实验/INTERFACES.md:67` 与 `:446`（源模型学习率 `1e-4`；源训练为 Adam、MSE、`1e-4`、50 epoch、batch size 16）、`消融实验/configs/protocol_frozen.json:124-134`（`source_training.learning_rate=0.0001`，`source_training.loss=mse`，损失引用 `src/models/cnn_model.py:157-159`）、`protocol_frozen.json:162-166` 与 `DECISIONS.md` STOP-2（源模型只在源训练窗口上做 MSE，源验证不进入 `fit`）。`models.py` 的 `build_source_cnn` 按这组设置编译。

这组设置和 BL7 排除项不是同一对象。源 CNN 仍用上面的 Adam、MSE 和 `1e-4`。TCN 构建函数不 compile。TCN 源拟合不借用这一段的适用对象；它使用自己的 `tcn_source_fit`。数值有相同的地方，但早停、patience、best_state、目标 Huber、cosine 和 EMA 都不进入源 TCN。

GAP-TCN-SOURCE-FIT 已关闭。`configs/tcn_frozen.json` 的 `tcn_source_fit` 冻结为：损失 `mse`，优化器 `adam`，学习率 `1e-4`，batch size 16，50 epoch，`clipnorm` 为 JSON null，`shuffle` 为 true，`validation_used_for_fit`、`validation_for_early_stopping`、`validation_for_checkpoint_selection`、`validation_for_source_ranking` 都为 false，`ema` 为 false，`final_weights` 为 `last_epoch_raw`，`source_revin` 为 `UNUSED`，`seed` 为 `cell_seed_reset_per_stage_a_key`。MSE 在源训练切片的 Min–Max 缩放后的 sales 通道上计算。源侧不包装 RevIN。源验证切片不传入 `fit`，不参与早停、检查点、源排序、融合权重或最终状态选择；源验证记录仍是 `NOT_APPLICABLE`。每个 `StageAKey` 开始时用 cell 种子复位一次 TensorFlow，然后按入选顺序逐个拟合。CNN 与 TCN 的缓存键含骨干，互不读取。缓存写入第 50 个 epoch 结束时的 RAW 权重，不写 EMA。这组值不是 BL7 的 early stopping、patience、best_state，也不是目标侧 Huber、cosine 或 EMA。`clipnorm` 的冻结值就是 null，这个字段必须出现；其余必需字段缺失、为 null，或类型与冻结值不符时，合同不完整。不完整时不把 MSE、`1e-4` 或 50 epoch 自动补上。`SITA_TCN` 仍返回 `BLOCKED_GAP_TCN_SOURCE_FIT`，不调用 TCN 的 `fit`，也不把合成权重当成已训练结果。合同所有必需字段完整且类型正确后才解除这道阻断。`TCN_REVIN` 无源，不走这道门，目标训练仍是 Huber、cosine 和 EMA。

RUN-DIRECTORY。冻结运行目录是 `消融实验/runs/<arm>/<protocol_tag>/<dataset>_h<horizon>/seed<seed>/`（`消融实验/INTERFACES.md:71-73`）。默认 Phase 1–3 的每个 `RunSpec` 都由这五项区分：学习率差异在 `alt_lr` 与 `main`，选源差异在 `random3`、`bottom3`、`k5`、`k10`、`ell15`，臂名区分 `F`/`P` 与两支 TCN。`P1`、`P2`、`P3` 主臂使用这条路径，不再拒绝。

无 RevIN 的 `P1`、`P2`、`P3` 仍是规范 §4.1 的可选开关，`protocol_tag` 没有单独取值。它们与对应主臂只差已有字段 `revin`，若只用冻结模板会写进同一目录。因此只在这三支 `revin=False` 的路径末尾追加 `revin0`。这不是新的 `protocol_tag`、学习率或损失。`F1`、`F3` 的无 RevIN 主臂不追加，因为臂名已经不同。Stage A 键仍不含 RevIN。这条目录规则不改变 OPEN-1 与 OPEN-2 的公式。

GAP-IO。已冻结为 schema v1。`io` 只序列化、校验、哈希和安全写入，不训练、不选源、不读取封存表，也不自动重试失败 cell。`cell_record.json` 的顶层键固定为 `CELL_RECORD_KEYS`：

`alpha_entropy`、`artifact_sha256`、`config_digest`、`elapsed_seconds`、`ema_shift`、`epoch0_val_rmse`、`finished_at`、`frozen_config_sha256`、`git_head`、`gradient_steps`、`n_test`、`n_train`、`n_validation`、`pair`、`prediction_path`、`raw_shift`、`run_spec`、`signed_error`、`source_info`、`stage_a_key_sha256`、`started_at`、`status`、`tensorflow_version`、`test_weight_variant`、`train_loss`、`unscored_test_dates`、`val_loss`、`val_rmse`、`val_smape`、`validation_rmse`、`validation_smape`、`validation_weight_variant`、`workspace_dirty`。

`run_spec` 固定为 13 个字段：`arm`、`protocol_tag`、`dataset`、`horizon`、`seed`、`backbone`、`revin`、`freeze_conv`、`selection_rule`、`requested_k`、`weight_mode`、`comparison_days`、`target_learning_rate`。无源臂的 `source_info` 是字符串 `ABSENT`，顶层 `alpha_entropy` 为 `null`。有源臂的 `source_validation_metric` 与 `source_validation_value_space` 为 `NOT_APPLICABLE`。顶层 `signed_error` 是 OPEN-1 的有限数。有源臂的顶层和 `source_info.alpha_entropy` 都是 OPEN-2 的有限数。`effective_source_count` 就是 `N_eff`，不另增字段。

- `OVERLAP_COUNT`。只对 `random3` 与 `bottom3` 的已完成 cell，在 `source_info.top3_overlap_count` 写整数。定义是同一 `dataset`、`horizon`、`seed`、30 日比较窗口、`K=3` 下，该 cell 的选择结果与 top/main 选择结果的源键交集大小。两边状态都是 `COMPLETED` 时才写 0 到 3 的整数。候选不足，或任一侧不是 `COMPLETED`，不写该字段，不用 0 代替缺失。交集由 `source_selection.top3_overlap_count` 计算。`io` 只接收、校验和序列化，不在 IO 内选源。
- `CONFIG_DIGEST`。保留 `git_head`、`workspace_dirty`、`frozen_config_sha256`。新增 `config_digest = SHA256(canonical_json({"frozen_config_sha256", "run_spec"}))`。canonical JSON 为 UTF-8、递归 `sort_keys`、`separators=(",", ":")`、`allow_nan=False`，不含 `git_head`，`run_spec` 用完整 13 字段。`git_head` 是代码提交。`workspace_dirty` 是工作区状态。`frozen_config_sha256` 是三份冻结 JSON 原始字节的聚合哈希。`config_digest` 是冻结配置身份与当前 cell `RunSpec` 的组合身份。正式 runner 发现 `workspace_dirty=True` 必须拒绝启动正式实验，不用未定义的代码 diff digest 代替。
- `SAMPLE_COUNT_KEYS`。`n_train = 6 - horizon`，`n_validation = 6 - horizon`，`n_test = 180`。`n_train` 与 `n_validation` 来自实际目标 `WindowBatch`。窗口构造、adaptation 和 IO 都校验实际样本数。IO 不在字段缺失时按 horizon 补数。
- `TIMESTAMP_TEXT`。`started_at` 与 `finished_at` 必须是带时区的 datetime，统一为 UTC，文本为 `datetime.isoformat(timespec="microseconds")` 且以 `+00:00` 结尾，保留六位微秒。拒绝 naive datetime。`finished_at` 不得早于 `started_at`。`elapsed_seconds` 等于这两个 UTC 时间相减后的 `total_seconds()`，且为有限数。
- `LIBRARY_VERSIONS`。只记录 `tensorflow_version`。不新增未冻结的通用依赖清单。
- `SHIFT_VARIANT`。`raw_shift` 与 `ema_shift` 都保留。IO 不删除其中任何一个。下游分析自行选择 RAW 或 EMA。
- `LOG_NAME_REVIN0`。无 RevIN 的 `P1`、`P2`、`P3` 日志名为 `<arm>__<protocol_tag>__<dataset>_h<horizon>_seed<seed>__revin0.log`。只有这三支无 RevIN 臂使用该后缀。`F1`、`F3` 不追加。Stage A cache key 与 `run_directory` 的协议身份不改变。
- `SKIP_ARTIFACT_HASH`。候选不足 `K` 时只写 `selection.json` 和 `DONE.json`。`DONE.json` 的 `artifact_sha256` 只能包含 `selection.json`。不创建 `predictions.parquet`，不创建 `cell_record.json`，不写空文件哈希，不写虚假预测路径或预测哈希，不自动重试。
- `PHASE_AUDIT_SCHEMA`。生产路径是 `消融实验/audit/phase_<phase>_audit.json`。四个分类固定为 `missing`、`hash_mismatch`、`failed`、`insufficient_candidates`。cell ID 固定为 `<arm>:<protocol_tag>:<dataset>:h<horizon>:seed<seed>`，例如 `F4:random3:D1:h3:seed42`。代码校验 phase、文件名和 cell ID。输出按确定顺序排列，不得重复，也不得跨类别混淆。未列入的 cell 保持缺行。
- `PREDICTION_PATH`。已完成 cell 的 `cell_record.json` 含 `prediction_path: "predictions.parquet"`。该值是相对于当前 cell `run_directory` 的相对路径。禁止绝对路径、机器路径或临时路径。预测文件的 SHA256 继续写在 `artifact_sha256`。
- `FAILED_BODY`。只约束目标 cell 的 `runs/.../FAILED.json`，不约束 Stage A cache。目标 cell 正文只能是 `{"attempt_count": 1}` 或 `{"attempt_count": 2}`。异常详情进入对应日志，不得写入 `FAILED.json`。带额外字段的 cell `FAILED.json` 必须拒绝。Stage A cache 的 `FAILED.json` 若含 `stage_a_key_sha256` 和 `error`，保持其独立协议。

OPEN-1、OPEN-2 与 GAP-TCN-SOURCE-FIT 已关闭。正式 runner 不再因这三项拒绝启动。工作区不干净时仍然拒绝。
