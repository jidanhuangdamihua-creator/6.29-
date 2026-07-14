# D1–D6 全量实验封存设计

日期：2026-07-14

状态：设计已由用户逐段批准，等待书面规格复核

范围：D1–D6 正式实验的数据协议、KNN 选源、迁移训练、盲测预测、指标、输出、D5 调度、缓存和封存验收

## 1. 目标

本设计把 D1–D6 收敛为一个可审计的正式实验协议，解决当前实现中会改变实验结论的差异：

- D1/D2 正式输入依赖服务器派生目录，本地缺少同一权威数据；
- D2 当前观察窗口与论文不一致，且原始日历存在缺日；
- D1–D6 的清洗、日期、KNN 和 source 训练窗口没有完全统一；
- 当前滚动原点评估会把 target 盲测期真实销售写入后续输入；
- 预测负值策略虽然进入结果字段，但没有在递归反馈和正式指标前真正执行；
- D5 没有按 heavy lane 优先运行，重复解析大数据并可能与另一个 D5 mode 重叠；
- 现有结果字段很多，但缺少足以独立重建一次正式运行的统一封存包。

正式协议的核心定义为：

> target 仅观察 30 个自然日；KNN 只用这 30 天与 source 的同日期数据选源；选中的 source 使用截止 target 观察期结束日的 180 个自然日进行预训练；随后 target 进行 180 个自然日的完全盲测，期间不读取真实销售、不重新训练、不微调。

## 2. 已比较的方案和决定

### 2.1 数据权威

比较过三种方案：继续依赖服务器派生目录、运行时临时修补、从原始数据生成项目内固定数据。采用第三种方案。

D1/D2 的正式 parquet 放在 `数据集/固化数据`，由单一、可重复的构建器从原始数据生成。正式运行不再依赖服务器上独有的 `数据集/派生数据/d1d2_protocol_v1`。构建结果必须携带原始输入哈希、构建代码/协议版本、字段规则、日期规则、重复聚合规则、缺日清单和输出哈希。

### 2.2 D5 缓存边界

比较过三种方案：每个 cell 重读原始数据、长期驻留一个包含 TensorFlow 的 mode 进程、mode 级不可变缓存加隔离 cell 进程。采用第三种方案。

mode owner 只构建一次数据、D5 authority、KNN 元数据和 prepared pool；cell 进程读取经过哈希验证的不可变缓存。这样保留 cell 失败隔离和可恢复性，同时利用操作系统页缓存和内存映射避免重复解析。不得让 TensorFlow 模型状态、随机数或逐 cell 内存长期累积在同一个解释器中。

### 2.3 盲测 horizon 执行

比较过三种方案：各 horizon 独立跳步递归、仅保留 horizon 1、由 horizon 1 推进共同盲测路径并同时评估 horizon 1–5。采用第三种方案。

同一个 `dataset × scenario × target × method × seed` 运行单元训练 horizon 1–5。每个预测原点生成未来 1–5 天预测，只把裁剪后的 horizon-1 预测写回历史并推进一天。horizon 2–5 读取的历史也只能由最初 30 天真实观察和此前 horizon-1 预测构成。

## 3. 日期和数据权威

### 3.1 target 通用协议

所有正式 target 使用连续 210 个自然日：

- 第 1–15 天：target train；
- 第 16–30 天：target validation；
- 第 31–210 天：180 天 blind test。

自然日而不是“现有记录行数”是执行权威。论文中的实际记录步数作为参考元数据保留，不改变统一执行边界。

### 3.2 D1

D1 使用论文 Table 2 的 target 启动日期：

- 观察期：2017-06-01 至 2017-06-30；
- 盲测期：2017-07-01 至 2017-12-27；
- 论文参考测试期：2017-07-01 至 2017-12-31。

正式 target 为 Store 1 / Item 10。正式固化 parquet 必须由原始 D1 数据生成并固定在项目内。

### 3.3 D2

论文没有直接打印 D2 的日期边界，但原始数据与论文 Table 5 形成了唯一数值指纹：使用 Brand 1 / Item 10 在 2018-06-01 至 2018-06-30 的销售和促销，与 Brand 1 的 Item 1–9 计算原始欧氏距离，能精确得到 Item 4、6、8 及 24.98、26.85、26.85。当前代码的 2018-06-05 起始日期不能复现该结果。

因此 D2 正式协议为：

- target：Brand 1 / Item 10；
- 观察期：2018-06-01 至 2018-06-30；
- 2018-06-02 缺失，补 `sales=0`、`promotion=0`，并标记 `is_synthetic_date=true`；
- 盲测期：2018-07-01 至 2018-12-27；
- 盲测期内 5 个原始缺日按同一规则补齐，并在 manifest 中列出；
- 论文参考测试期仍记录为 2018-07-01 至 2018-12-31，共 179 个实际记录步。

补 2018-06-02 不改变论文 KNN 指纹，因为所有比较实体在该日得到相同零向量。

### 3.4 D3

D3 正式 210 日窗口保持为 2015-01-03 至 2015-07-31：

- 观察期：2015-01-03 至 2015-02-01；
- 盲测期：2015-02-02 至 2015-07-31。

论文的 target `16/15/181` 记录步数仅作为参考字段；正式执行仍使用统一 `15/15/180` 自然日协议。

### 3.5 D4–D6

D4–D6 使用已固化 target authority，但要求恰好 210 个自然日：

- D4：2024-12-16 至 2025-07-13；
- D5：2017-01-17 至 2017-08-14；当前 2017-08-15 被排除，使 211 日窗口收敛为 210 日；
- D6：2015-10-26 至 2016-05-22。

任何固化 artifact、运行时常量或结果 manifest 与这些边界不一致时，正式运行必须 fail closed。

## 4. D1–D6 统一清洗契约

各数据集 adapter 只负责原始字段映射和数据集特有的字段聚合；共享管线固定执行以下顺序：

1. 标准化实体主键、日期、销售和特征列名；
2. 日期归一到自然日，拒绝无法解析的日期；
3. 按 `entity key + date` 聚合重复记录，聚合规则写入 manifest；
4. 按正式协议补齐连续自然日并记录合成日；
5. 按字段级策略填充缺失值，同时记录原始缺失数、填充值和原因；
6. 物理切分 source 180 日、target 30 日和 target blind 180 日；
7. 只在各自合法训练区间拟合 scaler、RFE 或其他数据统计量；
8. 将 blind truth 放入 evaluator-owned 对象，模型输入对象不得携带未来真实销售。

共享清洗报告至少记录：输入/输出行数、实体数、日期范围、重复数、缺日数、合成日、非有限值、负实际销量、每列填充数和最终哈希。实际销量的负值不得静默裁剪；若原始语义没有明确允许负销量，预检应拒绝正式运行并报告原始行。

模型特征可以因数据集而异，但每列必须归入一个或多个明确角色：

- `knn_observed`：只在 30 日观察窗用于相似度；
- `model_historical`：source 180 日和 target 30 日中的历史模型特征；
- `future_known`：预测时已确定的日历或预先公布计划特征；
- `evaluation_only`：盲测真实销售及其他事后才能知道的字段。

读取 parquet 时必须读取模型需要的销售、促销、日期及数据集专属特征，也读取审计和评估所需列；“列下推”不等于只读取 sales。读取后立即按上述角色隔离。

## 5. KNN 选源契约

### 5.1 时间边界

KNN 只能看到 target 的 30 个观察自然日和每个 source 在完全相同日期上的 30 天数据：

```text
knn_start = target_observed_start
knn_end   = target_observed_end
source_future_cutoff = target_observed_end
```

任何 source 的 target 盲测日期销售不得进入候选构造、距离、缩放、RFE、权重或 tie-break。

### 5.2 特征和距离

论文明确使用每个数据集所有可用特征和欧氏距离。因此正式 KNN 不是 sales-only：

- D1：观察期销售和日期特征；同日期特征在实体之间差为零；
- D2：至少为原始销售和促销；该定义必须继续通过论文距离指纹；
- D3：销售、日期、customer、open、promotion、holiday 等观察期可用特征；
- D4–D6：使用各数据集已声明且可在观察期合法获得的同类特征。

实体 ID、编码顺序、target/source 标签和任何事后字段不得作为距离特征。正式相似度使用固定列序和原始观测值的 30 日展平向量，不使用当前 D4–D6 的 `mean/std/min/max/last` 摘要替代。模型训练的 MinMax 缩放与 KNN 距离协议分开，不能反向改变 D2 论文指纹。

选择 top 3 source，记录完整候选池、排除原因、距离、稳定排序、tie-break、权重、候选池哈希和选中结果哈希。without information sharing 仅允许同组候选；with information sharing 使用协议定义的扩展组，但 target 自身始终排除。

## 6. source 180 日预训练和 target 迁移

KNN 30 日窗口与模型 source 训练窗口是两个独立数据视图：

```text
source_history_end   = target_observed_end
source_history_start = target_observed_end - 179 calendar days
source_pretrain_days = 180
knn_days             = 30
```

source 180 日的最后 30 天与 KNN 对齐，前 150 天只服务于 source 预训练。选中 source 后，模型使用该 source 的完整 180 日多特征数据；不得只用 KNN 30 日，也不得继续使用当前 D4–D6 的 300 日常量。若模型需要 source validation/early stopping，其训练和验证样本都必须从这 180 日池内按时间顺序产生，不得扩展日期边界。结果应分别记录 180 日 pool 的自然日数和窗口化后的实际 train/validation 样本数。

target 只使用 15 日 train 和 15 日 validation 进行迁移训练。source/target scaler、RFE、特征筛选和模型选择不得拟合 blind test。RFE 可使用 source 180 日与 target 已观察 30 日，但不能看到 target blind truth。

本统一 source 180 日规则是本实验的封存协议覆盖项：论文原始 D1–D3 source train 步数分别更长，不得把统一 180 日结果误标为论文原始 source split。论文步数保留在 reference metadata 中。

## 7. 30 日到 180 日的完全盲测

正式预测是一次启动的长期外推。target 训练在盲测开始前结束；之后 180 日内不重新训练、不重新拟合 scaler/RFE、不按真实结果选择模型，也不把真实销售写回输入。

对每个 `dataset × scenario × target × method × seed`：

1. 训练 horizon 1–5 模型；
2. 从第 30 个观察日建立唯一初始历史；
3. 在每个预测原点同时生成 horizon 1–5；
4. inverse transform 回原始销量空间；
5. 对每个预测执行 `max(prediction, 0)`；
6. 只把裁剪后的 horizon-1 预测写回递归历史；
7. 推进一个自然日，直到盲测第 180 天；
8. 预测全部完成后，evaluator 才按日期解封和连接真实值。

各 horizon 正式样本数为：h1=180、h2=179、h3=178、h4=177、h5=176。horizon 2–5 的历史输入同样只能含最初 30 日真实销售和随后已提交的 horizon-1 预测。

当前按单 horizon 隔离的 cell 会造成 horizon 2–5 为构造盲测路径而重复训练 h1。执行单元因此改为同一 seed 内统一运行五个 horizon，再输出五组结果。五个 horizon 和 seed 42–46 的正式矩阵保持不变。

## 8. 指标和预测裁剪

所有 D1–D6、方法、scenario、seed 和 horizon 只允许调用一个正式 sMAPE 实现：

```text
sMAPE = 100 * mean(
    2 * abs(y_pred_clipped - y_true)
    / (abs(y_true) + abs(y_pred_clipped) + 1e-8)
)
```

正式指标在原始销量空间、使用 inverse transform 后并已裁剪的预测计算。裁剪后的 h1 同时是递归反馈值，禁止使用负预测推进后再仅为输出裁剪。

`smape`、`smape_paper` 和 `original_scale_smape` 必须数值相同；normalized sMAPE 只能保留为明确标记的诊断列，不能参与正式排名、显著性检验或主结论。RMSE 和 MAE 使用相同日期上的原始真值与裁剪后预测。每个指标行必须携带样本数、日期首尾、日期/索引哈希、target 零值率、预测零值率、负值裁剪数量和指标协议 ID。

聚合顺序固定为：先对同一实验单元的逐日期预测计算 seed 级指标，再做 seed 平均，再做 target/dataset 层汇总。不得把不同 horizon 的逐日期样本直接混池后计算一个 sMAPE。

## 9. D5 heavy lane 和 16 CPU 调度

服务器调度固定为：

- 总计算线程预算不超过 16；
- 首个启动任务必须是 `d5_without`；
- D5 dedicated heavy lane 同一时刻只允许一个 D5 mode；
- D5 使用 6 个计算线程；普通 worker 使用 2 个计算线程；
- D5 运行时最多并行 5 个普通 worker，最重组合为 `6 + 5 × 2 = 16`；
- `d5_without` 成功完成后，`d5_with` 获得最高调度优先级并立即启动，不等待普通队列清空；
- `d5_without` 与 `d5_with` 永不重叠。

必须显式限制 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、TensorFlow intra-op 线程，并将 TensorFlow inter-op 保持在保守值，防止底层库超额并行。

失败语义：

- `d5_without` 失败：停止发放新的正式任务，不启动 `d5_with`；允许已在运行的原子 cell 完成并保存，最终状态为 `partial_failed`；
- `d5_without` 成功：先固化其结果，再启动 `d5_with`；
- `d5_with` 失败：完整保留已成功的 `d5_without`、已完成的 `d5_with` cell 和其他数据集结果，最终状态为 `partial_failed`；
- 后续失败不得删除、覆盖或降级已成功结果。

## 10. parquet 下推和缓存

parquet 通过 PyArrow dataset 或等价接口执行真实的列和日期 predicate pushdown。读取范围只覆盖：

- source：`source_history_start` 至 `target_observed_end`；
- target：正式 210 日窗口；
- 列：实体键、日期、销售、模型特征、KNN 特征、审计列和评估真值列。

不得整表载入后才做日期/列筛选。target blind truth 读取后立即进入 evaluator-owned sealed view，不能出现在模型 view。

缓存分两层：

- run-level immutable base cache：清洗后的日期/实体数据、D5 authority 和重建结果；
- mode-level cache：scenario 候选池、KNN JSON/元数据、30 日 KNN pool、180 日 source pool 和 target prepared pool。

缓存键至少包含原始/parquet SHA-256、协议版本、日期边界、字段清单、特征角色、缺日规则、scenario 和代码版本。写入使用临时路径后原子 rename；schema、哈希或 manifest 不匹配时必须拒绝缓存，不得静默重建后假装复用。

服务器旧 D5 正式结果因日期、source 180 日、盲测、裁剪或指标协议变化而无资格并入新结果。只有拥有完整 manifest 且所有输入哈希、协议、日期、列和缓存版本完全一致的不可变中间 artifact 才可复用；旧结果仅可作性能参考。

## 11. 正式输出包

每次 run 使用独立目录，禁止不同 run 直接追加到同一个正式 CSV。至少输出：

### 11.1 `experiment_results.csv`

每行唯一对应 `dataset × scenario × target × method × seed × horizon`。除指标外，至少包含：

- run/cell/protocol/schema 版本和状态；
- target/source/KNN 的实际起止日期、预期/实际自然日数；
- `knn_days=30`、`source_pretrain_days=180`；
- train/validation/test 样本数；
- feature role 清单和 feature hash；
- candidate pool、选中 source、距离、权重及 digest；
- scaler/RFE fit scope；
- blind/teacher-forcing/future cutoff 状态；
- seed、horizon、模型超参数和训练时间；
- raw/clipped prediction 诊断、指标空间、样本数和日期 digest；
- failure 类型、阶段和错误摘要。

### 11.2 `predictions/*.csv.gz`

逐日期预测轨迹，至少包含 run/cell 身份、dataset、scenario、target、method、seed、horizon、forecast origin、label date、`y_pred_raw`、`y_pred_clipped`、`was_clipped`、最终 evaluator 连接的 `y_true`、合成日期标志和 sample key。CSV 使用 gzip 只为降低 D5 存储；解压后仍是标准 CSV。

### 11.3 其他审计文件

- `source_selection.csv`：全候选、排除原因、距离、排名、权重和哈希；
- `failures.csv`：失败 cell、阶段、异常、已完成范围和重跑身份；
- `run_manifest.json`：输入哈希、Git revision、环境/依赖、日期、特征角色、线程预算、缓存哈希和调度时间线；
- `acceptance_report.json` 与 `acceptance_report.md`：机器和人工可读的封存判定；
- 原子 cell 结果及日志：供 partial run 审计和安全恢复。

## 12. 原子写入、恢复和封存状态

每个 cell 先写临时目录，完整通过 cell acceptance 后原子 rename 为成功目录。汇总器只读取带成功 manifest 的 cell，半写 CSV 不得进入汇总。

恢复时按完整 cell identity 和输入/协议 digest 判断是否可跳过；不能只凭文件名或“已有 CSV”判断。协议变化必须自动使旧 cell 失效。

状态定义：

- `running`：仍有正式 cell 未结束；
- `partial_failed`：至少一个预期 cell 失败或缺失，但成功数据已保存；
- `complete_unsealed`：cell 齐全，尚未通过全局验收；
- `sealed_success`：全局验收全部通过并生成 `SEALED_SUCCESS`；
- `sealed_failed`：验收明确失败，保留报告但不生成成功标志。

只有 expected manifest 中所有 dataset、scenario、target、method、seed、horizon 都恰好出现一次，且预测轨迹、日期、哈希、指标和状态一致时，才允许 `sealed_success`。

## 13. 验证和验收测试

实现必须通过不依赖全量训练的单元、合约和小型合成数据测试：

1. D2 日期为 2018-06-01 起始，缺日补齐，并精确复现 Item 4/6/8 距离指纹；
2. D1–D6 target 均为 30+180 自然日，D5 恰好 210 日；
3. source pool 恰好 180 自然日并截止 target observed end，KNN 恰好使用对齐的最后 30 日；
4. 修改 target blind truth 后，选源、训练数据、模型预测和递归输入 digest 不变；只有最终指标允许变化；
5. 修改 source 在 target blind 日期的销售后，KNN、source 预训练和预测不变；
6. horizon 2–5 输入不包含 blind truth，并获得 179/178/177/176 个样本；
7. 负预测在递归反馈和指标之前裁剪，raw 与 clipped 轨迹同时保留；
8. 所有方法共享 sMAPE 黄金样例，别名值一致且正式范围为 0–200%；
9. parquet reader 的测试 spy 证明 columns 和 date filters 被下推；
10. 相同 mode 的基础数据、D5 authority、KNN 和 prepared pool 只构建一次；协议/hash 变化会使缓存失效；
11. 调度 dry-run 证明先启动 `d5_without`、D5 不重叠、成功后立即启动 `d5_with`、失败分支保存结果且总线程不超过 16；
12. 删除任一结果行、预测日期或成功 manifest 后，全局验收拒绝封存；
13. CSV schema、主键唯一性、日期 digest、样本数和逐日轨迹能够相互重算并得到相同指标。

按照项目 `AGENTS.md`，任何 Python 验证、数据脚本或实验命令必须经 `tools/protection/codex_timeout.py` 的 180 秒保护运行。若保护器返回 124，Codex 立即停止，不拆分、不重试、不继续实验，并把原命令交给用户在 Terminal 手动执行。

## 14. 正式入口的完成标准

真正的全量入口必须完成以下单一工作流，而不是要求用户手工串联脚本：

1. preflight 输入、哈希、日期、feature role 和缓存；
2. 生成 expected experiment manifest；
3. 先发放 D5 heavy lane，再用普通 worker 填充资源；
4. 原子保存每个 cell 和逐日预测；
5. 按依赖关系启动 `d5_with`；
6. 聚合但不覆盖 partial result；
7. 运行全局 acceptance；
8. 只有全部通过才写 `SEALED_SUCCESS`。

入口的 dry-run 必须打印完整任务数、D5 顺序、每类线程数、预计输出主键和恢复命中情况，使服务器正式启动前可以人工核对。

## 15. 非目标

- 不把旧 D5 指标直接拼接进新正式实验；
- 不为了缩短运行时间改变模型方法、seed 42–46、horizon 1–5 或 source 数量；
- 不把 normalized sMAPE 用作正式结论；
- 不在盲测期做 teacher forcing、滚动重训或使用实际未来促销/营业状态，除非该列被协议和证据明确声明为预测时已知；
- 不执行与本次封存协议无关的模型结构重构。
