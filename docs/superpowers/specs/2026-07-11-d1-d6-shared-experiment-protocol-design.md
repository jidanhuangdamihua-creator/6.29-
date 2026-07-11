# D1–D6 共享实验协议与防泄漏改造设计

日期：2026-07-11

状态：已完成讨论，待用户书面审阅

适用范围：D1–D6 的数据预检、KNN 选源、CNN/基线样本生成、评估、审计与结果归档

## 1. 决策摘要

本项目采用“共享协议层 + 两条实验轨道”的结构：

- D1–D3 为严格论文复现轨道，只允许论文定义的数据范围、目标对象和候选源池。
- D4–D6 为扩展实验轨道，沿用统一的无泄漏选源、训练、评估和审计协议，但不得标记为论文原始复现。
- 所有数据集的 KNN 表征统一为截止日前连续 30 个日历日的逐日展平销量序列，不再使用统计签名作为正式选源特征。
- KNN、CNN 和基线共享同一截止日语义、样本清单、滚动预测原点和审计信息。
- 当前历史结果均视为 `legacy_unverified`；只有通过新协议预检并完成 5 个种子的正式重跑后，结果才可作为确认后的 baseline。

本次实现阶段只改造协议、数据生成脚本、运行入口、输出和轻量测试，不运行 D1–D6 正式实验，也不代替用户重新生成 D1–D6 数据。

## 2. 目标与非目标

### 2.1 目标

1. 消除 KNN 读取目标测试期或源未来数据的风险。
2. 固定 D1–D6 的目标、候选池、日期、特征、排序、并列处理和摘要算法。
3. 保证 KNN 选出的源就是后续 CNN 实际读取的源，并可逐元素追溯到原始切片。
4. 保证 CNN 与全部基线在相同有效样本上进行 horizon 1–5 的滚动预测评估。
5. 让缺数据、候选池错误、K 不足、重复键、目标混入源池等问题在训练前明确失败。
6. 保留现有主要运行脚本入口和主要结果字段，同时为新协议增加可审计字段。

### 2.2 非目标

- 不在本轮改造中运行 D1–D6 正式训练或批量数据管线。
- 不把 D4–D6 宣称为论文原始实验。
- 不继续使用旧结果计算新的汇总 baseline。
- 不通过缩小 K、填补候选源、退回统计签名或放宽日期要求来绕过预检失败。

## 3. 总体架构

新增共享协议层，现有 D1–D6 runner 只负责数据集适配和调用，不再分别定义截止日、候选池、KNN 特征或评估样本。

建议模块边界如下：

- `src/protocols/experiment_protocol.py`：协议版本、轨道、截止日、窗口、种子、指标空间和数据集规则。
- `src/protocols/candidate_pool.py`：候选池构造、合法性检查、KNN 表征、确定性排序和生产摘要函数。
- `src/protocols/rolling_origin.py`：horizon 1–5 的样本清单及 CNN/基线共用切分。
- `src/protocols/provenance.py`：源切片、CNN 输入、样本清单及结果审计记录。

共享层输出不可变的协议对象、候选池清单、选源结果和样本清单。训练代码只能消费这些对象，不得自行重新筛选源、重新推断日期或重新生成测试样本。

协议采用显式版本号；首个严格版本记为 `d1_d6_protocol_v1`。任何会改变候选池、窗口、特征、排序或样本身份的修改必须提升协议版本。

## 4. 双轨实验定义

| 数据集 | 轨道 | 目标 | 无信息共享候选池 | 有信息共享候选池 |
|---|---|---|---|---|
| D1 | strict_paper | Store1 × Item10 | Store1 × Item1–9 | Store1–3 × Item1–9 |
| D2 | strict_paper | Brand1 × Item10 | Brand1 × Item1–9 | Brand1–3 × Item1–9 |
| D3 | strict_paper | Store10 | Store1–9 | Store1–30，排除 Store10 |
| D4 | extended | 当前目标 store × item | 同一 store、同一 category 的其他 item | 允许其他 store、仍限同一 category 的其他 item |
| D5 | extended | 当前目标 store × item | 同一 store、同一 family 的其他 item | 允许其他 store、仍限同一 family 的其他 item |
| D6 | extended | 当前目标 store × item | 同一 store、同一 department 的其他 item | 允许其他 store、仍限同一 department 的其他 item |

通用约束：

- 目标自身始终从候选池排除。
- D4–D6 的“其他 item”以规范化后的完整 source key 判断，不可仅按显示名称判断。
- 每个候选必须有唯一、完整的规范化 source key；重复键直接失败。
- D1/D2 当前源文件不能覆盖上述论文候选池，因此数据再生成脚本必须修正；在用户重新生成前，严格预检应失败。

## 5. 唯一截止日与时间边界

所有组件只使用下列正式字段，不得从 `train_end`、旧版 `target_train_window.end` 或文件最大日期反推截止日：

```text
knn_observed_end = target_observed_start + 29 calendar days
source_observation_cutoff = knn_observed_end
target_test_start > knn_observed_end
```

约束如下：

- KNN 目标序列只能读取 `[target_observed_start, knn_observed_end]`。
- KNN 源序列只能读取同一组 30 个日历日，且日期不得晚于 `source_observation_cutoff`。
- CNN 的源训练切片及所有拟合变换不得读取 `source_observation_cutoff` 之后的数据。
- 目标测试标签严格晚于 `knn_observed_end`；每个预测原点的输入必须早于对应标签。
- 审计 CSV、摘要输入、runner 和测试均直接读取这些正式字段。

协议以连续日历日为单位。候选源缺少任一日期时，该候选不合法并从候选池有效集合排除；若因此不足 K，则预检失败。目标缺少任一日期时直接失败，除非数据集预处理协议明确声明“缺失日期代表零需求”，在原始数据生成阶段完成日历化并记录该规则；运行阶段不得临时补零。

## 6. KNN 选源协议

### 6.1 表征与缩放

每个目标和候选源均表示为按日期升序排列的 30 维逐日销量向量：

```text
[sales(day_1), sales(day_2), ..., sales(day_30)]
```

ID、store、item、category、family、department 和日期编码不得进入距离特征。统计签名可作为诊断输出，但不得参与正式 KNN 排名。

每个 `dataset × scenario × target` 任务使用一套统一缩放器。缩放器只能在该任务合法截止日前的目标 30 日向量和全部合法候选源 30 日向量上拟合；不得读取目标测试期、源未来期或其他任务数据。缩放参数写入审计记录或以摘要引用。

### 6.2 距离、排序和并列

- 所有 KNN 距离以 `float64` 计算。
- 主排序键为距离升序。
- `tie_tolerance` 固定为绝对误差 `1e-12`。
- tie group 按原始距离从小到大构造：每组以尚未分组的最小距离为锚点，所有与该锚点差值不超过 `1e-12` 的候选进入该组；组内按规范化 source key 的字典序稳定排序。后续候选不得通过链式相邻差值并入前组。
- 输出必须记录原始距离、排名、tie group、最终权重和规范化 source key。
- Top-K“不变”指有序 source key 列表、对应距离、权重以及 `selection_result_digest` 全部一致，而不仅是集合一致。

正式 baseline 的权重模式固定为 `inverse_distance`：

```text
score_i = 1 / (distance_i + 1e-8)
weight_i = score_i / sum(score_j),  j in ordered Top-K
```

距离和权重均以 `float64` 计算；非有限值直接失败。`raw_distance` 只允许作为独立标记的消融实验，不能混入正式 baseline 汇总。权重模式、稳定项和 K 均写入 selection digest 输入。

规范化 source key 使用固定字段顺序和类型；字符串去除首尾空白但不进行模糊匹配，数字 ID 使用十进制规范形式。每个数据集的字段顺序由协议固定，并写入摘要输入。

### 6.3 失败策略

以下情况必须在模型训练前失败，并给出数据集、场景、目标和具体原因：

- 目标 30 日窗口不完整或日期重复。
- 候选池为空、有效候选少于 K，或预期论文候选缺失。
- 候选 key 重复、字段缺失或目标自身进入候选池。
- 候选源 30 日窗口不完整、日期重复或含截止日之后的数据。
- KNN 特征列、日期顺序、缩放器拟合范围与协议不符。
- 运行期试图启用旧统计签名 fallback 或自动缩小 K。

## 7. 生产摘要机制

候选池审计、runner 输出和测试必须共同调用唯一的生产函数：

```text
build_candidate_pool_digest(
    protocol_version,
    dataset_id,
    scenario,
    target_key,
    group_cols,
    candidate_keys,
    observed_start,
    observed_end,
    feature_cols,
)
```

摘要规范：

- 使用 UTF-8 编码的规范 JSON，键名排序，紧凑分隔符，日期使用 ISO-8601。
- `candidate_keys` 在摘要前按规范化 key 字典序排序，使摘要表达候选池身份而非 KNN 排名。
- `group_cols` 和 `feature_cols` 保留协议声明顺序，因为顺序属于协议语义。
- 最终摘要为小写十六进制 SHA-256。
- 审计输出同时保存摘要值和完整规范化摘要输入，便于离线重算。
- with-sharing 与 without-sharing 分别生成摘要，不得复用。

另设 `selection_result_digest` 表达有序 Top-K 结果。其输入包含协议版本、candidate pool digest、K、权重模式、稳定项、有序排名、规范化 source key、原始 `float64` 距离、权重和 tie group；浮点数按确定的 17 位有效数字格式序列化后计算 SHA-256。

## 8. KNN 到 CNN 的强制溯源

CNN 不得根据商品名称或候选池重新查找源。它只能消费 KNN 返回的有序 source key 及下列切片标识：

```text
(store_or_brand, item_or_group, date_start, date_end)
```

对 D3 或其他不存在 item 维度的数据集，协议使用其明确声明的 key 字段，不伪造 item。

每个入选源必须保留：

- KNN 使用的原始 30 日切片及展平向量。
- CNN extractor 使用的完整源训练切片、日期范围和 source key。
- CNN 输入张量对应的日期顺序、特征顺序和标签日期。
- 从原始行到 KNN 向量、CNN 张量及标签的可验证映射。

正式运行前先验证选源 source key 与 CNN extractor key 完全一致。任何缺失、重复、重排或隐式替换均失败。

## 9. CNN 源训练与时间切分

KNN 的 30 日观察窗口只用于选源，不等于 CNN 的完整源训练窗口。CNN 可使用各数据集协议定义的合法源历史，但必须满足：

- 所有源特征和标签日期不晚于 `source_observation_cutoff`。
- 源样本按时间切分训练、验证、测试，不允许随机打乱后跨时间切分。
- 所有 scaler、特征选择器和其他拟合变换只在对应训练段拟合，再应用于验证和测试段。
- 源模型及迁移步骤只读取 KNN 选出的源；源权重与选源输出一致。
- 若旧实现所需最短历史无法在截止日前满足，任务失败，不得向未来扩展窗口。

## 10. 滚动预测与公平评估

### 10.1 样本协议

统一使用 rolling-origin 评估，预测 horizon 为 1、2、3、4、5。共享样本生成器先产生不可变 `sample_manifest`，CNN 和所有基线只消费该清单。

每条样本至少包含：

- dataset、track、scenario、target key。
- forecast origin、input date range。
- horizon、label date、原始尺度标签。
- sample key 和 sample manifest digest。

同一目标和 horizon 下，各方法必须使用完全相同的有效 sample key 有序列表。任何方法因特征缺失需要删除样本时，不能独自删除；应由协议层确定共同有效交集并重新生成统一 manifest，或使任务失败并报告原因。

### 10.2 基线特征可用性

基线只能读取预测原点当时可得的特征。允许项由数据集级 allowlist 明确列出。以下信息默认禁止：

- 预测标签日或其后的实际销量、交易、库存、客户数及聚合值。
- 使用完整数据拟合的统计量、编码器、缩放器或特征选择结果。
- 由未来记录回填的促销、价格或状态字段。
- 任何不能证明在预测原点已知的字段。

日历特征或明确提前发布的计划变量只有在 allowlist 中声明其可用时点后才可使用。

## 11. 指标、种子与汇总

正式 baseline 使用种子 `42, 43, 44, 45, 46`。统一设置 Python、NumPy、TensorFlow 和 PyTorch 的随机种子；可用时启用确定性计算，并记录无法完全确定的算子。

原始销量尺度为唯一主要指标空间：

- RMSE
- MAE
- sMAPE
- `Accuracy = 1 / (RMSE + 1e-8)`

归一化尺度指标只能作为诊断字段，不得替代主要结果或与原始尺度结果混合汇总。

最小结果粒度为：

```text
dataset × track × scenario × target × method × horizon × seed
```

汇总同时报告：

- 每个 horizon 的 5 种子 mean 和 std。
- horizon 1–5 的总体平均，并保留 5 种子 mean 和 std。
- 有效样本数和 sample manifest digest。

单种子结果只能标记为试运行，不得作为最终 baseline。

## 12. 输出、兼容与归档

保留现有 runner 入口和能够继续解释的主要结果字段。新增下列强制字段：

```text
protocol_track
protocol_version
knn_observed_start
knn_observed_end
knn_representation
target_test_excluded
source_future_excluded
candidate_pool_digest
selection_result_digest
horizon
seed
primary_metric_space
sample_manifest_digest
```

候选池审计还必须保存规范化摘要输入、候选总数、有效候选数、排除原因、K、ordered Top-K、距离、权重和 tie group。

旧结果缺少新协议字段时统一标记为 `legacy_unverified`：

- 可保留用于历史追踪。
- 不得与新协议结果拼接汇总。
- 不得用于论文表格、准确率提升对比或“已确认 baseline”的声明。

当前文件可先作为“改造前快照”存档，但不能以已确认 baseline 的名义封存。确认 baseline 的前提是：完成实现、通过严格预检和测试、用户重新生成所需数据，并按新协议完成 D1–D6 五种子重跑。

## 13. 严格测试契约

测试优先使用 Python 标准库 `unittest`，避免把当前缺失的 pytest 作为核心协议验证前提；已有 pytest 测试同步更新，但完整运行依赖其环境。

### 13.1 截止日不变性

构造含截止日前后数据的合成目标与候选源。只扰动目标测试期或源截止日之后的数据，要求以下内容逐项完全相同：

- 有序 Top-K source key。
- 距离、权重和 tie group。
- `candidate_pool_digest`。
- `selection_result_digest`。

测试直接读取 `knn_observed_end` 和 `source_observation_cutoff`，不得另行推断“截止日”。

### 13.2 观察期敏感性

合成数据必须预先建立明确距离 margin。将一个原本较远候选在观察期内的销量修改为确定的极端构造，使其成为预期 Top-1；测试先断言原始 margin 超过规定阈值，再断言 Top-1 精确翻转到该候选。不得依赖微小随机噪声。

### 13.3 排名与并列

- 非并列数据验证严格距离排序。
- 距离差不超过 `1e-12` 的候选验证同一 tie group 和规范化 source key 字典序。
- 重复运行验证有序 Top-K、距离、权重及 selection digest 完全一致。

### 13.4 摘要金标

测试直接调用生产 `build_candidate_pool_digest`，使用固定输入和固定 SHA-256 金标：

- with-sharing 与 without-sharing 各有独立金标。
- target key、候选 key、日期、特征列、场景或协议版本任一变化均必须改变摘要。
- 审计文件保存的规范化输入可由同一生产函数重算得到相同摘要。
- 测试不得复制一套独立摘要规则。

### 13.5 KNN–CNN 逐元素溯源

不运行真实训练，仅用小型合成数据验证：

1. 入选源的 `(store/item/date_start/date_end)` 或数据集对应 key 与原始切片完全一致。
2. KNN 展平向量与原始 30 日切片按日期逐元素一致。
3. CNN extractor 使用的 key 与 KNN 有序选源 key 完全一致。
4. CNN 输入张量、特征顺序、日期顺序和标签逐元素对应原始数据。
5. 任何 source key 替换、日期重排或数值修改均触发失败。

### 13.6 候选池与失败行为

- D1–D3 的候选 key 集合必须与论文规则精确一致。
- D4–D6 的分组和跨 store 规则精确一致，目标自身排除。
- 候选不足 K、缺日、重复日、重复 key、目标混入、未来日期、旧 fallback 均验证为明确失败。
- with/without 候选池在协议预期不同却得到相同摘要时，预检报告异常；若数据本身不足以形成跨 store 池，则任务失败。

### 13.7 公平评估

- CNN 与每个基线的 sample key、horizon、label date 和样本数完全一致。
- 特征 allowlist 拒绝未来实际量和未声明可用时点的字段。
- 指标从同一原始尺度预测与标签计算，并验证 horizon 级和总体聚合。
- 单种子输出不得通过 final-baseline 校验。

## 14. 实施与验证边界

实现顺序为：

1. 先增加会失败的协议单元测试与固定合成数据。
2. 实现共享协议、候选池摘要、KNN 排序和溯源对象。
3. 接入 D1–D6 数据适配器和现有 runner。
4. 接入统一 rolling-origin 样本与指标输出。
5. 修正 D1/D2 及必要的 D4–D6 数据再生成脚本，但不执行正式再生成。
6. 更新旧测试和文档，执行静态检查与三分钟内的轻量协议测试。

任何 Python 数据管线、模型训练、D1–D6 实验或可能写入大量输出的命令，均须通过仓库规定的 180 秒保护包装器运行。若超时返回 124，立即停止，不拆分、不降级、不重试，并把原命令交给用户手动执行。

## 15. 验收标准

设计对应的实现只有在以下条件全部满足后才可交付用户进行正式重跑：

- 共享协议成为 D1–D6 唯一的截止日、候选池、KNN 特征和样本来源。
- 严格测试契约全部通过，且测试未运行正式模型训练。
- D1/D2 缺失候选池在旧数据上能够预期地失败，修正后的生成脚本可产生协议要求的数据范围。
- 任何未来数据扰动均不影响选源，观察期确定性扰动能按预期改变排名。
- KNN 入选源到 CNN 张量和标签可逐元素追溯。
- CNN 与基线共享同一 sample manifest。
- 新输出包含全部协议、摘要、horizon、seed 和指标空间字段。
- 旧结果明确标记 `legacy_unverified`，不能进入新 baseline 汇总。
- 未由 Codex 运行 D1–D6 正式重跑。

用户完成数据再生成和五种子正式重跑后，只有通过结果完整性校验的产物才能归档为“已确认 baseline”，并作为后续准确率提升工作的对照。
