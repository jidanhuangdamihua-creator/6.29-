# 消融实验接口冻结

状态：`INTERFACE_FREEZE`。这是模块接口，不是阶段 0 验收，也不是实现。

本文件只约束以后写在 `消融实验/sita_ablation/` 和 `消融实验/analysis/` 里的新代码。数值以 `消融实验/configs/features_frozen.json`、`消融实验/configs/protocol_frozen.json`、`消融实验/configs/tcn_frozen.json` 和 `消融实验/DECISIONS.md` 为准。下面若复述了这些值，只是为了把签名和形状写完整。冻结文件与本文件不一致时，冻结文件优先，先改本文件，再写实现。

本文件没有新的学习率、轮数、ε、日期终点或损失阈值。`OPEN-1` 与 `OPEN-2` 的公式在 `DECISIONS.md`。`tcn_source_fit` 的损失、优化器、学习率和轮数以 `configs/tcn_frozen.json` 为准。

允许的类型来自标准库、`numpy`、`pandas` 和 `tensorflow`。不得 `import src`。模型张量是 `float32`。源选择的最小值、最大值、距离和权重在 `float64` 中计算，因为正式距离是 `np.linalg.norm`（`src/protocols/candidate_pool.py:1452`）。指标在 `float64` 中计算。

## 1. 依赖顺序

调用方向只允许向下。

1. `data` 读封存表并切出日期边界。
2. `windows` 在训练切片和验证切片内部做滑窗。测试标签覆盖全部 180 个测试日；输入可以借用测试期开始前的验证日观测。
3. `scaling` 拟合三套互不混用的缩放对象。
4. `source_selection` 只用源选择缩放和比较变量。
5. `models` 和 `revin` 只建网络，不读数据文件。
6. `stage_a` 训练已选源并缓存源权重。融合在目标初始化时按该臂的权重规则现算，不把一种融合结果放进共享缓存。
7. `arms` 只描述因子，不训练。
8. `adaptation` 用目标缩放后的窗口、配对初始值和 Stage A 融合参数做目标适配。
9. `metrics` 把预测反变换到原始销量后再计算。
10. `io` 原子写入记录、预测和 `DONE.json`。
11. `runner` 按上述顺序编排、跳过和续跑。
12. `analysis` 只读已经完成的 cell 记录，不训练、不改缓存。

`F1` 和 `F2` 不调用 `stage_a`。`P2` 不重新训练源模型，只读取 `K=3` 的 Top-1 权重。

## 2. 标识、cell 和运行目录

```python
DatasetId = Literal["D1", "D4", "D5", "D6"]
Horizon = Literal[1, 3, 5]
CellSeed = Literal[42, 43, 44, 45, 46]
Backbone = Literal["cnn", "tcn"]
SelectionRule = Literal["none", "top", "bottom", "random3"]
WeightMode = Literal["none", "inverse_distance", "uniform", "single"]
ProtocolTag = Literal["main", "alt_lr", "random3", "bottom3", "k5", "k10", "ell15"]
```

`ConditionId` 是统计分析单位，不是运行目录：

| 字段 | 取值 |
|---|---|
| `dataset` | `DatasetId` |
| `horizon` | `Horizon` |

十二个条件是四个数据集乘三个 `horizon`。一个条件的误差是五个种子的算术平均。种子不是独立样本。

`RunSpec` 是一次目标适配的完整身份：

| 字段 | 含义 |
|---|---|
| `arm` | `F1` `F2` `F3` `F4` `P1` `P2` `P3` `SITA_TCN` `TCN_REVIN` |
| `protocol_tag` | 上表七个标签之一 |
| `dataset` `horizon` `seed` | 条件与种子 |
| `backbone` | 主实验和 Tier 1 为 `cnn`；G5 为 `tcn` |
| `revin` | 见第 8 节 |
| `freeze_conv` | 只有 `P1` 为真 |
| `selection_rule` | 无源配置为 `none` |
| `requested_k` | 无源配置为 `null`；否则是该臂实际使用的源数 |
| `weight_mode` | 无源配置为 `none` |
| `comparison_days` | 无源配置为 `null`；否则是 `15` 或 `30` |
| `target_learning_rate` | 由数据集和 `protocol_tag` 决定，见下表 |

主协议学习率：D1、D4、D6 为 `3e-5`，D5 为 `3e-4`。`protocol_tag=alt_lr` 时对调：D1、D4、D6 为 `3e-4`，D5 为 `3e-5`。这是规范 §4.2 的 S1，不是新数值。源模型学习率始终是 `1e-4`，不随 `alt_lr` 改变。

规范 §9.1 的 `runs/<arm>/<dataset>_h<h>/seed<seed>/` 不能区分 S1 和主协议。接口因此把 `protocol_tag` 放进目录，避免续跑时互相覆盖：

```text
消融实验/runs/<arm>/<protocol_tag>/<dataset>_h<horizon>/seed<seed>/
```

`S3` 的 `K=1` 不另建目录，记录指向 `P2/main` 的同一路径。`S3` 的 `K=3`、`S2` 的 Top-3、`S4` 的 `ℓ=30` 都指向 `F4/main`。G5 使用臂名 `SITA_TCN` 和 `TCN_REVIN`，标签为 `main`。

`StageAKey` 不包含臂、RevIN、是否冻结、融合权重方式或目标学习率。它只标识源模型训练，不标识融合结果。融合结果若另行落盘，使用第 9 节的 `FusedInitKey`，其中必须有 `weight_mode`。`StageAKey` 的字段是：

| 字段 | 作用 |
|---|---|
| `dataset` `horizon` `seed` | 种子就是 Stage A 种子 |
| `backbone` | `cnn` 与 `tcn` 不得互读权重 |
| `selection_rule` | `top`、`bottom`、`random3` |
| `trained_k` | 实际训练的源数，只允许 `3`、`5`、`10` |
| `comparison_start` `comparison_end` | 闭区间日期，避免 `ℓ` 相同但日历不同 |
| `source_history_start` `source_history_end` | 冻结的源历史，不随 `ℓ=15` 缩短 |
| `model_features` | 该骨干的输入列 |
| `similarity_variables` | 冻结的比较列 |
| `scaling_rule` | 固定字符串 `per_similarity_variable_minmax` |
| `draw_rule` | `random3` 为 `sorted_permutation_v1`，其余为 `null` |
| `frozen_config_sha256` | 三份冻结 JSON 字节的 SHA256，按文件名排序后拼接 |

`trained_k` 不是 `1`。`P2` 没有自己的 Stage A 条目。

## 3. 三套 scaler

三套对象分开构造、分开保存、禁止互相赋值。

### 3.1 目标模型 scaler

```python
def fit_target_model_scaler(train_rows: DataFrame, columns: tuple[str, ...]) -> ModelScaler:
    ...
```

`ModelScaler` 是独立对象，字段为 `role="target_model"`、`columns`、`data_min`、`data_max`、`scale`、`min`。形状都是 `(C,)`，`C=len(columns)`。

- 拟合行：该数据集冻结的目标训练闭区间，且只使用目标序列。
- 拟合列：CNN 为 `features_frozen.json` 的 `model_features`。TCN 先对同一列做这个缩放，再只把 `sales` 切成通道 0。`sales` 在四个数据集的模型列里都是第一列。
- 变换行：训练、验证、测试行都可以变换，验证和测试不参与 `fit`。
- 全距为零的列沿用冻结的 sklearn 行为：`scale=1`，该列变换结果为 0。这不是源选择那条“整列置零”规则的同一个对象。
- 目标帧不另设填补值。模型列或比较列出现非有限值时，该 cell 失败。

反变换只作用于销量这一列：

```python
def inverse_sales(scaler: ModelScaler, values: ndarray) -> ndarray:
    # values: (N,) float32 或 float64，返回 (N,) float64
```

无 RevIN 的臂不做 RevIN 反变换，仍然用这个函数回到原始销量。规范 §4.1 的“不做输出恢复”指不做 RevIN 恢复；规范 §3.3 要求指标在原始销量尺度上。

### 3.2 源模型 scaler

```python
def fit_source_model_scaler(source_train_rows: DataFrame, columns: tuple[str, ...]) -> ModelScaler:
    ...
```

每个入选源各有一个对象，`role="source_model"`。它只拟合该源已裁到 `source_observation_cutoff`、并按冻结比例切出的训练日期行。列与该骨干的模型输入列相同，包含 `sales`。同一个对象变换该源的训练、验证和测试日期行。滑窗标签取变换后的 `sales`。

它不拟合目标行，也不拟合比较窗口里的其他源。目标适配的输入禁止经过任何一个源模型 scaler。正式路径里用源 scaler 变换目标帧，是为了迁移支路的源专家特征；本接口的预测是 `head_target(z_T)`，没有这条支路。

源序列在切分前，对数值模型列的缺失填 0；布尔列和分类列不填。全距为零时同样是 `scale=1`、变换结果为 0。

### 3.3 源选择 scaler

```python
def fit_selection_scaler(
    target_window: ndarray,
    candidate_windows: ndarray,
    variables: tuple[str, ...],
) -> SelectionScaler:
    ...
```

输入：

- `target_window`：`(V, L)` `float64`
- `candidate_windows`：`(N_legal, V, L)` `float64`
- `L` 是比较窗口的日历天数，`V` 是相似性变量数

`SelectionScaler` 的字段是 `role="source_selection"`、`variables`、`minimum (V,)`、`maximum (V,)`、`range (V,)`。拟合人口只有目标序列和全部合法候选在同一比较窗口上的观测。测试日期不得进入。模型 scaler 的参数不得进入。

对变量 `j`，`r_j = M_j - m_j`。`r_j > 0` 时 `g_j(x) = (x - m_j) / r_j`。`r_j = 0` 时，该变量在目标和每一个合法候选上的 `L` 个位置全部写成 0。这一列不参与距离区分，也不除零。

```python
def transform_selection(scaler: SelectionScaler, windows: ndarray) -> ndarray:
    # (V, L) -> (V*L,)；或 (N, V, L) -> (N, V*L)
    # 变量按 features_frozen.json 的声明顺序拼接
```

长度：`ℓ=30` 时 D1、D4、D6 为 30，D5 为 90。`ℓ=15` 时分别为 15 和 45。D5 的 `onpromotion` 只出现在比较向量里，不是模型输入。封存的 D6 `knn_schema.json` 把 `sell_price` 标成比较字段；本接口不采用，D6 比较变量只有 `sales`。

## 4. 每个 cell 的数据、日期和预测窗口

```python
def load_condition_frames(dataset: DatasetId) -> ConditionFrames:
    ...
```

`ConditionFrames` 在同一数据集的三个 `horizon` 之间共享，因为日期边界不依赖 `horizon`。字段：

| 字段 | 内容 |
|---|---|
| `dataset` | `DatasetId` |
| `source_rows` `target_rows` | 只含本数据集封存 parquet 的行 |
| `key_fields` | 序列主键 |
| `target_key` | 冻结目标 |
| `model_features` `similarity_variables` | 冻结列 |
| `source_history` | 闭区间 |
| `target_train` `target_val` `target_test` | 三个互不相交的闭区间 |
| `comparison_30` | 等于目标训练起点到目标验证终点 |
| `comparison_15` | 等于目标训练闭区间 |
| `data_sha256` | 该数据集 `source.parquet` 与 `target.parquet` 的 SHA256 |

路径解析各数据集的 `source.parquet` 与 `target.parquet`。不把 `prepared_180day_source.parquet` 当作选源或训练输入，不读固化 KNN 的 Top-K 和 D6 运行时统计量。D5 源历史重建额外读取 `prepared_180day_source_manifest.json` 冻结的 Favorita 权威 CSV（`oil.csv`、`transactions.csv`、`items.csv`、`stores.csv`、`holidays_events.csv`）；重建在消融目录独立实现，禁止 `import src`，禁止改写封存 parquet。

主键和目标：

| 数据集 | 主键 | 目标 | 源历史 / `source_observation_cutoff` | 训练 | 验证 | 测试 |
|---|---|---|---|---|---|---|
| D1 | `store_id, item_id` | `1/10` | 2017-01-02 至 2017-06-30 | 2017-06-01 至 2017-06-15 | 2017-06-16 至 2017-06-30 | 2017-07-01 至 2017-12-27 |
| D4 | `store_id, product_id` | `166/258` | 2024-07-19 至 2025-01-14 | 2024-12-16 至 2024-12-30 | 2024-12-31 至 2025-01-14 | 2025-01-15 至 2025-07-13 |
| D5 | `store_nbr, item_nbr` | `48/1159414` | 2016-08-20 至 2017-02-15 | 2017-01-17 至 2017-01-31 | 2017-02-01 至 2017-02-15 | 2017-02-16 至 2017-08-14 |
| D6 | `store_id, item_id` | `CA_1/FOODS_3_080` | 2015-05-29 至 2015-11-24 | 2015-10-26 至 2015-11-09 | 2015-11-10 至 2015-11-24 | 2015-11-25 至 2016-05-22 |

D5 **目标**在切分前只保留 `date <= 2017-08-14`。D5 **源候选**只使用 `date <= 2017-02-15`（`source_observation_cutoff`）。封存文件本身不改。切完后三个目标区间必须与上表逐日相等，否则停止。

D6 的“同一 department”用封存列 `dept_id`。依据是 `数据集/固化数据/d1_d6_sealed_v1/dataset6/source_schema.json` 的列名，以及正式组装使用的 `dept_id`（`src/transfer_methods/ms_katl_v3_frame_assembly.py:61`，`src/protocols/gate1_transformation.py:244`）。`experiment_protocol.py:316` 里的分组记号 `department` 不是 parquet 列名。

合法候选必须在拟合选择 scaler 和计算距离之前通过下列全部检查。缺任一门不得进入 scaler 总体，也不为了凑满 `K` 而收回。

1. 身份与分组：
   - D1：`store_id=1` 且 `item_id` 为 1 至 9。目标 `1/10` 不在候选中。
   - D4：`require_same_group` 为假，源池不做域过滤。`without` 只保留 `store_id=166`，并排除目标本身。
   - D5：同一 `family`，源池做域过滤，`without` 只保留 `store_nbr=48`，排除目标。
   - D6：同一 `dept_id`，`without` 只保留 `store_id=CA_1`，排除目标。
2. 源历史资格：冻结源历史闭区间上恰好 180 个不重复公历日（`exact_expected_date_set`）。D1/D4/D6 直接在裁窗后的源行上检查。D5 先按正式规则重建缺失自然日，再检查；观测行不变，合成行 `sales=0`、`onpromotion=0`，协变量按权威 CSV 填充（见 `protocol_frozen.json` 的 `d5_source_history_reconstruction` 与 `DECISIONS.md` STOP-D5-RECON）。
3. 比较窗口：当前 `ℓ` 对应的每一个日历日都存在，无重复日。
4. 比较变量：声明顺序上的每个相似性变量有限且可编码。

源模型行先裁到冻结的源历史终点（`source_observation_cutoff`），再按不重复日期做 `0.8/0.1/0.1`。切点是 `train_end=max(1, int(n*0.8))`，`val_end=min(n-1, max(train_end+1, int(n*0.9)))`。滑窗不跨过这三刀。审计里 D4 horizon 1 的 134 个样本来自另一套选源，不能写成消融的样本数。入选源在通过上述 180 日门之后，训练序列恰好是这 180 个日期。

```python
def build_partition_windows(
    rows: DataFrame,
    *,
    key_fields: tuple[str, ...],
    key: tuple[str, ...],
    feature_columns: tuple[str, ...],
    horizon: Horizon,
    window_size: int = 10,
) -> WindowBatch:
    ...

def build_test_windows(
    observed_rows: DataFrame,
    *,
    test_start: str,
    test_end: str,
    feature_columns: tuple[str, ...],
    horizon: Horizon,
    window_size: int = 10,
) -> WindowBatch:
    ...
```

`build_partition_windows` 只用于目标训练、目标验证和源模型的三个比例切片。输入窗和标签日期都要落在传入的那一个切片里。目标训练和验证都是 15 天，样本数是 `6-horizon`：horizon 1、3、5 分别为 5、3、1。源窗口仍不跨过 `0.8/0.1/0.1` 的三刀。

`build_test_windows` 负责规范 §3.3 的测试：180 天、逐日滚动、参数固定。每个冻结测试日都是一条标签，horizon 1、3、5 的评分样本数都是 180。标签日 `t` 的输入是结束于 `t-horizon` 的 10 天，形状仍是 `(180, 10, C)`。首条标签是测试起点，末条标签是测试终点。

测试期开始前的验证日观测可以作为测试输入。依据是冻结的标签定义：标签在输入窗终点之后 `horizon` 天（`消融实验/configs/features_frozen.json` 的 `lag_columns`）。在窗长 10、`horizon ≤ 5`、验证窗 15 天时，第一条测试样本的 10 个输入日全部落在验证窗内，不需要训练日，也不包含任何测试日。这些验证行只用已经拟合好的目标 scaler 做 `transform`。它们不重新拟合 scaler，不进入源选择，也不参与参数更新。原点滚进测试期之后，输入可以包含该原点当日及以前的测试观测；标签日和更晚的日期不能进入输入。参数在整个 180 天上保持固定。

四个数据集、三个 horizon 的首尾目标日和第一条输入如下。评分样本数都是 180。

| 数据集 | h | 首个测试目标日 | 最后一个测试目标日 | 样本数 | 第一条输入 |
|---|---:|---|---|---:|---|
| D1 | 1 | 2017-07-01 | 2017-12-27 | 180 | 2017-06-21 至 2017-06-30 |
| D1 | 3 | 2017-07-01 | 2017-12-27 | 180 | 2017-06-19 至 2017-06-28 |
| D1 | 5 | 2017-07-01 | 2017-12-27 | 180 | 2017-06-17 至 2017-06-26 |
| D4 | 1 | 2025-01-15 | 2025-07-13 | 180 | 2025-01-05 至 2025-01-14 |
| D4 | 3 | 2025-01-15 | 2025-07-13 | 180 | 2025-01-03 至 2025-01-12 |
| D4 | 5 | 2025-01-15 | 2025-07-13 | 180 | 2025-01-01 至 2025-01-10 |
| D5 | 1 | 2017-02-16 | 2017-08-14 | 180 | 2017-02-06 至 2017-02-15 |
| D5 | 3 | 2017-02-16 | 2017-08-14 | 180 | 2017-02-04 至 2017-02-13 |
| D5 | 5 | 2017-02-16 | 2017-08-14 | 180 | 2017-02-02 至 2017-02-11 |
| D6 | 1 | 2015-11-25 | 2016-05-22 | 180 | 2015-11-15 至 2015-11-24 |
| D6 | 3 | 2015-11-25 | 2016-05-22 | 180 | 2015-11-13 至 2015-11-22 |
| D6 | 5 | 2015-11-25 | 2016-05-22 | 180 | 2015-11-11 至 2015-11-20 |

闭区间天数已经核对为 180。标签日期集合必须与冻结测试区间逐日相等。缺任何一天就让该 cell 失败，不缩小样本，不写 `DONE.json`。

只在测试切片内部滑窗的做法不采用。那种构造要求输入和标签都在测试区间内，于是每个 horizon 开头有一段测试日没有样本。它和“180 天逐日测试”的差额就是这些未评分日期：horizon 1 缺前 10 天、样本数 170；horizon 3 缺前 12 天、样本数 168；horizon 5 缺前 14 天、样本数 166。正式调用 `build_tabular_sequence` 时若只传入测试切片，就会得到这个结果（`src/transfer_methods/ms_katl_mechanism_ablation.py:301-309`）。未评分日期是：

| 数据集 | h=1 未评分 | h=3 未评分 | h=5 未评分 |
|---|---|---|---|
| D1 | 2017-07-01 至 2017-07-10 | 2017-07-01 至 2017-07-12 | 2017-07-01 至 2017-07-14 |
| D4 | 2025-01-15 至 2025-01-24 | 2025-01-15 至 2025-01-26 | 2025-01-15 至 2025-01-28 |
| D5 | 2017-02-16 至 2017-02-25 | 2017-02-16 至 2017-02-27 | 2017-02-16 至 2017-03-01 |
| D6 | 2015-11-25 至 2015-12-04 | 2015-11-25 至 2015-12-06 | 2015-11-25 至 2015-12-08 |

`src/protocols/rolling_origin.py:132-159` 把第一个原点固定在验证终点。horizon 1 因此能覆盖全部测试日，horizon 3 和 5 仍会漏掉测试起点之后的前 `h-1` 天。本接口不用这道原点下限。测试记录里的 `unscored_test_dates` 必须是空列表。

`WindowBatch` 字段：

| 字段 | 形状 |
|---|---|
| `x` | `(N, 10, C)` `float32`，已经过对应模型 scaler |
| `y_scaled` | `(N,)` `float32`，缩放后的销量 |
| `y_original` | `(N,)` `float64`，反变换前保存的原始销量 |
| `input_start` `input_end` `label_date` | 长度 `N` 的日期 |
| `partition` | `train`、`validation` 或 `test` |

TCN 在缩放后另给 `x_sales`，形状 `(N, 10, 1)`，取 `sales` 通道。CNN 不使用这个字段。

断言：训练标签属于训练区间，验证标签属于验证区间，测试标签与冻结测试区间逐日相同。三个标签集合互不相交。测试输入可以早于测试起点，但不能晚于该样本的输入窗终点。源选择日期和两个模型 scaler 的拟合日期都不包含测试日期。

## 5. `source_selection`

```python
def rank_candidates(
    dataset: DatasetId,
    frames: ConditionFrames,
    *,
    comparison_days: Literal[15, 30],
    rule: Literal["top", "bottom", "random3"],
    k: int,
    seed: CellSeed,
) -> SelectionResult:
    ...
```

`rank_candidates` 先按第 4 节的四道门筛出合法候选，再在目标与这些合法候选的比较窗口上拟合选择 scaler，最后算距离。未通过身份、180 日源历史、比较窗口完整性或变量有效性的键不得进入 scaler，也不得进入距离向量。

距离是缩放后目标向量与每个合法候选向量的欧氏范数，形状 `(N_legal,)`。排序使用正式的锚定并列规则，`tie_tolerance=1e-12`（`src/protocols/experiment_protocol.py:205`，`src/protocols/candidate_pool.py:233-268`）。这不是新的阈值，冻结权重公式旁边的正式排序就是这个容差。

- `top`：升序排名的前 `K` 个。
- `bottom`：同一全序的最后 `K` 个。
- `random3`：先按主键字典序排列合法候选，再用第 7 节的 `random3` 生成器做一次无放回排列，取前 `K` 个。`K` 固定为 3。训练顺序等于这个抽取顺序。

`N_legal < k` 时返回 `status="SKIPPED_INSUFFICIENT_CANDIDATES"`，`selected=()`。不补齐，不训练。

`SelectionResult` 还包含 `distances (K,)`、`inverse_distance_weights (K,)`、`uniform_weights (K,)` 和 `selection_scaler`。逆距离权重是 `1/(distance+1e-8)` 后归一化，和为 1。均匀权重每个都是 `1/K`。`K=1` 时两种权重都是 `[1.0]`。臂在读取缓存时选择其中一种，选择本身不因为权重方式而重跑。

重合数是两个源键集合的交集大小，取值 0 到 3。计算它时调用同一函数和同一个 `random3` 流，但不因此训练额外源模型。

## 6. `models` 与 `revin`

CNN 与 TCN 都用 TensorFlow。TCN 结构不从 BL7 继承早停、patience 或 best_state。源拟合的 MSE 和 `1e-4` 写在 `tcn_source_fit`，不是目标侧学习率，也不是 BL7 的早停循环。

### 6.1 CNN

源模型和目标骨干的卷积部分相同，权重不是同一套。

| 层 | 输出 |
|---|---|
| 输入 | `(N, 10, C)` |
| `conv1` | Conv1D，32，核 3，`same`，ReLU；核 `(3, C, 32)`，偏置 `(32,)` |
| `pool1` | MaxPool 2，时间 10 变为 5 |
| `conv2` | 64，核 `(3, 32, 64)`，偏置 `(64,)`，时间仍为 5 |
| `pool2` | 时间 5 变为 2 |
| `conv3` | 128，核 `(3, 64, 128)`，偏置 `(128,)`，时间仍为 2 |

`C` 是 D1 的 5、D4 的 13、D5 的 11、D6 的 11。Flatten 后是 `(N, 256)`。

源模型在 Flatten 后接 Dropout `0.1` 和 `dense_out`，核 `(256, 1)`，偏置 `(1,)`。编译为 Adam 与 MSE。源模型的 Adam 不设置 `clipnorm`，因为正式常数是 `None`（`src/utils/experiment_hyperparams.py:7`）。

目标路径在 Flatten 处停止。其后是 `feature_dropout` `0.1` 和 `head_target`，核同样是 `(256, 1)`。没有 adapter、交叉注意力、迁移头、门控或域判别器。预测等于 `head_target(z_T)`，`z_T` 的形状是 `(N, 256)`。

```python
def build_source_cnn(input_shape: tuple[int, int]) -> tf.keras.Model: ...
def build_target_cnn(input_shape: tuple[int, int]) -> tuple[tf.keras.Model, tf.keras.Model]:
    # 返回 (全网络, 以 flatten 为输出的提取器)
```

融合张量只有 `conv1`、`conv2`、`conv3` 的 kernel 和 bias。位置对应加权平均，不做滤波器重排。`P1` 冻结的也只是这六个张量。

### 6.2 TCN

输入 `(N, 10, 1)`。两块、通道 16、核 3、膨胀 1 和 2。因果卷积先在时间轴左侧填充 `(3-1)*dilation`，再使用 `valid` 卷积。每块是两次因果卷积和 ReLU；输入通道不等于输出通道时才有 `1x1` 残差，所以只有第 0 块有可训练残差。头是最后时刻的 `Dense(1)`，核 `(16, 1)`，偏置 `(1,)`。

可融合张量与冻结名单一一对应：

| 冻结名 | Keras 形状 |
|---|---|
| `blocks.0.conv1` | 核 `(3, 1, 16)`，偏置 `(16,)` |
| `blocks.0.conv2` | 核 `(3, 16, 16)`，偏置 `(16,)` |
| `blocks.0.residual` | 核 `(1, 1, 16)`，偏置 `(16,)` |
| `blocks.1.conv1` | 核 `(3, 16, 16)`，偏置 `(16,)` |
| `blocks.1.conv2` | 核 `(3, 16, 16)`，偏置 `(16,)` |

`head` 和 RevIN 仿射参数不融合。

### 6.3 RevIN

```python
def build_revin(num_features: int) -> RevIN:
    ...
```

CNN 的 `num_features=C`，TCN 为 1。`eps=1e-5`，仿射开启，目标通道 0。`affine_weight` 初始为 1，`affine_bias` 初始为 0，形状都是 `(num_features,)`。这两个初值不消耗随机数。

`normalize(x)` 的 `x` 是 `(N, 10, F)`。均值和方差沿时间轴，并 `stop_gradient`。标准差是 `sqrt(variance + 1e-5)`。返回 `x_norm (N, 10, F)`、`mean (N, 1, F)`、`stdev (N, 1, F)`。

销量预测的反变换使用通道 0 的均值、标准差、γ 和 β：

```text
(y - beta) / (gamma + 1e-10) * stdev + mean
```

`y` 的形状是 `(N, 1)`。无 RevIN 的臂不调用这两个函数。反变换结果仍在目标 Min–Max 空间，随后只对销量列做 `inverse_sales`。

## 7. 种子、配对初始化和数据顺序

正式约定是 Stage A 种子等于 cell 种子（`configs/ms_katl_corrected_sita_60cell_manifest.json:43`）。四个随机流分开复位，后一个流不继续使用前一个流留下的全局状态。

| 流 | 种子 | 进入点 |
|---|---|---|
| `stage_a` | cell 种子本身 | 每个 `StageAKey` 训练开始时复位 TensorFlow 一次，然后按入选顺序逐个建源模型并拟合 |
| `target_init` | cell 种子本身 | 在加载融合权重之前，单独复位并建立随机目标模板 |
| `target_shuffle` | cell 种子本身 | `numpy.random.default_rng(cell_seed)`，每个臂新开一个生成器 |
| `random3` | SHA256 派生 | 见下，不使用前三个对象 |

`random3` 的 32 位种子是 `SHA256(f"{cell_seed}|random3|v1")` 小端前四字节的无符号整数，再交给一个新的 `default_rng`。这是接口版本 `v1`，不是冻结 JSON 里的协议常数。改变这个字符串就必须改变 `StageAKey.draw_rule`，使旧缓存失效。

Bootstrap 不属于这四个流。它只用 `20261001`，见第 12 节。

配对规则：

1. 每个 `dataset × horizon × seed × backbone` 只建立一次随机目标模板。模板包含卷积或 TCN 骨干、预测头和 RevIN 的初始值。
2. 所有臂的预测头初始值都来自这份模板。RevIN 的 γ 和 β 在所有臂上都是 1 和 0。
3. `F1` 与 `F2` 的随机骨干权重就是这份模板，因此彼此相同。
4. 有源臂先装入模板，再只覆盖可融合张量。覆盖值由第 9 节按该臂的 `weight_mode` 从同一份源权重现算。`F3`、`F4`、`P1` 使用逆距离平均。`P3` 使用每个 `1/3` 的均匀平均。`P2` 只覆盖 Top-1 源模型的对应张量。逆距离结果不得装入 `P3`。
5. 每个臂在权重已经装入之后、epoch 0 之前，再次把 TensorFlow 复位到 cell 种子。训练期 Dropout 从这次复位开始。验证和测试使用 `training=False`，不抽取 Dropout。
6. 数据顺序是每个 epoch 一次 `generator.permutation(N)`，生成器跨 epoch 连续使用。同一 `RunSpec` 的训练样本数相同，因此同一条件、同一种子、同一骨干下的各臂排列序列相同。这同时满足规范 §4.1 的两对约束和 §9.4 的全体一致要求。

`audit/shared_init_check.json` 记录预测头、RevIN 和 `F1`/`F2` 骨干的 SHA256，并断言上述相等关系。

## 8. `arms`

```python
def expand_runs(phase: Literal[1, 2, 3], *, include_no_revin_variants: bool = False) -> tuple[RunSpec, ...]:
    ...
```

默认 `include_no_revin_variants=False`。Tier 3 的 `X1`、`X2`、`X3` 不在这个函数的返回值里。

| 臂 | 标签 | 初始化 | RevIN | 冻结 conv | 选源 | K | 权重 |
|---|---|---|---|---|---|---|---|
| `F1` | `main` | 模板 | 无 | 否 | 无 | — | — |
| `F2` | `main` | 模板 | 有 | 否 | 无 | — | — |
| `F3` | `main` | 融合 | 无 | 否 | `top` | 3 | 逆距离 |
| `F4` | `main` | 融合 | 有 | 否 | `top` | 3 | 逆距离 |
| `P1` | `main` | 融合 | 有 | 是 | `top` | 3 | 逆距离 |
| `P2` | `main` | Top-1 | 有 | 否 | `top` | 1 | `[1]` |
| `P3` | `main` | 融合 | 有 | 否 | `top` | 3 | 每个 `1/3` |
| `F1`–`F4` | `alt_lr` | 与对应主臂相同 | 与主臂相同 | 与主臂相同 | 与主臂相同 | 与主臂相同 | 与主臂相同 |
| `F4` | `random3` | 融合 | 有 | 否 | `random3` | 3 | 逆距离 |
| `F4` | `bottom3` | 融合 | 有 | 否 | `bottom` | 3 | 逆距离 |
| `F4` | `k5` | 融合 | 有 | 否 | `top` | 5 | 逆距离 |
| `F4` | `k10` | 融合 | 有 | 否 | `top` | 10 | 逆距离 |
| `F4` | `ell15` | 融合 | 有 | 否 | `top` | 3 | 逆距离，比较窗改为 15 天 |
| `SITA_TCN` | `main` | TCN 融合 | 有 | 否 | `top` | 3 | 逆距离 |
| `TCN_REVIN` | `main` | TCN 模板 | 有 | 否 | 无 | — | — |

`ℓ=15` 只改变比较窗口。源历史终点仍是冻结的 30 日窗口终点。`ℓ>30` 直接拒绝。

## 9. `stage_a`

```python
def load_or_train_stage_a(key: StageAKey, frames: ConditionFrames) -> StageABundle:
    ...
```

一个 bundle 的训练顺序是：选源、为每个入选源拟合它自己的源 scaler、只在该源训练窗口上建窗、按第 7 节的一次复位顺序训练。源 CNN 训练是 Adam、MSE、学习率 `1e-4`、50 epoch、batch size 16。batch size 来自正式清单的同一个字段（`configs/ms_katl_corrected_sita_60cell_manifest.json:30`），正式 Stage A 入口把它传给源 CNN（`src/transfer_methods/ms_katl_corrected_sita_60cell.py:637-646`）。源 `fit` 不接收 `validation_data`。

TCN 源拟合以 `configs/tcn_frozen.json` 的 `tcn_source_fit` 为准：Adam、MSE、学习率 `1e-4`、`clipnorm` 为 null、`shuffle` 为 true、50 epoch、batch size 16。MSE 用源训练切片上、源 scaler 做完 Min–Max 的 sales 通道。源 `fit` 不接收验证切片，验证也不参与早停、检查点、源排序、融合权重或最终状态选择。最终缓存是第 50 个 epoch 结束时的 RAW 权重，`final_weights` 为 `last_epoch_raw`，不写 EMA。源侧 RevIN 为 `UNUSED`。不继承 BL7 的 early stopping、patience、best_state，也不继承目标侧 Huber、cosine 或 EMA。每个 `StageAKey` 开始时按 cell 种子复位一次 TensorFlow。CNN 与 TCN 缓存不互读。上述必需字段缺一、为 null（`clipnorm` 的冻结值本身是 null，但字段必须出现），或类型不符时，`SITA_TCN` 保持 `BLOCKED_GAP_TCN_SOURCE_FIT`，不得用默认值补齐。`TCN_REVIN` 不拟合源模型，目标训练路径不变。

源侧验证指标不计算。bundle 里对应字段写成字符串 `NOT_APPLICABLE`，数值空间也是 `NOT_APPLICABLE`。

共享缓存只保存选源结果、距离、每个源的模型权重和该源自己的 scaler。目录里没有融合张量：

```text
消融实验/cache/stage_a/<sha256(canonical_json(StageAKey))>/
    key.json
    selection.json
    source_<rank>.npz
    source_scaler_<rank>.json
    DONE.json
```

`selection.json` 保存入选源、排名和距离。`rank` 从 0 开始，顺序就是训练顺序。`source_<rank>.npz` 保存该源模型的全部权重。`DONE.json` 保存这些文件的 SHA256。逆距离权重和均匀权重都不写进这个目录；它们在初始化时由距离和 `K` 计算。

```python
def fuse_init(
    bundle: StageABundle,
    weight_mode: Literal["inverse_distance", "uniform", "single"],
) -> dict[str, list[ndarray]]:
    ...
```

`inverse_distance` 先计算 `1/(distance+1e-8)` 并归一化，再对全部入选源的可融合张量做位置加权平均。`uniform` 使用 `1/K`，`K=3` 时每个权重是 `1/3`。`single` 只取 `rank=0`，结果等于该源的对应张量。`F3`、`F4` 和 `P1` 调用 `inverse_distance`。`P3` 调用 `uniform`。`P2` 调用 `single`。两次调用可以读同一个 bundle，但必须各自用自己的权重。距离恰好使两种权重相同的情况可以出现；除此以外，不允许把逆距离平均值复制给 `P3`。

融合张量如果要落盘，键必须带上融合规则，不能复用 `StageAKey`：

```python
FusedInitKey = StageAKey 加上 weight_mode
```

```text
消融实验/cache/fused_init/<sha256(canonical_json(FusedInitKey))>/
    key.json
    fused.npz
    DONE.json
```

`key.json` 同时写入 `StageAKey` 和 `weight_mode`。读取时两者都要匹配。`F4` 和 `P1` 可以共用 `inverse_distance` 这份融合文件。`P3` 只能读 `uniform`。`P2` 只能读 `single`。缺 `weight_mode` 的旧式 `fused.npz` 视为无效，不能装入任何臂。

共享和隔离：

- `F3`、`F4`、`P1`、`P3` 以及 S1 里对应的 `F3`/`F4` 读取同一个 CNN 源缓存：`top`、`trained_k=3`、30 日比较窗、同一数据集、horizon 和种子。它们共享的是入选源、距离、源权重和源 scaler。S1 不因为目标学习率不同而重训源模型。`P3` 不共享 `F4` 的融合张量。
- `P2` 读取这个 bundle 的 `rank=0` 权重，不写新的 Stage A 目录，也不把该源单独再训练一遍。它的融合键是 `weight_mode=single`。
- `random3`、`bottom`、`trained_k=5`、`trained_k=10`、15 日比较窗、`tcn` 各自是不同的键。即使入选源和 `K=3` 的前几个重合，也不交叉读取 `.npz`。源模型是一次全局复位后按顺序训练的，顺序不同就不能把权重当成同一个对象。
- 候选不足时写 `selection.json` 和一份 `status=SKIPPED_INSUFFICIENT_CANDIDATES` 的 `DONE.json`，不写权重。

`K=5` 或 `K=10` 可以独立于 `K=3` 训练。不要求先完成 `F4`，也不把前三个源的文件链接到 `F4` 缓存。

## 10. `adaptation`

```python
def adapt_target(
    spec: RunSpec,
    windows: TargetWindows,
    target_scaler: ModelScaler,
    init_template: WeightSnapshot,
    stage_a: StageABundle | None,
) -> AdaptationResult:
    ...
```

`TargetWindows` 含训练、验证和测试三个 `WindowBatch`。无源臂的 `stage_a` 是 `None`。有源臂缺少 bundle 时失败。

优化是 Adam、Huber `delta=1.0`、50 epoch、batch size 16、不早停。学习率日程是 `CosineDecay(initial_learning_rate, decay_steps=epochs*steps_per_epoch)`，不额外传入 `alpha`。`steps_per_epoch = max(1, ceil(N_train/16))`。总梯度步数是 `50 * steps_per_epoch`。样本数小于 16 时，每个 epoch 只有一步，这一步包含全部训练样本。

EMA 动量 `0.99`。初始化为更新前的变量副本。每一步优化之后执行 `ema = 0.99*ema + 0.01*variable`。每个 epoch 结束时的验证使用当前 RAW 权重。全部 epoch 结束后把 EMA 写回变量，测试前向看到的是 EMA。

epoch 0 在任何优化器更新之前，用 RAW 权重对验证窗口计算一次。训练损失和验证损失留在 Huber 所在的缩放空间。记录用的验证 RMSE 和 sMAPE 先做 RevIN 反变换（若开启）和销量反变换，再在原始销量上计算。cell 记录里的验证指标等于最后一个 epoch 的 RAW 原始尺度指标。测试指标使用 EMA。

`P1` 的六个卷积张量 `trainable=False`，并且不进入优化器变量列表。运行结束后它们必须与初始值逐元素相等。RevIN、预测头和其他可训练参数仍更新。

参数偏移对 `conv1`、`conv2`、`conv3`、预测头和 RevIN 分别计算。绝对偏移是该组张量拼接后的 L2。初始范数大于 0 时，相对偏移是绝对偏移除以初始范数。初始范数为 0 时，相对偏移写 `null`，状态写 `UNDEFINED_ZERO_INIT_NORM`，不用 0 或其他数代替。RevIN 的 β 初值是 0，所以这一组在未改动时会进入这个状态。

返回字段至少包括：每个 epoch 的训练损失、验证损失、验证 RMSE、epoch 0 验证 RMSE、总梯度步数、RAW 与 EMA 的偏移、测试预测 `(180,)`。预测与 `windows.test.label_date` 对齐，并且这 180 个日期等于冻结测试区间。`unscored_test_dates` 为空。

## 11. `metrics` 与 `io`

```python
def evaluate_original_sales(y_true: ndarray, y_pred: ndarray) -> MetricPair:
    ...
```

输入都是 `(N,)` 原始销量，反变换之后、裁剪之前。函数先把负数裁成 0，再计算：

```text
RMSE = sqrt(mean((pred - true)^2))
sMAPE = 100 * mean(2*|pred-true| / (|true| + |pred| + 1e-8))
```

分母为零的样本不删除。`MetricPair` 包含 `rmse`、`smape`、`zero_sales_count`、`smape_zero_denominator_count`、`clipped_true_negative_count`、`clipped_pred_negative_count`。零销量天数是裁剪前真实值等于 0 的样本数。sMAPE 零分母数是裁剪后 `|true|+|pred|==0` 的样本数。

`signed_error = mean(pred - true)`，在与 RMSE 相同的裁剪后原始销量上计算。正值是高估。`alpha_entropy = -sum(alpha_i * ln(alpha_i))`，`ln` 为自然对数；每个权重先裁到 `[1e-12, 1]`，裁剪后的 0 项按 `0 * ln(0) = 0`。不记录归一化熵。有效源数 `N_eff = 1/sum(alpha_i^2)` 仍是 `effective_source_count`，用裁剪前的权重。无源臂的 `alpha_entropy` 为 `null`。

```python
def write_run_atomically(directory: Path, payload: RunPayload) -> DoneRecord:
    ...
def inspect_done(directory: Path) -> ResumeDecision:
    ...
```

`RunPayload` 先写入临时文件，再在同一目录重命名。顺序是预测文件、`cell_record.json`、日志已单独存在，最后才写 `DONE.json`。没有 `DONE.json` 的目录视为未完成。

预测文件是 `predictions.parquet`，列固定为：

| 列 | 类型 |
|---|---|
| `label_date` | 日期 |
| `y_true_original` | `float64` |
| `y_pred_original` | `float64` |
| `weight_variant` | 测试行固定为 `EMA` |

`cell_record.json` 是 schema v1。顶层键只有 `CELL_RECORD_KEYS`，见 `DECISIONS.md` 的 GAP-IO。`run_spec` 是第 2 节的 13 个字段。无源臂的 `source_info` 是字符串 `ABSENT`，`alpha_entropy` 为 `null`。有源臂在源信息块里写 `source_validation_metric = NOT_APPLICABLE`、`source_validation_value_space = NOT_APPLICABLE`，并写入 OPEN-2 的 `alpha_entropy`。顶层 `signed_error` 是 OPEN-1 的有限数。

版本字段只包括四项，含义分开：

| 字段 | 含义 |
|---|---|
| `git_head` | 代码提交 |
| `workspace_dirty` | 工作区是否有未提交改动 |
| `frozen_config_sha256` | 三份冻结 JSON 原始字节的聚合哈希 |
| `config_digest` | `SHA256(canonical_json({"frozen_config_sha256", "run_spec"}))` |

canonical JSON 使用 UTF-8、递归 `sort_keys`、`separators=(",", ":")`、`allow_nan=False`，不把 `git_head` 算进去。`run_spec` 使用完整 13 字段。库版本只写 `tensorflow_version`。

`n_train` 与 `n_validation` 等于实际目标 `WindowBatch` 的样本数，冻结值是 `6 - horizon`。`n_test` 是 `180`。窗口构造、adaptation 和 IO 都校验这个实际数量。IO 不在上游没给出样本数时自行补上。

`started_at` 和 `finished_at` 是 UTC 时间，文本形如 `2026-10-02T01:02:03.123456+00:00`，始终保留六位微秒。naive datetime 被拒绝。其他时区先换算成 UTC。`finished_at` 不得早于 `started_at`。`elapsed_seconds` 等于这两个 UTC 时间相减的 `total_seconds()`。

`raw_shift` 和 `ema_shift` 都写入。下游再决定使用 RAW 还是 EMA。

已完成 cell 写 `prediction_path: "predictions.parquet"`。这是相对当前 `run_directory` 的路径。预测 SHA256 仍在该记录的 `artifact_sha256` 里，并且这项只包含 `predictions.parquet`。

`random3` 与 `bottom3` 的已完成 cell 可以在 `source_info` 里多一个 `top3_overlap_count`。它是同一 dataset、horizon、seed、30 日窗口、`K=3` 下与 top/main 选择的源键交集，取值 0 到 3。只有两侧都是 `COMPLETED` 才写。缺失时省略字段，不写 0。这个整数由选源或 runner 计算后传入。

`DONE.json` 字段：

| 字段 | 含义 |
|---|---|
| `run_spec` | 第 2 节的身份 |
| `status` | `COMPLETED` 或 `SKIPPED_INSUFFICIENT_CANDIDATES` |
| `artifact_sha256` | 完成 cell 含预测文件和 `cell_record.json`；候选不足时只有 `selection.json` |
| `stage_a_key_sha256` | 无源臂为 `null` |
| `started_at` `finished_at` `elapsed_seconds` | 完成时刻，格式同上 |

候选不足时目录里只有 `selection.json` 和 `DONE.json`。不写预测文件、cell 记录、空文件哈希或虚假预测路径，也不把这种情况当成失败去重试。

续跑决定：

| 磁盘状态 | `ResumeDecision` |
|---|---|
| 没有 `DONE.json` | `RUN` |
| `DONE.json` 可解析且文件哈希一致 | `SKIP` |
| `DONE.json` 存在但哈希或身份不符 | `HOLD_HASH_MISMATCH`，不覆盖、不删除、不自动重跑 |
| 目标 cell `FAILED.json` 且正文只有 `attempt_count=1` | `RETRY_ONCE` |
| 目标 cell `FAILED.json` 且正文只有 `attempt_count=2` | `HOLD_FAILED` |

目标 cell 的 `FAILED.json` 正文只能是 `{"attempt_count": 1}` 或 `{"attempt_count": 2}`。异常文本写入该 cell 的日志。多出来的字段被拒绝。这条规则不适用于 Stage A cache：那里的 `FAILED.json` 仍可包含 `stage_a_key_sha256` 和 `error`。

同一 `RunSpec` 最多由 runner 再试一次，种子不变。两次失败都保留。候选不足不是失败，不重试。

有 RevIN，或臂不是 `P1`/`P2`/`P3` 时，日志名是 `消融实验/logs/<arm>__<protocol_tag>__<dataset>_h<horizon>_seed<seed>.log`。无 RevIN 的 `P1`、`P2`、`P3` 在 `.log` 前追加 `__revin0`。`F1` 和 `F3` 不追加。Stage A 键和 `run_directory` 不因这条日志后缀改变。

数据哈希在每次续跑开始时重新计算八个 parquet。与 `audit/data_freeze_hashes.json` 不一致时整个 runner 停止。这个审计文件尚未在本轮创建；实现阶段第一次运行前写入，其后只校验。

## 12. `runner`

```python
python -m sita_ablation.runner --phase {1|2|3} --arms ... --workers N [--dry-run] [--only-cell D1:1:42]
```

`--workers` 没有冻结默认值，调用时必须给出，且大于等于 1。每个进程设置 TensorFlow `intra_op_parallelism_threads=1` 和 `inter_op_parallelism_threads=1`。进程上限是当时机器物理核数减 1；接口不把核数写成常数。

每个 cell 开始时调用 `tf.config.experimental.enable_op_determinism()`。确定性抽查与正式 cell 使用同一设置。抽查重复三个 cell，比较预测逐元素是否相同；不同则记录最大绝对差，不改写已有 `DONE.json`。

`--dry-run` 只展开 `RunSpec`、缓存键和将要跳过的目录，不建网络、不写预测。`--only-cell` 的三个整数或字符串按 `dataset:horizon:seed` 过滤。

正式 runner 启动前调用 `require_clean_workspace_for_formal_run`。`workspace_dirty=True` 时拒绝启动。不用代码 diff digest 代替这个判断。`tcn_source_fit` 完整且类型正确后，GAP-TCN-SOURCE-FIT 不再单独阻断。字段不完整时仍阻断，不自动补默认值。OPEN-1 与 OPEN-2 已有公式，不再单独阻断。

Phase 结束时审计文件写到 `消融实验/audit/phase_<phase>_audit.json`。`write_phase_audit` 校验 phase、文件名和 cell ID。四个分类名固定为 `missing`、`hash_mismatch`、`failed`、`insufficient_candidates`。每个 cell ID 形如 `F4:random3:D1:h3:seed42`。列表按确定顺序写出，同一 ID 不能重复，也不能出现在两个分类里。没有传入的 cell 保持缺行，分析时不得填补。

## 13. `analysis`

```python
def condition_mean(errors: ndarray) -> ndarray:
    # errors: (12, 5) float64；返回 (12,)
def relative_improvement(error_x: ndarray, error_y: ndarray) -> ndarray:
    # 两个 (12,)；RI = (E_Y - E_X) / E_Y * 100
def win_loss_tie(ri: ndarray) -> WinLossTie:
    ...
def paired_wilcoxon(ri: ndarray) -> WilcoxonResult:
    ...
def percentile_bootstrap(ri: ndarray, *, n_resamples: int = 100_000, seed: int = 20261001) -> Interval:
    ...
def holm(p_values: ndarray) -> ndarray:
    ...
```

`relative_improvement` 在 `E_Y == 0` 时抛出，不返回无穷或 0。`paired_wilcoxon` 调用 `scipy.stats.wilcoxon(ri, zero_method="wilcox", method="auto", alternative="two-sided")`，并把实际使用的方法记入结果。Bootstrap 对 12 个条件的 RI 有放回抽取 100000 次，取每次平均值的 2.5% 和 97.5% 分位。生成器是 `default_rng(20261001)`。

Holm 只在族 A 的五项比较内部、以及族 B 的四项比较内部进行。RMSE 与 sMAPE 各做一次。交互项 `I_c = RI(F4, F3) - RI(F2, F1)` 单独做 Wilcoxon 和 bootstrap，不进入 Holm。G5 和 S1 至 S4 报告未校正 p 值和 bootstrap 区间。

`tables` 和 `figures` 只消费这些结果和 cell 记录。输出文件名沿用规范 §7。`analysis/restat_vs_baselines.py` 的基准 CSV 由调用者传入路径，接口不搜索旧结果目录。

任一比较的十二个条件里有缺失 cell 时，该比较标记为缺失并在表注写明，不换用其他种子或配置。

## 14. 实现前仍不能填的字段

`signed_error` 与 `alpha_entropy` 按 `DECISIONS.md` 的 OPEN-1 和 OPEN-2 计算。GAP-TCN-SOURCE-FIT 已关闭；`tcn_source_fit` 不完整时仍不开始正式实验。
