# Source Pool 构建逻辑审计：build_paper_source_pool()

## 1. 审计目标

本次审计要回答的问题是：现有框架中 `build_paper_source_pool()` 函数目前按什么条件筛选 source 实体。

审计方式为静态代码阅读和调用链追踪，不运行正式训练实验，不修改核心实验逻辑。重要发现是：当前代码库中没有找到名为 `build_paper_source_pool()` 的函数定义或调用；当前框架实际承担 source pool 构建职责的是 `build_source_target_split()`，正式全量实验脚本还会在其结果上叠加 `_apply_information_sharing_filter()`。

## 2. 函数定位

- 精确函数名：`build_paper_source_pool`
- 函数定义文件路径：代码中未确认。
- 起止行号：代码中未确认。
- 所属模块：代码中未确认。
- 精确调用位置：0 处。执行 `rg -n "def build_paper_source_pool|build_paper_source_pool\(" --glob '!docs/**' --glob '!_git_backup/**' --glob '!数据集/**' .` 没有代码匹配。

当前实际 source pool 构建链路：

- `data_preprocessing.py:476-664`：`build_source_target_split(df, config)`，从标准化后的 DataFrame 中切出 `source_df` 和 `target_df`。
- `src/data_processing/data_preprocessing.py:476-664`：同名镜像实现，内容与根目录版本一致。
- `experiment_runner.py:624-647`：`prepare_base_data_for_experiments()` 调用 `build_source_target_split()` 并返回 `source_df` / `target_df`。
- `src/experiment/experiment_runner.py:430-455`：`prepare_base_data_for_experiments()` 的 `src` 版本也调用 `build_source_target_split()`。
- `scripts/run_full_paper_experiments.py:373-442`：`_apply_information_sharing_filter()` 在全量 paper runner 中进一步按 information sharing 场景过滤 source pool。
- `scripts/run_full_paper_experiments.py:483-491`：`run_experiment()` 调用 `_apply_information_sharing_filter()`。

## 3. 函数输入与输出

由于 `build_paper_source_pool()` 不存在，下面按当前实际链路说明。

| 参数名 | 类型或推测类型 | 来源 | 作用 |
|---|---|---|---|
| `df` | `pd.DataFrame` | `extract_datetime_features(raw_df)` 输出，见 `experiment_runner.py:624-626` | 作为 source/target 初筛的完整候选数据 |
| `config` | `dict` 或配置对象 | runner 传入，见 `experiment_runner.py:615-626` | 读取 `dataset_name`、`paper_reproduction.*`、`preprocessing.*` 等配置 |
| `dataset_name` | `str` | `config.dataset_name` 或 `df.attrs["dataset_name"]`，见 `data_preprocessing.py:494-498` | 决定 Dataset1/2/3 的严格协议分支 |
| `strict_paper_mode` | `bool` | `paper_reproduction.strict_paper_mode` 或 `paper_strict_mode`，见 `data_preprocessing.py:203-208` | 是否启用 Dataset1/2/3 的严格 source/target 规则 |
| `strict_paper_split` | `bool` | `paper_reproduction.strict_paper_split`、`paper_strict_split` 或 strict mode，见 `data_preprocessing.py:259-266` | 只约束 target 时间窗口完整性，不直接筛 source entity |
| `paper_split_protocol` | `dict` | `paper_reproduction.paper_split_protocol` 或 fallback，见 `data_preprocessing.py:211-245` | 提供 target observed/test 天数；`source_selection_window` 只被记录，不在 source 初筛中使用 |
| `source_df` | `pd.DataFrame` | `build_source_target_split()` 返回 | source pool 的 DataFrame 表示 |
| `target_df` | `pd.DataFrame` | `build_source_target_split()` 返回 | target 序列；正式全量脚本用它决定 no-sharing 的实体/区域过滤 |
| `use_information_sharing` | `bool` | `scripts/run_full_paper_experiments.py:478` | 决定 `_apply_information_sharing_filter()` 保留全量 pool 还是局部 pool |
| `protocol` | `dict` | `load_paper_protocol(config)` | Dataset3 region 字段、严格 top-k 等后续协议信息 |

返回值数据结构：

- `build_source_target_split()` 返回 `(source_df, target_df)`，二者都是 `pd.DataFrame`，见 `data_preprocessing.py:664`。
- `source_df` 不是 list 或 dict，而是包含多条 source 时间序列行的 DataFrame。每个 source entity 在后续 KNN 中按 `("entity_id", "item_id")` 分组，见 `source_selector.py:457-507` 和 `source_selector.py:702-719`。
- `source_df` 行字段来自标准化后的数据，包括至少 `date`、`entity_id`、`item_id`、`sales` 以及时间特征/数据集特征；具体列由各数据集标准化和特征推断决定。
- `source_df.attrs` 会记录 `split_role`、`split_mode`、`split_config`、`strict_paper_mode`、`paper_split_protocol`、`strict_dataset_name`，见 `data_preprocessing.py:630-652`。
- `_apply_information_sharing_filter()` 返回过滤后的 `pd.DataFrame`，并设置 `information_sharing_scenario`、`signature_static_feature_cols`、`source_pool_scope_mode`，见 `scripts/run_full_paper_experiments.py:390-435`。
- 后续使用者包括 `SourceSelector.select_top_k_sources()`，以及 `SS-TL`、`MSWA-TL`、`MSSB-TL`、`MSML-TL`、`MSML-TL-RFE` 方法；例如 `experiment_runner.py:683-695` 和 `mswa_tl.py:452-464`。

## 4. 当前实际筛选条件

| 筛选条件 | 代码逻辑 | 保留对象 | 排除对象 | 影响 |
|---|---|---|---|---|
| 精确 `build_paper_source_pool()` | 代码中未确认。精确搜索无定义、无调用 | 不适用 | 不适用 | 说明当前框架没有一个独立同名 source pool 构建函数 |
| 全局排序 | `sorted_df = df.sort_values(["date", "entity_id", "item_id"])`，见 `data_preprocessing.py:494` | 所有输入行，但排序稳定 | 不排除行 | 不改变 pool 内容，只影响后续分组/序列顺序 |
| Dataset1 严格协议：允许实体 | strict mode 且 Dataset1 时，`allowed_entities` 默认 `[1,2,3]`，先执行 `entity_id.isin(allowed_entities)`，见 `data_preprocessing.py:505-518` | allowed entities 中的 source/target 候选 | 不在 allowed set 中的实体 | 会显著缩小 source pool，影响可迁移 source 范围 |
| Dataset1 严格协议：source item | source 取 `source_item_ids` 默认 `1..9`，见 `data_preprocessing.py:505-518` | `item_id` 在 `1..9` 的行 | 默认排除 `item_id=10` 的 target 商品和其他非 source item | 会按商品 id 固定 source pool，但不排除 target entity 的其他 source items |
| Dataset1 严格协议：target | target 取 `entity_id=1` 且 `item_id=10`，见 `data_preprocessing.py:507-515` | 单个 target entity-item | 非 target 行 | 只影响 target_df，不直接保证 source 不含同 entity |
| Dataset2 严格协议：target item 排除 | target 取 `entity_id="B1"` 且 `item_id=10`；source 取 `item_id != 10`，见 `data_preprocessing.py:519-528` | 所有非 target item 行，跨 entity/brand | 所有 `item_id=10` 行，包括非 B1 的同 item | source pool 可能很大；保留 B1 的 item 1-9 |
| Dataset3 严格协议：target store/item id 排除 | `target_store_id` 默认 10；source 取 `item_id != 10`，target 取 `item_id == 10`，见 `data_preprocessing.py:529-532` | 非 10 的 item/store 行 | `item_id=10` 行 | 以 `item_id` 承载 store id，排除 target store 本身对应行 |
| 非严格/通用分支：推断 source/target item | `_infer_source_target_items()` 从 config 或默认规则推断；target 优先 config 的 `dataset.target_product_id`，否则 item 10，最后取最大 item；source 默认 `1..9` 或所有非 target item，见 `data_preprocessing.py:444-473`、`538-540` | 推断出的 source items | 推断出的 target items | 与数据集具体论文设定不一定一致，属于工程默认 |
| target 时间窗口裁剪 | target 只保留最近 `train+val+test` 天，见 `data_preprocessing.py:542-607` | target 最近窗口行 | target 更早历史 | 只影响 target_df；source_df 仍使用完整历史 |
| source split 元数据 | source 记录 ratio split config，见 `data_preprocessing.py:630-648` | source 全历史 DataFrame | 不排除 source 行 | source 的 train/val/test 是后续 temporal split，不是 pool 构建过滤 |
| with information sharing | `_apply_information_sharing_filter()` 在 `use_information_sharing=True` 时直接返回 source_df，见 `scripts/run_full_paper_experiments.py:399-401` | 初筛后的完整 source pool | 不额外排除 | pool 最大，允许跨 entity/store/brand/region |
| without information sharing：默认同 entity | 先取 `target_entities`，再 `source_df["entity_id"].isin(target_entities)`，见 `scripts/run_full_paper_experiments.py:403-405` | 与 target entity_id 相同的 source 行 | entity_id 不在 target entities 的 source 行 | 缩小 no-sharing pool；Dataset1/2 标签上标注 same_store/same_brand，但实际判断列是 `entity_id` |
| without information sharing：Dataset3 same region | strict mode + Dataset3 时，如果 region 字段在 source/target 都存在，按 target region 过滤；见 `scripts/run_full_paper_experiments.py:406-420` | 与 target region 相同的 source 行 | 不同 region 的 source 行 | 有 region 元数据时更贴近 same-region 规则 |
| Dataset3 region fallback | 如果 region 字段不可用或过滤为空，则返回完整 source_df 并标注 fallback，见 `scripts/run_full_paper_experiments.py:421-428` | Dataset3 初筛后的完整 source pool | 不额外排除 | no-sharing 下可能退化为跨 region pool，影响论文一致性 |
| without information sharing 空池检查 | 如果过滤后为空，抛出错误，见 `scripts/run_full_paper_experiments.py:437-441` | 非空 filtered pool | 空 pool | 防止继续训练空 source pool |
| KNN top-k 后续选择 | `select_top_k_sources()` 对已有 source_df 分组、算距离并取 `min(k, len(source_keys))`，见 `source_selector.py:867-889` | 距离最近的 k 个 source group | 距离排序靠后的 source group | 这是后续 source selection，不是 source pool 构建；影响最终训练使用的 source 数量 |
| 严格 multi-source top-k=3 后续限制 | strict mode 且 multi-source 方法时 source_count 设为协议中的 `multi_source_top_k`，见 `scripts/run_full_paper_experiments.py:480-481`、`configs/default_config.json:216-221` | KNN 最终选出的 3 个 source | KNN 排名第 4 及之后 | 限制训练用 source 数量，但不限制 pool 大小 |

## 5. 当前没有筛选但理论上可能需要检查的条件

| 条件 | 当前是否检查 | 代码依据 | 风险 |
|---|---|---|---|
| source entity 时间跨度 | pool 构建层未检查 | `source_df` 构建后仅设置 attrs，见 `data_preprocessing.py:630-664` | 时间跨度不足的 source 可能进入 pool；后续 KNN observed sequence 可能在覆盖不足时报错 |
| source 有效销售天数 | 未检查 | 未见按非零销售天数过滤 source 的逻辑；`source_selector.py:407-420` 直接按 group 构签名 | 低有效天数 source 可能参与 KNN 或训练 |
| 全零销量序列 | 未检查 | `source_selector.py:278-283` 将数值转向量，不排除全零；pool 层也无 all-zero 过滤 | 全零/低信息 source 可能被选中，影响距离和迁移质量 |
| 缺失严重实体 | pool 构建层未检查 | `extract_datetime_features()` 有全局 `dropna()`，见 `data_preprocessing.py:438`，但没有按 entity 缺失率过滤 | 全局缺失行被删，实体级缺失密度未控制 |
| source 是否覆盖 observed window | pool 构建层未检查；KNN observed sequence 会检查 | `source_selector.py:263-276` 要求每个 source 覆盖 target observed dates 和相同行数 | 风险从 pool 构建推迟到 KNN 阶段，可能导致运行时失败 |
| source 是否覆盖 test window | 未检查 | KNN 只声明排除 target test，见 `source_selector.py:285-315`；source pool 无 test 覆盖判断 | source 训练可用性依赖后续 temporal split，不在 pool 阶段保证 |
| source 和 target 时间对齐 | pool 构建层未检查；KNN observed sequence 阶段检查 observed dates | `source_selector.py:394-420` 按 target dates 构 source signature | 不使用 KNN 的流程或工程摘要模式下可能缺少严格对齐 |
| 区分 source domain 和 target domain | 部分检查 | Dataset1/2/3 strict 分支按 item/entity/region 做近似，见 `data_preprocessing.py:505-532`、`scripts/run_full_paper_experiments.py:403-435` | domain 定义依赖字段语义；Dataset2 same_brand 实际用 entity_id overlap 表达 |
| 强制排除 target entity 自身 | 未强制排除 entity 自身；只排除 target item/store id | Dataset1 source 可包含 target entity=1 的 item 1-9，见 `data_preprocessing.py:511-518`；Dataset2 source 可包含 B1 的 item 1-9，见 `data_preprocessing.py:522-528` | 如果论文要求 source 与 target entity 完全不同，当前实现不满足；代码中未确认该要求 |
| 强制排除 target entity-item 自身 | 大多通过 target item 排除实现 | Dataset1/2 排除或不纳入 `item_id=10`；Dataset3 排除 `item_id=10`，见 `data_preprocessing.py:516-532` | 可避免同一 target item 进入 source，但不等价于排除 target entity |
| 控制 source pool 数量 | pool 层未控制 | pool 构建返回 DataFrame，无 top-k；top-k 在 `source_selector.py:886-889` | pool 规模可能很大，性能和 KNN 失败面增加 |
| 使用 KNN 距离筛选 pool | 否，KNN 是后续选择 | `SourceSelector.select_top_k_sources()` 的输入已经是 source_df，见 `source_selector.py:702-719` | 不能把 KNN top-k 当作 pool 构建条件 |
| 使用 RFE 筛选 source | 否，RFE 在 KNN 选源之后筛特征 | `msml_tl_rfe.py:1104-1124` 先 KNN 选源，`msml_tl_rfe.py:1187-1225` 后 RFE | RFE 不会清理 source pool，只影响特征子集 |
| 使用 `source_history_days` | 代码中未确认 | 精确搜索 `source_history_days` 在核心文件中无匹配 | source 历史长度没有由该字段约束 |
| cold-start window 检查 | pool 层未检查 | 核心 pool 构建未见 `cold_start` / `cold-start` 条件 | cold-start 任务主要由 target 窗口体现，source 可行历史未单独验证 |
| 论文中的 source selection 规则 | 只能部分确认 | 配置写有 `source_pool_scope=all_source_items` 和 strict top-k，见 `configs/default_config.json:151-158`、`216-221` | 当前代码无法证明严格来自论文，存在工程近似 |

## 6. 与论文设定的关系

当前实现更接近：

3. 先粗筛 source pool，后续再由 KNN / RFE / TL 模块细筛。

理由：

- pool 构建层输出的是 `source_df` DataFrame，而不是最终 selected source 列表，见 `data_preprocessing.py:476-664`。
- KNN top-k 在 `SourceSelector.select_top_k_sources()` 内执行，输入参数已经是 source pool，见 `source_selector.py:702-719`。
- MSML-TL-RFE 先执行 KNN 选源，再构建联合 RFE 训练数据并做特征筛选，见 `msml_tl_rfe.py:1104-1124`、`1187-1225`。
- 配置中有论文对齐字段，例如 `source_pool_scope=all_source_items` 和 strict top-k=3，见 `configs/default_config.json:151-158`、`216-221`，但 pool 构建本身没有完整验证 source 时间跨度、有效销售天数、销量密度等论文可能要求。

当前代码无法证明该逻辑严格来自论文，只能说明这是当前工程实现中的 source pool 构建方式。

## 7. 对 D1-D6 数据集扫描的影响

当前代码的正式 registry 只枚举 Dataset1-Dataset3：`dataset_registry.py:12-65` 中 `DATASET_REGISTRY` 只有 Dataset1/2/3，`list_dataset_names()` 返回 `["Dataset1", "Dataset2", "Dataset3"]`。正式 runner 也只覆盖这三个路径，见 `scripts/run_full_paper_experiments.py:894-901`。因此 D4-D6 的 source pool 扫描影响在当前代码中未确认。

基于当前 Dataset1-Dataset3 逻辑：

- source pool 的规模可能过大。Dataset2 strict source 是所有 `item_id != 10` 的行，Dataset3 region fallback 会返回完整 source pool，见 `data_preprocessing.py:526-532`、`scripts/run_full_paper_experiments.py:421-428`。
- 可能混入时间跨度不足的实体。pool 构建层不检查 source group 的日期覆盖；只有 KNN observed sequence 在构签名时检查 target observed dates，见 `source_selector.py:263-276`。
- 可能混入全零或低密度实体。pool 构建层没有非零销售天数、销量密度或全零序列过滤。
- 可能把目标实体也放进 source pool。Dataset1 和 Dataset2 会保留 target entity 的其他 source items；without-sharing 甚至会按 target entity_id 保留同 entity source，见 `data_preprocessing.py:511-528`、`scripts/run_full_paper_experiments.py:403-405`。
- cold-start 任务设计可能被削弱。如果 source pool 中包含 target entity 的其他长期 item，任务更像同实体跨 item 迁移；代码中未确认是否符合论文对 cold-start source domain 的限制。
- `source_history_days` 的可行范围不由 pool 构建保证。当前 source 使用完整历史并设置 ratio split，见 `data_preprocessing.py:630-648`，代码中未确认 `source_history_days` 字段。
- observed window 主要影响 target 和 KNN，而不是 pool 构建。target 最近窗口在 `data_preprocessing.py:590-607` 裁剪；KNN observed window 在 `experiment_runner.py:156-173` 或 `source_selector.py:285-315` 处理。

## 8. 结论摘要

- 当前代码库中没有 `build_paper_source_pool()` 的定义或调用。
- 当前 source pool 初筛由 `build_source_target_split()` 完成，返回 `source_df` / `target_df` 两个 DataFrame。
- 严格模式下 Dataset1 按 allowed entities 和 source item ids 筛 source；Dataset2 按 `item_id != target_item_id` 筛 source；Dataset3 按 `item_id != target_store_id` 筛 source。
- 正式全量脚本会按 information sharing 场景进一步过滤：with-sharing 保留完整 pool；without-sharing 默认保留 target entity_id 对应 source，Dataset3 有 region 过滤与 fallback。
- 当前实现通常排除 target item/entity-item，但没有强制排除 target entity 自身。
- source pool 层不限制数量；top-k 或 top-3 是后续 KNN source selection 的结果，不是 pool 构建条件。
- source pool 层不检查 source 时间跨度、有效销售天数、全零序列、销量密度或缺失率。
- RFE 不参与 source pool 构建；它在 KNN 选出 sources 后筛选特征。
- 最大风险是 pool 过粗，可能把低质量、时间不对齐或 target entity 自身的 source 序列带入后续 KNN/训练。
- 是否需要修复取决于论文复现实验是否要求严格 source domain、source 历史长度和有效销量密度约束；当前代码中未确认这些规则。

## 9. 后续建议

### 必须确认

- 确认 `build_paper_source_pool()` 是否是旧函数名、计划函数名，还是外部分支中的函数。
- 确认论文对 source pool 的实体排除规则：是否允许 target entity 的其他 items 作为 source。
- 确认 Dataset2 的 same-brand 语义是否等价于当前 `entity_id` 过滤。
- 确认 Dataset3 是否需要外部 region metadata；当前 fallback 会让 no-sharing 退化为完整 pool。

### 建议补充

- 增加只读审计脚本，统计每个 source group 的日期跨度、unique days、非零销售天数、缺失率、全零标记。
- 在结果记录中显式输出 source pool group 数量、row 数量、target entity 是否在 pool 中。
- 为 KNN observed sequence 的覆盖失败增加 pool 级预检报告，避免训练阶段才失败。
- 明确区分 `source_pool_candidates`、`knn_selected_sources`、`rfe_selected_features` 三类概念。

### 可暂缓

- 暂缓修改训练逻辑。
- 暂缓将 KNN/RFE 规则上移到 source pool 构建层，先完成论文规则确认。
- 暂缓 D4-D6 结论，因为当前 registry 和正式 runner 未枚举 Dataset4-Dataset6。

## 10. 附录：关键代码摘录

文件路径：`data_preprocessing.py:444-473`

作用说明：通用分支推断 source/target item。

```python
source_items = _get_cfg(config, "preprocessing.source_item_ids", None)
target_items = _get_cfg(config, "preprocessing.target_item_ids", None)

if source_items and target_items:
    return [int(x) for x in source_items], [int(x) for x in target_items]

target_item_from_cfg = _get_cfg(config, "dataset.target_product_id", None)
...
if target_item == 10 and all(item in unique_items for item in default_source_block):
    source = default_source_block
else:
    source = [int(x) for x in unique_items if int(x) not in target]
```

文件路径：`data_preprocessing.py:502-532`

作用说明：strict paper mode 下 Dataset1/2/3 的 source/target 初筛规则。

```python
if dataset_name == "Dataset1":
    allowed_entities = set(int(v) for v in strict_spec.get("allowed_entities", [1, 2, 3]))
    target_entity_id = int(strict_spec.get("target_entity_id", 1))
    target_item_id = int(strict_spec.get("target_item_id", 10))
    source_item_ids = set(int(v) for v in strict_spec.get("source_item_ids", [1, 2, 3, 4, 5, 6, 7, 8, 9]))

    narrowed = sorted_df[sorted_df["entity_id"].isin(allowed_entities)].copy()
    target_df = narrowed[
        (narrowed["entity_id"].astype(int) == target_entity_id)
        & (narrowed["item_id"].astype(int) == target_item_id)
    ].copy()
    source_df = narrowed[
        narrowed["item_id"].astype(int).isin(source_item_ids)
    ].copy()
elif dataset_name == "Dataset2":
    ...
    source_df = sorted_df[
        pd.to_numeric(sorted_df["item_id"], errors="coerce") != target_item_id
    ].copy()
elif dataset_name == "Dataset3":
    target_store_id = int(strict_spec.get("target_store_id", 10))
    source_df = sorted_df[pd.to_numeric(sorted_df["item_id"], errors="coerce") != target_store_id].copy()
```

文件路径：`data_preprocessing.py:542-664`

作用说明：target 时间窗口裁剪和 source attrs 设置；source 本身不按时间跨度裁剪。

```python
paper_split_protocol = _resolve_paper_split_protocol(config)
target_train_val_days = _safe_int(paper_split_protocol.get("target_observed_window_days", 30), 30)
target_test_days = _safe_int(paper_split_protocol.get("target_forecast_window_days", 180), 180)
...
target_max_date = target_df["date"].max()
target_min_date = target_max_date - pd.Timedelta(days=total_days - 1)
target_df = target_df[target_df["date"] >= target_min_date].copy()
...
source_df.attrs["split_role"] = "source"
source_df.attrs["split_mode"] = _get_cfg(config, "preprocessing.source_split_mode", "ratio")
source_df.attrs["split_config"] = {
    "train_ratio": _get_cfg(config, "preprocessing.source_train_ratio", 0.8),
    "val_ratio": _get_cfg(config, "preprocessing.source_val_ratio", 0.1),
    "test_ratio": _get_cfg(config, "preprocessing.source_test_ratio", 0.1),
    "date_boundaries": _get_cfg(config, "preprocessing.source_date_boundaries", {}),
}
return source_df, target_df
```

文件路径：`scripts/run_full_paper_experiments.py:373-442`

作用说明：正式全量实验中的 information-sharing source pool 过滤。

```python
if use_information_sharing:
    source_df.attrs["source_pool_scope_mode"] = "with_information_sharing_full_pool"
    return source_df

target_entities = set(target_df["entity_id"].dropna().unique().tolist())
filtered = source_df[source_df["entity_id"].isin(target_entities)].copy()

if strict_paper_mode and dataset_name == "Dataset3":
    ...
    if not region_filtered.empty:
        filtered = region_filtered
        filtered.attrs["source_pool_scope_mode"] = "without_information_sharing_same_region"
        return filtered
    filtered = source_df.copy()
    filtered.attrs["source_pool_scope_mode"] = "without_information_sharing_region_fallback"
    return filtered
```

文件路径：`source_selector.py:702-719`、`source_selector.py:867-889`

作用说明：KNN 后续 top-k source selection，不是 pool 构建。

```python
def select_top_k_sources(
    self,
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    group_cols: Tuple[str, str] = ("entity_id", "item_id"),
    ...
) -> Dict[str, object]:
    ...
    distances = self.compute_euclidean_distances(target_signature, source_signatures)
    top_k = min(k, len(source_keys))
    sorted_indices = np.argsort(distances)
    selected_indices = sorted_indices[:top_k]
```

文件路径：`source_selector.py:263-276`、`source_selector.py:407-420`

作用说明：KNN observed sequence 表示会检查每个 source 是否覆盖 target observed dates。

```python
if expected_dates is not None:
    expected = pd.to_datetime(pd.Index(expected_dates)).dropna().sort_values()
    work = work[work["date"].isin(expected)].sort_values(["date"]).reset_index(drop=True)
    if int(work["date"].nunique()) != len(expected):
        raise ValueError(
            "paper_observed_sequence requires each source to cover the target observed dates. "
            f"expected_unique_dates={len(expected)} actual_unique_dates={int(work['date'].nunique())}."
        )
```

文件路径：`msml_tl_rfe.py:1104-1124`、`msml_tl_rfe.py:1187-1225`

作用说明：RFE 在 KNN 选源之后执行，不参与 source pool 构建。

```python
# --- Step 1: 选源 ---
selection_result = selector.select_top_k_sources(
    target_df=selection_target_df,
    source_df=source_df,
    feature_cols=feature_cols,
    k=requested_k,
    ...
)
selected_sources = selection_result.get("sources", []) if isinstance(selection_result, dict) else selection_result
...
# --- Step 4: 构建联合 RFE 训练数据 ---
joint_train_df = build_joint_rfe_training_dataframe(...)
# --- Step 5: 执行 RFE ---
rfe_result = run_rfe_feature_selection(...)
```
