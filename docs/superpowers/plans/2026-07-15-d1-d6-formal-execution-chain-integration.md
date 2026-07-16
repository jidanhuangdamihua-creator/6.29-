# D1–D6 Formal Execution Chain Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改既有 D1–D6 封存设计的前提下，将已经完成的 Task 1–13 协议模块迁移到唯一官方正式入口，使正式执行从 sealed 数据开始，经 truth-free 联合预测、typed artifacts、fenced 原子发布和完整验收，最终可靠到达 `SEALED_SUCCESS`。

**Architecture:** 保留现有设计和协议模块，新增一个统一的 seed-bundle worker，令 D1–D6 六个数据集全部走同一条正式调用链。`run_unified_d1_d6.py` 继续作为单一生命周期所有者，shell supervisor 只负责有界调度和进程组管理，不再自行定义产物权威或状态。旧 loader、逐 horizon runner 和 legacy publisher 可以保留用于非封存兼容路径，但 formal mode 必须通过结构和测试双重门禁禁止调用它们。

**Tech Stack:** Python 3.9/3.10、pandas、PyArrow、JSON Schema、pytest、Bash、现有 `RunRecovery`/artifact schema/blind rollout/fitted predictor 模块。

## Global Constraints

- 权威设计保持为 `docs/superpowers/specs/2026-07-15-d1-d6-experiment-sealing-design.md`，冻结 SHA-256 为 `914ab6e4b3ac2eca7d2bb1c7cc2811a75c905995269b15b3300b0038f7343f6d`。
- 原 Task 1–13 实施计划保持为历史执行记录，不通过本计划重写，当前 SHA-256 为 `70ae0719ca765051198aa2dd5e5e84bdd6c4e935dc28f8f24681aa687736f6ac`。
- 除非权威设计内部存在无法同时满足的直接矛盾，否则禁止修改设计、增加新协议、扩大数据集范围、增加方法、增加 horizon 或改变 seeds。
- 正式矩阵固定为 6 datasets × 2 modes × 5 seeds = 60 seed bundles；每个 bundle 固定覆盖 6 methods × h1–h5。
- formal mode 只能读取 `configs/sealed_deployments/d1_d6_sealed_v1.json` 声明并精确绑定的 content-addressed immutable sealed root；不得固定回退到旧 `数据集/固化数据/d1_d6_sealed_v1/`。
- worker 不得接收 evaluator truth；worker trace 不得包含 `y_true`；只有 evaluator 可以将 truth 加入 evaluated trace。
- 正式产物只能通过注册 typed schema、冻结 binding、当前 attempt 的 fencing token 和原子目录发布成为 accepted。
- `SEALED_SUCCESS` 必须是成功路径的最后一次写入，只能由 `complete_unsealed` 经完整 Task 12 acceptance 产生。
- 所有 Codex 执行的 Python 测试和验证命令必须使用 `python tools/protection/codex_timeout.py --timeout 180 -- <command...>`；返回 124 时立即停止，不得拆分、重试或继续其他测试。唯一例外是 Gate 1A-X 中由用户在 Terminal/tmux 执行的正式 operator 命令，该命令不得由 Codex运行，也不得使用 180 秒 wrapper。
- 测试使用 `PYTHONDONTWRITEBYTECODE=1`、`-p no:cacheprovider` 和 `/tmp` basetemp；不得创建正式 `outputs/runs/`、不得生成真实全量训练。
- 本计划不以测试数量、模块存在或局部函数通过作为完成证明；完成证明仅来自官方入口的端到端调用链及独立只读审计。

---

## 1. 决策与范围冻结

### 1.1 本次决策

现有设计内容视为已经完成并冻结。本计划不是 Task 1–13 的第二轮设计，也不是再造一套协议，而是唯一的：

```text
Task 14: Integrate the official formal execution path
```

Task 14 只解决一件事：让已经实现的协议能力成为正式 CLI 实际执行的能力。

唯一允许的正式数据流为：

```text
sealed dataset handle
→ worker/evaluator truth-isolated views
→ exact Predictor/KNN Schema gate
→ source 180-day validation and 30-day KNN
→ truth-free fitted predictor bundle
→ blind h1–h5 rollout
→ Worker Prediction Trace
→ evaluator truth join
→ Evaluated Prediction Trace
→ Formal Result Rows
→ fenced atomic bundle publication
→ validate/rehydrate accepted artifacts
→ freeze artifact binding set
→ 12-mode/60-bundle aggregate acceptance
→ complete_unsealed
→ final sealing gate
→ SEALED_SUCCESS
```

### 1.2 明确不做的事项

- 不重新讨论 D1–D6 窗口、方法集合、seeds、horizons、sMAPE 定义或特征角色。
- 不新增第七种方法，不扩大到 D7，不调整论文协议。
- 不为了兼容旧 runner 而削弱 truth 隔离、schema、trace、recovery 或 final gate。
- 不把旧输出迁移成新 sealed 输出；旧产物只能保留为 legacy audit evidence。
- 不用真实训练证明调度正确；端到端 gate 使用小型 deterministic fixture 和 fake fitted predictors。
- 不在 Task 14 中进行无关重构、性能优化或代码风格统一。

### 1.3 变更控制

若实施中出现新的建议，必须先分类：

| 类别 | 处理方式 |
|---|---|
| 现有设计的明确要求未接入 | 纳入 Task 14 对应 gate |
| 正式调用链缺陷 | 纳入 Task 14，先写失败测试 |
| 新功能或更严格的新协议 | 记录到 deferred list，不在本计划实施 |
| 与正式封存无关的 legacy 缺陷 | 不修改 |
| 权威设计内部直接矛盾 | 停止实施，形成一条书面 decision record，由用户批准后才可改设计 |

任何人不得以“顺手完善”为理由扩大本计划。

---

## 2. 最近一周问题总结与反思

### 2.1 客观工作量

2026-07-08 至 2026-07-15：

- 共 104 个提交；
- 每日提交数依次为 7、7、11、12、7、27、19、14；
- 最后封存分支相对 `bfee110a` 有 14 个提交；
- 该分支修改 97 个文件，新增约 13,505 行、删除约 1,523 行；
- 最终轻量审查运行 229 个目标测试，全部通过。

这些数据说明问题不是缺少投入，也不是没有测试，而是工作被同时分散到数据、训练、协议、产物、恢复、调度和文档多个层次，集成证明晚于模块实现。

### 2.2 最近审计发现的共同根因

上次审计的多个 BLOCKER/HIGH 可以收敛为三个系统性根因：

1. **双执行栈并存。** 新 sealed/protocol 模块已实现，正式 CLI 仍能进入旧 loader、旧逐 horizon runner 和旧 publisher。
2. **验证层级错位。** 测试证明模块独立正确，却没有证明官方命令真实调用这些模块。
3. **完成定义过早。** “Task 代码和单测完成”被汇报为“正式系统完成”，直到最终审计才检查端到端接线。

因此，新发现多为此前未检查到的集成缺口，而不是每次修复都重新制造了一组完全不同的问题。

### 2.3 具体流程问题及改进措施

| 最近出现的问题 | 为什么原检验没有挡住 | 本计划的改进措施 |
|---|---|---|
| sealed loader 已存在，但 D4–D6 正式 runner 仍读旧/raw 数据 | 测试直接调用新 loader，没有从官方 task command 启动 | 增加 official-entry negative call gate；formal worker 只接受 `SealedDatasetHandle` |
| truth isolation、fitted bundle、blind rollout 都通过单测，但正式 runner 仍处理完整 target/y_true | 测试对象是协议函数，不是正式 child process | 使用官方 CLI 小型端到端测试；修改 evaluator truth 后比较 worker trace semantic digest |
| typed schemas 存在，正式产物仍是 legacy CSV/manifest | schema contract 与 publisher integration 分开测试 | 让 bundle acceptance 只消费 typed identities；旧 publisher 在 formal test 中被 monkeypatch 为必然失败 |
| recovery/fencing/rehydration 库通过，但调度器未使用 | fake scheduler 只验证进程生命周期 | 每个 bundle 必须产生 append-only cell transitions；旧 token 发布必须在端到端测试中被拒绝 |
| `finalize_sealed_run` 存在但官方成功路径未调用 | sealing gate 测试直接调用函数 | shell 官方命令必须从 prepare 运行至 marker；断言 marker 是最后发布对象 |
| preflight 对 D3–D6 repair proof 缺失仍返回 ready | preflight 测试使用满足当前弱合同的 fixture | 当前本地 `unavailable/null` proof 作为失败 fixture，明确要求 blocked |
| 代码 SHA 不包含 ignored sealed 数据 | 本地机器已有数据，fresh checkout 未作为 gate | 新增 sealed data deployment manifest 和 fresh-checkout/install/preflight gate |
| 最终审计才发现跨模块旁路 | Task review 关注各自文件 | 每个集成 gate 完成后进行一次只读 call-graph review；未通过不得进入下一 gate |

### 2.4 对原工作流程的评价

原流程在“规范细化、模块测试、审计证据”方面有效，但不适合一次性完成跨七层的正式封存迁移。问题不是 TDD 或审计本身，而是缺少自第一天起持续运行的 official-entry end-to-end gate。

本计划保留以下有效做法：

- 设计优先、协议显式化；
- 轻量测试和 180 秒保护器；
- append-only 审计和 fail-closed；
- 最终独立只读审计。

同时改变以下做法：

- 从“按模块完成”改为“按可运行的正式切片完成”；
- 从“最后统一集成”改为“每个 gate 都从官方入口验证”；
- 从“测试通过即完成”改为“正向结果 + 旁路调用次数为零 + 故障注入均通过”；
- 从“发现问题继续扩需求”改为“冻结设计，新需求延期”。

### 2.5 每条 BLOCKER 的统一关闭流程

每条审计 BLOCKER/HIGH 都按同一个闭环处理。禁止一次提交同时迁移多个尚未独立通过门禁的调用边。

1. **打开报告点出的官方入口行。** 从该行沿调用链向下追踪，记录实际 import、被调函数、输入对象、产物和异常处理；不能只看函数名称。
2. **定位现有新协议能力。** 优先复用已经实现并通过测试的 sealed loader、schema、truth isolation、fitted bundle、blind rollout、typed artifact、recovery、rehydration 和 acceptance API。
3. **判断迁移类型。** 标记为“直接替换”或“需要薄 orchestration glue”。不得假设每个 BLOCKER 都有一个可以一换一替代的函数。
4. **先写失败测试。** 从官方 task command 或 CLI 进入，令旧函数一旦被调用就抛出异常，并断言新函数收到精确的 typed inputs。
5. **迁移完整调用合同。** 同时迁移输入类型、truth boundary、schema identity、fencing token、artifact identity、失败码和返回码，不能只替换函数名。
6. **删除 formal 分支和 fallback。** legacy 函数可以保留给明确标记的 compatibility CLI，但 formal ownership 下必须不可达；不存在 warning 后继续或 silent fallback。
7. **运行 official dry-run。** 证明任务规划、sealed identity、命令路径、60-bundle 计划和零写入正确。dry-run 不得声称证明 fit/predict/rollout 已执行。
8. **运行 mini sealed integration call。** 使用真实 official entry、mini sealed fixture 和 deterministic fake fitted predictor，实际经过新 orchestration，但不训练真实模型。用 spy/call record 证明目标新函数被调用且 legacy 调用数为 0。
9. **只读复核该 BLOCKER。** 检查 call sites、imports、fallback、返回码和测试证据；通过后才提交并进入下一条。

每条 BLOCKER 的提交记录必须包含：

```text
审计编号：
旧官方入口与行号：
旧调用函数：
新协议函数：
迁移类型：直接替换 / 薄 orchestration glue
删除的 formal fallback：
失败测试：
official dry-run 命令、退出码和零写入证据：
mini integration 命令、退出码和新旧函数 call counts：
残余 legacy 用途：无 / compatibility-only 路径：
```

#### 为什么 dry-run 后仍需要 mini integration

当前 dry-run 的职责是解析 sealed identities、preflight、task matrix 和 scheduler contract，并保证不创建 RUN_ROOT。它不会执行模型 fit、blind rollout、typed publication 或 recovery cell transitions。因此仅看到 dry-run 输出新脚本名称，不能证明运行时没有在新脚本内部回退到 legacy 函数。

本计划把验证分为两层：

- **official dry-run：** 证明控制面、身份和零写入；
- **mini integration call：** 证明数据面实际触发新 loader/schema/truth/rollout/artifact/recovery 调用，仍不进行真实训练。

两者必须同时通过，任何一者不能替代另一者。

### 2.6 BLOCKER 逐条迁移工作表

下表将上次审计发现映射到当前代码和 Task 14 的目标接口。表中的“薄 glue”只允许组合既有能力和传递 typed identities，不得复制协议算法。

| ID | 官方入口中的 legacy 调用 | 已有新协议能力 | 迁移方式 | 必须删除或禁止的 formal fallback | 最小非训练验证 |
|---|---|---|---|---|---|
| B1 sealed-only 输入 | `run_strict_protocol_baseline.build_matrix_tasks()` 生成 `run_full_paper_experiments.py`/`run_d4_experiment.py`/`run_d5_experiment.py`/`run_d6_experiment.py`；D4/D6 调 `load_parquet_source_target()`，D5 调 `load_parquet_source_target_with_diagnostics()` | `load_sealed_target_views()`、`read_sealed_projection()`、sealed manifest/schema/repair sidecars | Gate 1B 新增薄 `SealedDatasetHandle.open()`，Gate 2 使所有 60 tasks 直接调用唯一 bundle worker | formal task 中所有旧 scripts、raw D5 authority、旧 `数据集/固化数据` root、smoke/raw fallback | dry-run 断言 60 commands 只含新 worker且 RUN_ROOT 不存在；mini call 断言 sealed handle call > 0，两个 legacy loader calls = 0 |
| B2 truth-free joint rollout | D1–D3 调旧 `run_no_tl_experiment()` 等实验函数；D4–D6 调 `run_single_entity_experiment()` 并逐 horizon 循环 | `load_sealed_target_views()`、`create_worker_cache()`、`create_evaluator_cache()`、`run_truth_free_fit()`、`fit_formal_method_bundle()`、`run_blind_rollout()`、`join_worker_trace_with_truth()` | 薄 seed-bundle orchestration 按固定顺序组合现有 API | full target dataframe 进入 worker、worker 构造/返回 `y_true`、逐 horizon child fit、h2–h5 feedback | mini call 断言每 method bundle fit = 1、blind rollout = 6、legacy experiment calls = 0；修改 evaluator truth 后 worker semantic digest 不变 |
| B3 exact Schema gate | `run_full_paper_experiments.py` 动态解析 feature cols；D4 runtime feature mismatch 只 warning；`_resolve_model_feature_cols()` 静默 drop；source 数值缺失统一填 0 | `get_predictor_schema()`、`get_knn_schema()`、`audit_future_known_lineage()`、`PredictorFeatureMask` | 在 `SealedDatasetHandle` 与 fit adapter 之间加入 exact ordered-schema assertion；直接使用既有 schema registry | 动态交集、静默 drop、warning 后继续、通用 fill-zero | mini fixture 分别注入 missing/extra/reordered/dtype/lineage/repair drift，均在 fit call count = 0 时 fail closed |
| B4 adoption/repair proof | `adopt_and_seal_d3_d6.py` 调通用 `validate_adopted_pair(source, target)`；随后硬编码 `not_reconstructed_during_adoption`/`unavailable` sidecar；preflight 接受 null/null | `calendarize_and_fill()`、`canonicalize_source_sales()`、`prepare_daily_sequence_pool()` 的完整 180-day validation、`get_predictor_schema()`、`get_knn_schema()`、现有 adoption report/schema helpers | **Gate 1A** 只负责 proof 生产和 producer-level 自检；**Gate 1A-T** 实现 operator-only 完整 root builder；**Gate 1A-X** 由用户执行正式物化；**Gate 1A-V** 由 Codex只读验收并绑定 deployment manifest；**Gate 1B** 独立消费并严格校验 proof | structural-only 大表跳过、unavailable counts、null repair mask/digest、optional proof comparison、原地替换旧 root | Gate 1A 用 mini authority 证明 producer；Gate 1A-T 用 mini fixture 证明 root-level publication 合同；Gate 1A-X 构建新的 immutable content-addressed root；Gate 1A-V 证明旧 root、D1/D2、D3–D6 source/target 和 outputs/runs 不变且 proof identity 三处一致；Gate 1B 删除、置 null 或篡改任一 proof 后 formal preflight blocked；任何训练 call = 0 |
| B5 typed artifacts | 正式 runner 调 `publish_formal_seed_bundle_output_frame()`，其下进入 legacy `publish_formal_cell_frame()` 和 ad hoc manifest | `publish_prediction_artifact()`、`build_worker_manifest()`、`join_worker_trace_with_truth()`、`derive_formal_result_row()`、artifact schema registry | seed-bundle orchestration 直接发布全部 typed artifacts；legacy CSV 仅由 verified evaluated trace 派生 | formal path 中 `publish_formal_seed_bundle_output_frame()`、self-signed fields、未注册字段和未认证 CSV authority | mini call 用 registry 逐个读回 artifacts；typed publisher calls > 0，legacy publisher calls = 0；worker trace schema 中不存在 `y_true` |
| B6 recovery/fencing/binding | shell 以 TSV 表示 task 状态；`_current_fencing_token()` 在发布时读取 mutable state；legacy publisher token 默认 0；bundle files 分步发布 | `RunRecovery.set_cell_state()`、`publish_cell_directory()`、`heartbeat()`、`resume()`、`ArtifactRehydrator.rehydrate()`/`freeze_binding_set()`、`resolve_bound_artifact()` | 薄 lifecycle orchestration 显式传递 attempt-held token；生产结束后冻结完整 binding 再启动 aggregate | token 0、发布时领取新 token、TSV 权威状态、accepted sidecar、binding 外 path resolution | mini supervisor 注入 stale worker/crash/missing artifact；断言旧 token 发布失败、accepted 不重复、rehydration fit/predict = 0、aggregate 只经 binding |
| B7 final sealing 不可达 | shell 成功后只执行 `aggregate`；`aggregate` 停在 `complete_unsealed`；`finalize_sealed_run()` 无官方 CLI 调用点 | `accept_sealed_run_records()`、`finalize_sealed_run()`、`RunRecovery.transition()`、`publish_prediction_artifact()` | 增加 `final-seal` operation 并由 shell 在 aggregate 后唯一调用 | aggregate 后直接退出 0、直接写 marker、调用者提交未认证内存 proofs | 60-bundle mini supervisor 到达 success；故意破坏 trace 后到达 sealed_failed；成功 marker mtime/事件序列为最后写入 |
| B8 server sealed data 不自包含 | `.gitignore` 忽略 `数据集/`，代码 SHA 不提供数据安装或 byte-level authority；没有可替换的单一 legacy 函数 | sealed manifest/validation report/content SHA 与 formal preflight identity | **Gate 1A-T** 实现受审 operator；**Gate 1A-X** 发布新的 immutable content-addressed root；**Gate 1A-V** 验证后提交唯一 deployment manifest；**Gate 1B** 只实现安装/验证、resolver 和 fresh-checkout gate，不得创建第二份 manifest | 未记录的人工复制、本地绝对路径、服务器既有同名文件被默认信任、原地覆盖旧 root | Gate 1A-V 证明 manifest 精确绑定新 root 的正式六数据 bytes且旧 root 不变；Gate 1B 在临时 fresh-install root 按该 manifest 安装 mini artifacts，缺失/替换 bytes 时 preflight blocked，完整 bytes 时 ready；RUN_ROOT 仍不存在 |

#### 每条迁移的通过标准

一条 BLOCKER 只有在下列五项同时成立时才能关闭：

1. 报告指出的旧 call site 在 formal ownership 下不可达；
2. 新函数由 official-entry mini call 实际调用，而不是只被 import；
3. 输入、身份、失败语义和产物一起迁移，不存在中间 legacy adapter 把合同降级；
4. official dry-run 零写入，mini integration 不运行真实训练；
5. 对应负向测试证明删除 fallback 后会 fail closed，而不是换一个位置继续 fallback。

---

## 3. 文件与职责边界

### 新建文件

- `tools/operations/materialize_d1_d6_sealed_authority.py`
  - Gate 1A-T operator-only CLI；在 private build root 中组装完整 D1–D6、验证 content set，并以一次 root-level rename 发布新的 immutable root。
  - 不原地修改旧 sealed root，不提供 raw/legacy/smoke fallback，也不由 Codex执行正式物化。
- `tests/test_materialize_d1_d6_sealed_authority.py`
  - Gate 1A-T mini fixture RED/GREEN；验证失败保留、proof/identity、content-set digest、single-root publication 和 strict dry-run。
- `configs/sealed_deployments/d1_d6_sealed_v1.json`
  - Gate 1A-V 唯一 Git 跟踪产物；精确绑定新 immutable deployment root 中正式 D1–D6 sealed artifacts 的规范相对路径、size 和 SHA-256。
  - 只是薄部署索引，不包含 run artifact binding、attempt、fencing、rehydration、raw/legacy authority 或服务器绝对路径。
- `scripts/run_formal_seed_bundle.py`
  - 唯一正式 child-process CLI；只接受 dataset、mode、seed、run root、attempt id 和 fencing token。
  - 不接受 `--horizon`、raw path、legacy parquet root 或 smoke fallback。
- `src/experiment/formal_seed_bundle.py`
  - 实现一个 seed bundle 的纯编排；组织六种方法、h1–h5 rollout、worker/evaluator trace 和 bundle manifest。
  - 不拥有全局调度、run state transition 或最终聚合。
- `src/utils/formal_sealed_inputs.py`
  - 将 manifest、binding、sealed parquet、schema、lineage、repair proof 和 source window 组合为只读 `SealedDatasetHandle`。
  - 这是 formal worker 唯一的数据入口。
- `tests/fixtures/formal_sealed_mini/`
  - 小型确定性 D1–D6 sealed fixture，保持真实 schema、窗口身份和 sidecar 结构，但不进行真实训练。
- `tests/fixtures/fake_fitted_formal_methods.py`
  - 确定性的 truth-free fitted predictors；用于验证 rollout、trace、恢复和 sealing，不替代 adapter 单测。
- `tests/test_formal_no_legacy_calls.py`
  - 从官方 task command 验证旧 loader、旧 runner、旧 publisher 调用次数为 0。
- `tests/test_formal_seed_bundle_integration.py`
  - 验证一个 bundle 的 sealed-input 到 fenced typed publication。
- `tests/test_formal_end_to_end_sealing.py`
  - 验证 60 个轻量 bundles 从官方 supervisor 到最终 marker。
- `tests/test_formal_server_data_contract.py`
  - 验证 ignored sealed 数据的独立部署 manifest、内容摘要和 fresh-install preflight。

### 修改文件

- `scripts/run_strict_protocol_baseline.py`
  - 60 个 task 全部改为调用 `scripts/run_formal_seed_bundle.py`。
- `scripts/adopt_and_seal_d3_d6.py`
  - 仅在 Gate 1A 接入现有 `canonicalize_source_sales()` 并发布完整 D3–D6 source-sales repair proof；不进行 raw rebuild，不实现新的 repair 协议。
- `scripts/run_unified_d1_d6.py`
  - 保存 attempt token，而不是发布时重新读取当前 token；增加 binding freeze、cell lifecycle 和 `final-seal` operation。
- `scripts/parallel_mode_runner.sh`
  - 启动 heartbeat；把 attempt id/token 传入 child；成功聚合后调用 `final-seal`；不再把 TSV 作为权威状态。
- `scripts/validate_d1_d6_protocol_inputs.py`
  - 对 D3–D6 adoption/canonicalization proof 缺失 fail closed，并验证部署 manifest。
- `tests/test_hybrid_sealed_builder.py`
  - Gate 1A mini adoption、canonicalization 调用、proof 完整性、稳定性和 source 内容保持测试。
- `tests/test_formal_entry_preflight.py`
  - Gate 1A 只增加与当前 preflight 能力兼容的 repository/fixture 回归测试，也可保持不变；Gate 1B 负责完整 proof 删除/null/篡改的 formal preflight blocked 测试。
- `scripts/run_full_paper_experiments.py`、`scripts/run_d4_experiment.py`、`scripts/run_d5_experiment.py`、`scripts/run_d6_experiment.py`
  - 保留 legacy 能力；formal flag 必须拒绝或转交新 worker，不能继续内部执行旧路径。
- `src/utils/parquet_data_loader.py`
  - 保留 legacy loader；formal code 不再导入 `load_parquet_source_target*`。
- `src/utils/entity_experiment.py`
  - 保留 method adapter；新 worker 只能从 exact schema gate 调用 `fit_formal_method_bundle`。
- `src/utils/run_recovery.py`
  - 仅补充正式编排所需的小型接口或验证；不重写已通过测试的状态机。
- `src/utils/result_acceptance.py`
  - final gate 从冻结 binding 解析 typed artifacts，拒绝 legacy/self-signed paths。
- `README.md`
  - 仅在端到端 gate 完成后更新正式命令和服务器数据安装步骤。

### 禁止的结构变化

- 不再创建第二个 recovery/state-machine 实现。
- 不再为 D1–D3 和 D4–D6 分别创建正式 orchestrator。
- 不复制 artifact schema 或 blind rollout 逻辑到 scripts。
- 不删除 legacy runner；通过 formal boundary 和测试隔离它们。
- 不在 shell 中重新实现 JSON identity、digest 或 acceptance。

---

## 4. Task 14 实施门禁

以下均属于同一个 Task 14。每一 Gate 都必须形成独立可审查、可拒绝的提交；前一 Gate 未通过不得进入下一 Gate。

### Gate 0：冻结设计并建立审计映射

**Files:**
- Create: `tests/test_formal_design_freeze.py`
- Modify: `docs/superpowers/plans/2026-07-15-d1-d6-formal-execution-chain-integration.md`

**Interfaces:**
- Consumes: 两份现有权威文档。
- Produces: 设计 digest gate、Task 1–13 到 Gate 1A、Gate 1A-T、Gate 1A-X、Gate 1A-V、Gate 1B、Gate 2–8 的覆盖映射。

- [ ] **Step 1: 写设计冻结测试**

```python
from hashlib import sha256
from pathlib import Path


def test_authoritative_design_is_frozen():
    path = Path("docs/superpowers/specs/2026-07-15-d1-d6-experiment-sealing-design.md")
    assert sha256(path.read_bytes()).hexdigest() == (
        "914ab6e4b3ac2eca7d2bb1c7cc2811a75c905995269b15b3300b0038f7343f6d"
    )
```

- [ ] **Step 2: 运行测试并确认通过**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_formal_design_freeze.py -q -p no:cacheprovider \
--basetemp /tmp/d1-d6-task14-design-freeze
```

Expected: `1 passed`。

- [ ] **Step 3: 将本计划第 9 节覆盖矩阵与冻结 digest 一同提交**

```bash
git add docs/superpowers/plans/2026-07-15-d1-d6-formal-execution-chain-integration.md tests/test_formal_design_freeze.py
git commit -m "test: freeze D1-D6 sealing design for Task 14"
```

**Gate 0 acceptance:** 现有设计 bytes 未改变，所有后续变更可映射到现有要求；没有新协议条目。

### Gate 1A：补齐 D3–D6 adoption repair proof

Gate 1A 是 Task 2 的漏接调用点修复，不是新的封存协议。D3–D6 继续使用
`adopted_solidified` provenance，不进行 raw rebuild，不预先假定或人为改写
source/target parquet bytes。Gate 1A 必须先于 Gate 1A-T，Gate 1A-T 必须先于用户执行的 Gate 1A-X，
Gate 1A-X 必须先于 Codex只读验收的 Gate 1A-V，Gate 1A-V 必须先于 Gate 1B；Gate 2 只有在 Gate 1A、
Gate 1A-T、Gate 1A-X、Gate 1A-V 和 Gate 1B 均通过后才能开始。Gate 1B 不得直接消费未经
Gate 1A-V 验收和 deployment manifest 绑定的 producer 输出。

Gate 1A 的责任边界固定为 **proof 生产和 producer-level 自检**。本 Gate 不修改正式
preflight 生产实现，不要求正式 preflight 已能拒绝每一种 proof tamper；完整
tamper fail-closed 属于 Gate 1B 的独立 proof-consumer 提交。

2026-07-16 的 D3–D6 全量只读 replay 已在 180 秒保护器下返回 124。该命令不得重试、
拆分、简化或续跑。Gate 1A acceptance 不包含正式 D3–D6 全量 parquet replay，只使用
mini adopted fixtures 证明真实调用链、proof 闭合和确定性。正式 sealed root 的 byte identity
沿用此前已记录的只读审计证据；Gate 1A 不得声称它重新验证了全量 D3–D6 内容。

**Files:**
- Modify: `scripts/adopt_and_seal_d3_d6.py`
- Modify: `tests/test_hybrid_sealed_builder.py`
- Modify: `tests/test_formal_entry_preflight.py`

**Interfaces:**
- Consumes: 现有 `calendarize_and_fill()`、`canonicalize_source_sales()`、`validate_adopted_pair()`、现有 dataset manifest、validation/adoption report、schema 和 provenance sidecars。
- Produces: 现有 sealed dataset manifest、`source_sales_canonicalization.json` 和 adoption report 中完整且确定的共同 source-sales repair proof identity；不产生新 schema、binding、resolver、生产验证器或运行时 authority。

- [ ] **Step 1: 写 Gate 1A 失败测试，证明 mini adoption 真实调用现有 calendarization 和 canonicalization**

测试必须使用 mini adopted parent parquet 和 `tmp_path`，通过 `scripts/adopt_and_seal_d3_d6.py` 的真实 adoption 边界先进入现有 `calendarize_and_fill()`，再将其真实生成的 `calendar_row_missing_mask` 交给 `canonicalize_source_sales()`。测试可用 spy 记录既有函数被调用，但不得复制 canonicalization 实现、从最终 sealed source 反推 mask 或硬编码零 repair。

- [ ] **Step 2: 写 Gate 1A proof 合同测试**

测试必须断言：

```text
repair_reason_counts
rows_examined
affected_rows
repair_mask_sha256
affected_date_digest
status != not_reconstructed_during_adoption
status != unavailable
manifest repair identity == repair sidecar identity == adoption report repair identity
```

counts 必须闭合为 `original_nan`、`original_negative`、`calendar_row_missing` 三个现有 reason；三项 counts 总和必须等于 `len(affected_rows)`，且 `len(affected_rows) <= rows_examined`。digest 必须保留现有 SHA-256 格式并由真实 calendarization/canonicalization 输出计算，不得硬编码空集合 digest 或零 counts；success status 只能由成功执行路径产生，不得在未执行 canonicalization 时预填。

- [ ] **Step 3: 写 Gate 1A 确定性和 source 保持测试**

对无 NaN、无负值、无 infinity 且无缺失日期的 mini parent，断言 adoption 重放后的 source 逻辑内容与 parent 相同；不得因为补 proof 而改写 bytes。重复 adoption 到两个独立临时输出目录时，proof identity 必须相同。若 mini fixture 的真实 replay 证明 source 内容必须变化，才允许通过现有 canonicalization 结果发布变化后的 source；不得预先假定或人为改写。本 Gate 不对正式 D3–D6 parquet 做全量 replay。

- [ ] **Step 4: 修改 adoption 发布边界，复用现有实现**

在 `scripts/adopt_and_seal_d3_d6.py` 中按固定 source key/window 调用现有 `calendarize_and_fill()` 和 `canonicalize_source_sales()`，复用现有 adopt validation、manifest 和 audit helpers。完整 proof 必须进入现有 manifest、sidecar 和 adoption report 的共同 identity；producer 必须在发布前自检 counts/rows/digests/identity 闭合。不得新建 canonicalization 算法、repair 协议或第二套生产验证器，不得实现 `SealedDatasetHandle` 或 deployment resolver，不得修改正式运行链。

- [ ] **Step 5: 写 Gate 1A producer-level proof 破坏测试**

在 mini adopted output 上删除、置 null 或篡改任一 producer proof 字段，通过现有可复用 proof identity/一致性 helper 证明 producer-level contract 失败。若当前没有可复用 helper，测试可直接对公开 sidecar、manifest 和 adoption report 字段做精确交叉断言，不得为此新增第二套生产验证器。`tests/test_formal_entry_preflight.py` 在 Gate 1A 只能增加与当前 preflight 能力兼容的 repository/fixture 回归测试，也可保持不变；不得放宽 preflight，不得修改 `scripts/validate_d1_d6_protocol_inputs.py`。所有测试只写 pytest `tmp_path`，不得修改正式 sealed root，不得调用训练、fit 或 predict。

- [ ] **Step 6: 运行 Gate 1A 测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_hybrid_sealed_builder.py \
tests/test_formal_entry_preflight.py -q -p no:cacheprovider \
--basetemp /tmp/d1-d6-task14-adoption-proof
```

Expected: 全部通过；mini adoption 真实调用现有 calendarization 和 canonicalization，proof 完整、闭合、稳定，manifest/sidecar/adoption report identity 一致，producer-level 破坏被检出；不做正式 D3–D6 全量 replay，不运行训练，不写正式 sealed root。

- [ ] **Step 7: 提交 Gate 1A**

```bash
git add scripts/adopt_and_seal_d3_d6.py tests/test_hybrid_sealed_builder.py tests/test_formal_entry_preflight.py
git commit -m "fix: publish complete D3-D6 adoption repair proofs"
```

**Gate 1A acceptance:** D3–D6 adoption producer 在 mini fixtures 上真实使用现有 calendarization/canonicalization；`calendar_row_missing_mask` 来自真实 calendarization；proof 完整、闭合且确定；manifest、repair sidecar 和 adoption report 身份一致；producer-level 破坏被检出；无修复 mini source 的逻辑内容没有无依据变化；没有正式全量 replay、新协议、第二套实现或正式 preflight 生产变更。

### Gate 1A-T：实现 operator-only 物化工具

Gate 1A-T 只实现并验证一个 operator-only 工具，不读取或处理正式 D3–D6 大文件，不创建正式
deployment root，也不创建正式 deployment manifest。它把代码实施与长耗时数据操作分开：Codex
负责工具、mini fixture、dry-run 和轻量验证；正式物化只能由 Gate 1A-X 的用户 Terminal/tmux
操作执行。

2026-07-16 已返回 124 的旧命令 `python scripts/adopt_and_seal_d3_d6.py --dataset all ...` 永久禁止，
不得由 Codex或用户重试、续跑、拆分复用或作为手工命令执行。既有
`.d1_d6_sealed_v1.gate1a-p-staging-a` 不是可信 authority；Gate 1A-T、Gate 1A-X 和 Gate 1A-V 均不得
清理、读取、散列、选择或复用其内部内容。

**Files:**
- Create: `tools/operations/materialize_d1_d6_sealed_authority.py`
- Create: `tests/test_materialize_d1_d6_sealed_authority.py`
- Modify: `docs/superpowers/plans/2026-07-15-d1-d6-formal-execution-chain-integration.md` only if implementation reveals a plan-only clarification that does not change the frozen design

**Forbidden modifications:**
- `scripts/adopt_and_seal_d3_d6.py`
- `src/data_processing/sealed_daily.py` 中的 canonicalization 或 producer 路径
- `scripts/validate_d1_d6_protocol_inputs.py`
- `scripts/run_unified_d1_d6.py`
- `数据集/固化数据/d1_d6_sealed_v1/`
- Gate 1B files

**Interfaces:**
- Consumes: Gate 1A producer `scripts/adopt_and_seal_d3_d6.py@c4a905cd`、现有旧 sealed root、parent root、同文件系统 private build root 和现有 `src.utils.run_artifacts.sha256_file()`。
- Produces: 经 mini fixtures 验证的 operator CLI；成功正式执行时输出 machine-readable execution report、deployment manifest candidate 和新 immutable sealed root，但 Gate 1A-T 本身不产生这些正式数据。
- Does not produce: 新 canonicalization、raw rebuild、legacy/smoke fallback、formal preflight、resolver、worker、runner、run artifact、正式 D1–D6 数据或 Git-tracked deployment manifest。

operator CLI 固定为：

```text
python tools/operations/materialize_d1_d6_sealed_authority.py
  --old-sealed-root PATH
  --parent-root PATH
  --private-build-root PATH
  --final-deployment-parent PATH
  --report-output PATH
  --manifest-candidate-output PATH
  [--dry-run]
```

不得增加 raw rebuild、legacy、smoke、fallback、entity、batch、日期区间、文件片段或 timeout 参数。
`--private-build-root` 和计算得到的 final path 在启动时都必须不存在；private build root 必须与
`--final-deployment-parent` 位于同一文件系统。旧 sealed root 在整个执行期间只读且不得移动、删除、
改名、覆盖或写入。

- [ ] **Step 1: 写 mini fixture 失败测试**

`tests/test_materialize_d1_d6_sealed_authority.py` 必须先覆盖以下失败合同：

```text
test_cli_help_lists_only_fixed_arguments
test_cli_rejects_existing_private_or_final_root
test_cli_rejects_cross_device_private_build_root
test_failure_keeps_old_root_unchanged_and_publishes_no_final_root
test_failure_marks_private_root_non_authoritative
test_dataset_is_never_split_below_dataset_level
test_source_or_target_identity_drift_blocks_publication
test_incomplete_or_mismatched_repair_proof_blocks_publication
test_manifest_candidate_rejects_extra_missing_absolute_parent_or_symlink_paths
```

fixtures 只能使用 `tmp_path` 中的 tiny D1–D6 artifacts，并通过 monkeypatch/spies 替代大文件 producer
工作量；不得 mock 掉 operator 自身的 inventory、proof validation、content-set digest 或 root-level
publication 边界。

- [ ] **Step 2: 运行 RED 并确认预期失败**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_materialize_d1_d6_sealed_authority.py -q \
-p no:cacheprovider --basetemp /tmp/d1-d6-task14-gate1a-t-red
```

Expected: 因 operator 模块尚不存在或固定接口尚未实现而失败；不得读取正式 sealed root。

- [ ] **Step 3: 实现固定 operator 工作流**

工具必须按以下顺序执行：

1. lstat 所有输入/输出根，拒绝 symlink、重叠路径、现有 private/final path、现有 report/manifest-candidate output 和跨设备 publication；
2. 冻结旧 sealed root 与 `outputs/runs/` 的完整 inventory；
3. 创建 private build root，并写入 `NON_AUTHORITATIVE.json`；
4. 将旧 root 的 D1/D2 规定 artifact 集合逐字节复制到 `dataset1/`、`dataset2/`，逐文件 fsync；
5. 按 D3、D4、D5、D6 顺序分别以 `subprocess.run()` 调用现有 CLI：`[sys.executable, "scripts/adopt_and_seal_d3_d6.py", "--dataset", "dN", "--parent-root", parent_root, "--output-dir", private_build_root]`；使用参数数组且 `shell=False`；
6. 每个 dataset 独立记录 started_at、finished_at、returncode、status、stdout/stderr log path、error 和 validation result；`timeout=None`，不设置 180 秒内部 timeout，也不进一步拆分；任一 returncode 非零立即停止且不运行后续 dataset；
7. 每个 dataset 完成后立即验证规定 artifact 集合、source/target size/SHA/bytes、repair proof 闭合、digest 格式和三处 proof identity；
8. 全部 D1–D6 完成后，将全部规定 artifacts 表示为按 `relative_path` 排序的 entries；每项精确含 `relative_path`、`size_bytes`、lowercase 64-hex `sha256`，以 UTF-8、sorted keys、compact separators、`ensure_ascii=false`、`allow_nan=false` 且无尾随换行编码；`content_set_digest` 精确定义为这些 canonical list bytes 的 lowercase SHA-256；
9. final root 名称固定为 `d1_d6_sealed_v1_deploy_<content_set_digest前16位>`；
10. 在各自输出目录的私有临时文件中生成完整新 root inventory、旧/新 source-target 对比、每 dataset proof identity、manifest candidate 和 execution report candidate；临时输出必须 fsync，最终输出路径仍不存在；
11. 重新验证旧 root inventory 和 `outputs/runs/` inventory 完全不变；
12. 删除 private root 中的 `NON_AUTHORITATIVE.json`，fsync 全部文件、dataset directories、private root 和 parent；
13. 以一次 `os.replace(private_build_root, final_root)` 发布整个新 root；final path 已存在时 fail closed；
14. fsync final deployment parent，重读 final root 并核对 content-set digest；
15. 只有发布和发布后复核全部成功，才将成功 manifest candidate 和 `status=success` execution report 的临时文件原子 rename 到用户指定路径并 fsync 各自父目录；
16. 若 final rename 后的复核或外部输出发布失败，必须删除任何已出现的 manifest candidate，把新 final root 以一次 rename 退回原 private build path、恢复 `NON_AUTHORITATIVE.json` 并 fsync parent，然后写 `status=failed` execution report；不得保留一个未报告成功的 final root。

规定 artifact 集合为：D1/D2 各含 11 项
`source.parquet`、`target.parquet`、`manifest.json`、`validation_report.json`、
`source_schema.json`、`target_schema.json`、`predictor_schema.json`、`knn_schema.json`、
`calendarization_audit.json`、`source_sales_canonicalization.json`、`provenance.json`；D3–D6 另各含
`adopt_validation_report.json`。不得发布额外、缺失或 symlink artifact。

deployment manifest candidate 顶层字段精确为 `manifest_version`、`sealed_root_version`、
`deployment_root`、`content_set_digest` 和 `datasets`；`deployment_root` 只能是新 root basename，禁止绝对
路径或父目录；`content_set_digest` 使用上述 lowercase 64-hex。每个 artifact entry 只含
`logical_role`、`path`、`size_bytes` 和 `sha256`，其中 path 相对于新 root，必须是规范 POSIX relative
path。manifest candidate 使用 UTF-8、sorted keys、compact separators、`ensure_ascii=false`、
`allow_nan=false` 和唯一尾随换行。

每个 D3–D6 proof 必须满足：status 不为 null/`unavailable`/
`not_reconstructed_during_adoption`；reason keys 精确为 `original_nan`、`original_negative`、
`calendar_row_missing`；`sum(counts) == len(affected_rows)`；`len(affected_rows) <= rows_examined`；
repair mask 为 lowercase 64-hex；affected-date digest 为 `sha256:<lowercase 64-hex>`；sidecar、manifest
和 adoption report proof 完全相同，manifest mask/count identities 与 proof 相同。

任一异常必须阻止 final rename 和成功 manifest：若 private root 已创建，保留它并确保
`NON_AUTHORITATIVE.json` 存在；execution report 记录 `status=failed`、失败阶段、dataset、异常类型和
稳定错误码；manifest-candidate output 不得存在；进程返回非零。工具不得自动重试 dataset。

- [ ] **Step 4: 实现 `--help` 和严格 `--dry-run`**

`--dry-run` 只解析参数、规范化路径、核对路径互不重叠及计划动作，输出 canonical planned-operation
JSON；不得遍历或散列正式数据、调用 producer、创建 private/final root、report、manifest candidate
或 `outputs/runs/` 内容。退出 0 只证明 CLI/路径计划有效，不证明数据可物化。

- [ ] **Step 5: 运行 GREEN、dry-run、AST/compile 和设计冻结测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_materialize_d1_d6_sealed_authority.py -q \
-p no:cacheprovider --basetemp /tmp/d1-d6-task14-gate1a-t-green

PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python tools/operations/materialize_d1_d6_sealed_authority.py --dry-run \
--old-sealed-root /tmp/gate1a-t-dry-run/old \
--parent-root /tmp/gate1a-t-dry-run/parents \
--private-build-root /tmp/gate1a-t-dry-run/build \
--final-deployment-parent /tmp/gate1a-t-dry-run/deployments \
--report-output /tmp/gate1a-t-dry-run/report.json \
--manifest-candidate-output /tmp/gate1a-t-dry-run/manifest-candidate.json

PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -c 'from pathlib import Path; paths=(Path("tools/operations/materialize_d1_d6_sealed_authority.py"), Path("tests/test_materialize_d1_d6_sealed_authority.py")); [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in paths]'

PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_formal_design_freeze.py -q -p no:cacheprovider \
--basetemp /tmp/d1-d6-task14-gate1a-t-design-freeze

git diff --check
```

Expected: GREEN tests 全部通过；dry-run 退出 0 且不创建列出的任何路径；compile 和设计冻结测试通过；
tracked diff 仅为 operator、operator tests 和已批准的计划澄清。Codex 不得运行非 `--dry-run` 的正式
operator 命令。

- [ ] **Step 6: 提交 Gate 1A-T**

```bash
git add tools/operations/materialize_d1_d6_sealed_authority.py \
tests/test_materialize_d1_d6_sealed_authority.py \
docs/superpowers/plans/2026-07-15-d1-d6-formal-execution-chain-integration.md
git diff --cached --name-only
git commit -m "feat: add operator tool for sealed authority materialization"
```

**Gate 1A-T acceptance:** operator CLI 的 mini RED/GREEN、失败保留、proof/identity、content-set digest、
single-root atomic publication 和 dry-run 合同全部通过；没有读取正式大数据、正式 producer 调用、训练、
正式 root/manifest 输出或 Gate 1B 修改。

### Gate 1A-X：用户执行正式物化

Gate 1A-X 是唯一允许处理正式 D1–D6 大文件的 Gate。它只能由用户在 Terminal 或 tmux 中执行已审核、
已提交的 Gate 1A-T operator 工具。Codex 不执行、代跑、续跑、拆分、重试或 babysit 该命令，也不把它
包装在 `codex_timeout.py` 中。旧 `.d1_d6_sealed_v1.gate1a-p-staging-a` 继续视为非可信失败目录；本 Gate
不清理、不读取、不复用它。

**User-operated mutable paths:**
- Create private, Git ignored: `数据集/固化数据/.d1_d6_sealed_v1.gate1a-x-build`
- Publish immutable, Git ignored: `数据集/固化数据/d1_d6_sealed_v1_deploy_<content_set_digest前16位>`
- Create reports outside Git: `/tmp/d1-d6-task14-gate1a-x/`
- Read only: `数据集/固化数据/d1_d6_sealed_v1/`

- [ ] **Step 1: 用户在 tmux 中执行唯一正式命令**

先进入 tmux session，再复制执行以下完整 zsh 命令块：

```bash
mkdir -p /tmp/d1-d6-task14-gate1a-x
set -o pipefail
python tools/operations/materialize_d1_d6_sealed_authority.py \
  --old-sealed-root 数据集/固化数据/d1_d6_sealed_v1 \
  --parent-root 数据集/固化数据 \
  --private-build-root 数据集/固化数据/.d1_d6_sealed_v1.gate1a-x-build \
  --final-deployment-parent 数据集/固化数据 \
  --report-output /tmp/d1-d6-task14-gate1a-x/execution-report.json \
  --manifest-candidate-output /tmp/d1-d6-task14-gate1a-x/deployment-manifest-candidate.json \
  2>&1 | tee /tmp/d1-d6-task14-gate1a-x/materialize.log
status=${pipestatus[1]}
printf '%s\n' "$status" > /tmp/d1-d6-task14-gate1a-x/exit-code.txt
exit "$status"
```

该命令明确不使用 180 秒 wrapper。建议用户先运行 `tmux new -s d1-d6-gate1a-x`，在 session 内执行上述
命令块。stdout/stderr、独立 exit code、execution report 和 manifest candidate 均写到 `/tmp`；命令
不得修改 Git 跟踪文件、旧 sealed root 或 `outputs/runs/`。

- [ ] **Step 2: 用户只报告完成状态，不人工编辑候选文件**

用户执行结束后只向 Codex报告命令已结束；不得手工修补 execution report、manifest candidate、
private build root 或 final root。非零退出时 Gate 立即为 `BLOCKED AT GATE 1A-X`；不得让 Codex或用户
自动重试，必须先只读审查失败报告并形成新的明确授权。

**Gate 1A-X acceptance:** 用户命令自然退出 0；旧 root 未修改；工具发布一个新且此前不存在的 immutable
root，并写出 execution report、manifest candidate、完整 inventory、旧/新 source-target 比较和每
dataset proof identity。Gate 1A-X 的结果尚未成为 Git-tracked deployment authority，必须进入 Gate 1A-V。

### Gate 1A-V：Codex 只读验收正式物化结果

Gate 1A-V 只读消费用户执行结果。除最终复制并提交已验证的 manifest candidate 外，Codex 不修改新旧
sealed root、execution report 或任何数据文件，不运行 producer，也不清理 private/失败 staging。

**Files:**
- Create and commit after all checks pass: `configs/sealed_deployments/d1_d6_sealed_v1.json`
- Read only: `/tmp/d1-d6-task14-gate1a-x/exit-code.txt`
- Read only: `/tmp/d1-d6-task14-gate1a-x/execution-report.json`
- Read only: `/tmp/d1-d6-task14-gate1a-x/deployment-manifest-candidate.json`
- Read only: old and newly published sealed roots

- [ ] **Step 1: 验收 operator 退出和 report 状态**

确认 exit code 精确为 0；execution report 为 canonical JSON 且 `status=success`；报告中的 Gate 1A-T
代码 SHA、旧 root identity、final root name、content-set digest、开始/结束时间和 D3–D6 四个 dataset
独立 started_at、finished_at、returncode=0、status 和 validation result 完整。任何缺失、非零、failed、
unavailable 或代码 SHA 不匹配均立即停止。

- [ ] **Step 2: 重算新旧 root byte-level inventory**

使用 `src.utils.run_artifacts.sha256_file()` 对新旧 roots 的规定 artifact 集合按 POSIX relative path
重算 size/SHA-256，并与 execution report inventory 精确比较。断言：旧 root inventory 与 Gate 1A-X
前记录完全相同；D1/D2 全部 bytes 相同；D3–D6 source/target bytes 相同；新 root 无额外、缺失、
symlink、绝对路径、`..`、根外引用或重复 normalized path；final root 名称前缀与 content-set digest
前 16 位一致。

- [ ] **Step 3: 重验 D3–D6 proof identity 和 manifest candidate coverage**

逐 dataset 重验完整 proof status、三项 reason counts、闭合关系、两个 digest 格式、sidecar/manifest/
adoption report identity 和 manifest mask/count identity。manifest candidate 必须精确绑定 D1/D2 各 11
项、D3–D6 各 12 项，共 70 个 artifacts；每项只含 `logical_role`、规范 POSIX relative `path`、
`size_bytes`、lowercase SHA-256。所有 size/SHA 必须与新 root 实际 bytes 一致；manifest 不得含 raw、
legacy、KNN root、attempt、fencing、binding、rehydration、mutable state 或本机绝对路径。

- [ ] **Step 4: 验证输出保护和 repository preflight**

确认 `outputs/runs/` inventory 与 Gate 1A-X 前完全相同。repository preflight 必须针对 manifest candidate
声明的新 deployment root 返回 ready；不得回退到旧 root。所有 Codex Python 验收命令仍通过 180 秒
wrapper，任何一个返回 124 时立即停止 Gate 1A-V，不拆分或重试。

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -c 'import json; from pathlib import Path; from scripts.validate_d1_d6_protocol_inputs import validate_formal_entry_preflight; operation=json.loads(Path("/tmp/d1-d6-task14-gate1a-x/execution-report.json").read_text(encoding="utf-8")); result=validate_formal_entry_preflight(Path(operation["final_root"]), run_id="gate1a-v-read-only"); print(json.dumps(result, ensure_ascii=False, sort_keys=True)); assert result["status"] == "ready", result'

git diff --check
git status --short
git diff --name-only
```

- [ ] **Step 5: 复制已验证候选并重验 canonical bytes**

只有 Steps 1–4 全部通过后，才将候选 bytes 原样复制为
`configs/sealed_deployments/d1_d6_sealed_v1.json`。复制后重算两者 SHA-256，必须完全相同；JSON 使用
UTF-8、sorted keys、compact deterministic separators、`ensure_ascii=false`、`allow_nan=false` 和唯一
尾随换行。manifest digest 是最终文件 bytes 的 SHA-256，不在 manifest 内自引用。

- [ ] **Step 6: 提交 Gate 1A-V deployment authority**

```bash
git add configs/sealed_deployments/d1_d6_sealed_v1.json
git diff --cached --name-only
git commit -m "data: bind materialized D1-D6 sealed deployment"
```

Expected: staged/committed path 精确只有 `configs/sealed_deployments/d1_d6_sealed_v1.json`。

**Gate 1A-V acceptance:** 用户正式物化 exit 0；execution report success；新 immutable root 完整发布；
旧 root inventory、D1/D2 全部 bytes、D3–D6 source/target bytes 和 `outputs/runs/` 均不变；D3–D6 proof
完整且三处 identity 一致；manifest candidate 精确绑定新 root 的 70 个规定 artifacts；repository
preflight ready；唯一 Git-tracked deployment manifest 已按候选原 bytes 提交。Gate 1B 只有在本 Gate
为 `GATE PASSED` 后才能开始。

### Gate 1B：建立 sealed-only 数据句柄和部署合同

**Files:**
- Create: `src/utils/formal_sealed_inputs.py`
- Create: `tests/test_formal_server_data_contract.py`
- Modify: `scripts/validate_d1_d6_protocol_inputs.py`
- Modify: `scripts/run_unified_d1_d6.py`

**Interfaces:**
- Consumes: Gate 1A-V 已验收的新 immutable sealed root、已提交的 `configs/sealed_deployments/d1_d6_sealed_v1.json`、`load_sealed_target_views()`、sealed manifest、validation report、schema/lineage registry、现有 input identity/digest 工具和 frozen run binding 边界。
- Produces: 对已物化 proof 的严格 formal preflight enforcement，以及 `SealedDatasetHandle.open(dataset_id, deployment_manifest_path, run_plan_input_identity) -> SealedDatasetHandle`；handle 暴露 truth-isolated views、source frame 和不可变 identities，不暴露 raw/legacy path。Gate 1B 只消费 Gate 1A-V authority，不生成、修补或推断 repair proof，也不得创建第二份 deployment manifest或回退到旧 root。

- [ ] **Step 1: 写失败测试，严格拒绝缺失、unavailable、null 或篡改的 repair proof**

```python
def test_formal_preflight_blocks_unavailable_source_repair_proof(sealed_fixture):
    sealed_fixture.write_repair(
        dataset_id="D4",
        status="not_reconstructed_during_adoption",
        counts="unavailable",
        repair_mask_sha256=None,
    )
    report = sealed_fixture.preflight()
    assert report["status"] == "blocked"
    assert "SOURCE_SALES_REPAIR_PROOF_MISSING" in report["failure_codes"]
```

负向 fixture 必须分别删除、置 null 或篡改 success status、三项
`repair_reason_counts`、`rows_examined`、`affected_rows`、`repair_mask_sha256` 和
`affected_date_digest`。formal preflight 必须校验三项 counts 总和等于
`len(affected_rows)`、`len(affected_rows) <= rows_examined`、两个 digest 形式/身份，以及
sidecar、manifest 和 deployment identity 一致。缺失或 null 使用稳定失败码
`SOURCE_SALES_REPAIR_PROOF_MISSING`；不闭合或 identity mismatch 使用本计划固定的对应 mismatch
失败码。任一字段损坏都必须使全局 preflight 返回 `blocked`。

- [ ] **Step 2: 写失败测试，证明 fresh install 必须按内容摘要部署**

```python
def test_server_install_rejects_missing_or_changed_sealed_bytes(sealed_fixture):
    installation = sealed_fixture.install_from_manifest()
    installation.dataset(6).source_path.write_bytes(b"changed")
    report = installation.preflight()
    assert report["status"] == "blocked"
    assert "ARTIFACT_BYTES_MISMATCH" in report["failure_codes"]
```

- [ ] **Step 3: 实现完整 repair-proof enforcement、`SealedDatasetHandle` 和 fail-closed preflight**

最小公开接口固定为：

```python
@dataclass(frozen=True)
class SealedDatasetHandle:
    dataset_id: str
    source_path: Path
    target_views: TargetViews
    predictor_schema: PredictorFeatureSchema
    knn_schema: KnnFeatureSchema
    identities: Mapping[str, str]
    source_sales_repair: Mapping[str, object]
    future_known_lineage: tuple[FutureKnownLineage, ...]
```

构造入口固定为 `SealedDatasetHandle.open(*, dataset_id: str, deployment_manifest_path: Path, run_plan_input_identity: Mapping[str, object]) -> SealedDatasetHandle`，不得增加 raw root、legacy parquet directory 或 truth dataframe 参数。sealed 输入由部署 manifest 和 run plan identity 绑定；`artifact_binding_set.json` 只负责当前 run 内已经发布或再水化的产物权威。

实现必须在打开后重新计算内容 SHA，拒绝 symlink、`..`、绝对外部路径和 binding 外文件。

- [ ] **Step 4: 从 formal input identity 删除 D5 raw authority 和旧 KNN roots**

`discover_formal_input_identity()` 只枚举 frozen binding 中的内容寻址 artifacts；raw 文件不得进入正式 plan，也不得成为运行时 fallback。

- [ ] **Step 5: 运行 Gate 1B 测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_formal_server_data_contract.py \
tests/test_formal_entry_preflight.py tests/test_hybrid_sealed_builder.py \
tests/test_adopt_validation_contract.py tests/test_source_sales_canonicalization.py \
-q -p no:cacheprovider --basetemp /tmp/d1-d6-task14-sealed-inputs
```

Expected: 全部通过；Gate 1A-V 验收的 proof、新 immutable root 和唯一 deployment manifest 可被独立消费；formal preflight 验证完整 status/counts/rows/digests 和 sidecar/manifest/deployment identity；删除、置 null 或篡改任一 proof 字段时 blocked 并返回稳定失败码；fresh install 的 missing/changed bytes 被拒绝；不运行训练。

- [ ] **Step 6: 提交 Gate 1B**

```bash
git add src/utils/formal_sealed_inputs.py scripts/validate_d1_d6_protocol_inputs.py scripts/run_unified_d1_d6.py tests/test_formal_server_data_contract.py
git commit -m "feat: require sealed-only formal dataset handles"
```

**Gate 1B acceptance:** fresh install 能仅凭 Gate 1A-V 提交的 deployment manifest 和已物化 proof 校验新 immutable root 中六个 sealed datasets；formal preflight 对完整 repair proof fail closed，并在任一字段缺失、置 null、不闭合或 identity mismatch 时 blocked；formal identity 中没有 raw、legacy、smoke 或旧 root fallback；Gate 1B 不生成或修补 repair proof，也不创建第二份 deployment manifest。

**Gate 1 combined acceptance:** Gate 1A、Gate 1A-T、Gate 1A-X、Gate 1A-V 和 Gate 1B 均为 `GATE PASSED` 后，才允许进入 Gate 2。

### Gate 2：建立唯一 truth-free seed-bundle worker

**Files:**
- Create: `src/experiment/formal_seed_bundle.py`
- Create: `scripts/run_formal_seed_bundle.py`
- Create: `tests/fixtures/fake_fitted_formal_methods.py`
- Create: `tests/test_formal_seed_bundle_integration.py`
- Modify: `scripts/run_strict_protocol_baseline.py`

**Interfaces:**
- Consumes: `SealedDatasetHandle`、`fit_formal_method_bundle()`、`run_blind_rollout()`、worker cache、current attempt identity。
- Produces: `run_formal_seed_bundle(request: FormalSeedBundleRequest) -> FormalSeedBundleCandidate`，candidate 只位于 attempt staging directory，尚未 accepted。

- [ ] **Step 1: 写失败测试，固定一个 seed bundle 只拟合每个 method 一次联合 bundle**

```python
def test_one_seed_bundle_uses_joint_horizon_fit_once_per_method(formal_request, spies):
    candidate = run_formal_seed_bundle(formal_request, adapters=spies.adapters)
    assert spies.method_bundle_fit_calls == {
        "No-TL": 1,
        "SS-TL": 1,
        "MSWA-TL": 1,
        "MSSB-TL": 1,
        "MSML-TL": 1,
        "MSML-TL-RFE": 1,
    }
    assert candidate.horizons == (1, 2, 3, 4, 5)
```

- [ ] **Step 2: 写失败测试，修改 evaluator truth 不改变 worker trace**

```python
def test_evaluator_truth_mutation_does_not_change_worker_predictions(formal_request):
    first = run_formal_seed_bundle(formal_request)
    changed = formal_request.with_evaluator_truth_offset(1000.0)
    second = run_formal_seed_bundle(changed)
    assert first.worker_semantic_digests == second.worker_semantic_digests
    assert first.evaluated_artifact_digests != second.evaluated_artifact_digests
```

- [ ] **Step 3: 实现请求和 candidate 边界**

```python
@dataclass(frozen=True)
class FormalSeedBundleRequest:
    run_root: Path
    dataset_id: str
    scenario: str
    seed: int
    attempt_id: str
    fencing_token: int
    sealed_deployment_manifest_path: Path


@dataclass(frozen=True)
class FormalSeedBundleCandidate:
    cell_id: str
    staging_directory: Path
    manifest_path: Path
    identities: Mapping[str, str]
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5)
```

worker orchestration 必须按以下顺序调用：exact schema gate、source 180-day validation、KNN last-30-day selection、method bundle fit、blind rollout、worker trace publication。evaluator truth join 发生在 worker trace bytes 固定之后。

- [ ] **Step 4: 把 60 个正式 task 全部改为新 child CLI**

`build_matrix_tasks()` 生成的 command 第一段脚本必须全部为 `scripts/run_formal_seed_bundle.py`。新 CLI 同时出现 `--horizon` 或缺少 `--seed` 时必须退出非零。

- [ ] **Step 5: 运行 Gate 2 测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_formal_seed_bundle_integration.py \
tests/test_joint_horizon_method_bundle.py tests/test_blind_rollout_protocol.py \
tests/test_fitted_predictor_adapters.py tests/test_truth_isolation.py \
tests/test_source_pretrain_180d.py tests/test_daily_knn_protocol.py \
-q -p no:cacheprovider --basetemp /tmp/d1-d6-task14-bundle-worker
```

Expected: 全部通过；fit call counts 精确；truth mutation invariant 成立。

- [ ] **Step 6: 提交 Gate 2**

```bash
git add src/experiment/formal_seed_bundle.py scripts/run_formal_seed_bundle.py scripts/run_strict_protocol_baseline.py tests/fixtures/fake_fitted_formal_methods.py tests/test_formal_seed_bundle_integration.py
git commit -m "feat: route formal seeds through truth-free bundle worker"
```

**Gate 2 acceptance:** 六个数据集共享一个正式 worker；worker API 中不存在 truth argument；无逐 horizon child task。

### Gate 3：建立 formal legacy-deny 门禁

**Files:**
- Create: `tests/test_formal_no_legacy_calls.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Modify: `scripts/run_d4_experiment.py`
- Modify: `scripts/run_d5_experiment.py`
- Modify: `scripts/run_d6_experiment.py`

**Interfaces:**
- Consumes: Gate 2 的官方 task commands。
- Produces: 结构检查和运行时 spy，证明 formal path 的 legacy call count 为 0。

- [ ] **Step 1: 写 command-level 测试**

```python
def test_all_60_formal_tasks_use_only_new_bundle_entry(tmp_path):
    tasks = tuple(
        task
        for dataset_id in range(1, 7)
        for scenario in ("without", "with")
        for task in build_matrix_tasks(
            dataset=f"d{dataset_id}",
            scenario=scenario,
            output_dir=tmp_path / f"d{dataset_id}_{scenario}",
        )
    )
    assert len(tasks) == 60
    assert {Path(task.command[1]).name for task in tasks} == {"run_formal_seed_bundle.py"}
    forbidden = {"--horizon", "run_full_paper_experiments.py", "run_d4_experiment.py", "run_d5_experiment.py", "run_d6_experiment.py"}
    assert all(not forbidden.intersection(task.command) for task in tasks)
```

- [ ] **Step 2: 写运行时 negative-spy 测试**

把以下函数 monkeypatch 为一旦调用就抛出 `AssertionError`：

```python
load_parquet_source_target_with_diagnostics
run_single_entity_experiment
run_no_tl_experiment  # legacy module function
publish_formal_seed_bundle_output_frame
publish_formal_cell_frame
```

通过官方 bundle CLI 的 in-process main 执行 mini fixture，Expected: 成功且所有 forbidden spy call count 为 0。

- [ ] **Step 3: legacy runner 明确拒绝 formal ownership**

旧 runner 的 `--strict-paper-mode` 不再代表 sealed formal authority；README 和 `--help` 必须标注 compatibility-only。若传入由 supervisor 保留的新 formal ownership token，旧 runner 必须退出非零。

- [ ] **Step 4: 运行 Gate 3 测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_formal_no_legacy_calls.py tests/test_formal_protocol_matrix.py \
tests/test_full_paper_runner_solidified_parquet.py -q -p no:cacheprovider \
--basetemp /tmp/d1-d6-task14-no-legacy
```

Expected: 全部通过；60 个 task 不含 legacy scripts。

- [ ] **Step 5: 提交 Gate 3**

```bash
git add tests/test_formal_no_legacy_calls.py scripts/run_full_paper_experiments.py scripts/run_d4_experiment.py scripts/run_d5_experiment.py scripts/run_d6_experiment.py
git commit -m "test: forbid legacy calls from formal execution"
```

**Gate 3 acceptance:** 下次审计无需从大量代码推断旁路是否存在；一条测试直接证明正式 command 和运行时调用均未触达 legacy path。

### Gate 4：typed artifact、fencing 和原子 bundle 发布

**Files:**
- Modify: `src/experiment/formal_seed_bundle.py`
- Modify: `scripts/run_formal_seed_bundle.py`
- Modify: `scripts/run_unified_d1_d6.py`
- Modify: `src/utils/result_acceptance.py`
- Test: `tests/test_formal_seed_bundle_integration.py`
- Test: `tests/test_run_recovery_state_machine.py`

**Interfaces:**
- Consumes: typed artifact registry、`RunRecovery.set_cell_state()`、`publish_cell_directory()`、`accept_cell()`、attempt-held token。
- Produces: 一个 fenced、原子、typed、身份完整的 accepted bundle directory。

- [ ] **Step 1: 写失败测试，固定 typed artifact 集合**

每个 candidate bundle 必须且只能包含：

```text
worker_manifest.json
worker_prediction_trace.csv.gz
evaluated_prediction_trace.csv.gz
source_selection_trace.csv.gz
formal_result_rows.csv
bundle_result_manifest.json
```

测试逐一使用 registry descriptor 读取，拒绝缺失、额外、schema drift、canonical digest drift 和 semantic digest drift。

- [ ] **Step 2: 写失败测试，旧 attempt 不能冒用新 token**

```python
def test_stale_bundle_cannot_publish_with_current_state_token(run_root):
    first = create_attempt(run_root)
    stale_candidate = build_candidate(first)
    second = expire_and_resume(run_root, first)
    with pytest.raises(StaleFencingTokenError):
        publish_candidate(stale_candidate, fencing_token=first.fencing_token)
    assert not accepted_directory(run_root, stale_candidate.cell_id).exists()
```

- [ ] **Step 3: 删除 formal publication 对 `_current_fencing_token()` 的依赖**

token 只能来自 `FormalSeedBundleRequest`，并与 attempt id 一同进入 worker manifest、bundle manifest、mode acceptance、aggregate manifest 和 recovery event。任何发布函数都不得在最后时刻从 mutable `state.json` 领取 token。

- [ ] **Step 4: 接入原子目录发布**

固定状态顺序：

```text
queued
→ in_flight (CAS using attempt token)
→ build and validate private staging directory
→ publish_cell_directory with a validator that verifies every byte and identity
→ atomic rename and accepted cell event under the same lock and token
```

formal path 不得在 `publish_cell_directory()` 之后再单独调用 `accept_cell()`；现有目录发布接口已经在同一锁和 token 下完成 rename 与 accepted event。信号或异常只能留下 `in_flight`、`failed` 或 `orphaned`，不得留下 accepted sidecar 与不完整目录组合。

- [ ] **Step 5: 运行 Gate 4 测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_formal_seed_bundle_integration.py \
tests/test_prediction_artifacts.py tests/test_artifact_schema_contract.py \
tests/test_run_recovery_state_machine.py tests/test_run_layout_and_atomic_publication.py \
-q -p no:cacheprovider --basetemp /tmp/d1-d6-task14-publication
```

Expected: 全部通过；stale-token 和 partial-publication 测试明确通过拒绝条件。

- [ ] **Step 6: 提交 Gate 4**

```bash
git add src/experiment/formal_seed_bundle.py scripts/run_formal_seed_bundle.py scripts/run_unified_d1_d6.py src/utils/result_acceptance.py tests/test_formal_seed_bundle_integration.py tests/test_run_recovery_state_machine.py
git commit -m "feat: publish formal bundles through fenced typed artifacts"
```

**Gate 4 acceptance:** accepted bundle 只能由当前 attempt 的同一 token 和完整 typed identity 产生；不存在 legacy acceptance sidecar 先于内容发布的窗口。

### Gate 5：binding、heartbeat、恢复和调度生命周期

**Files:**
- Modify: `scripts/run_unified_d1_d6.py`
- Modify: `scripts/parallel_mode_runner.sh`
- Modify: `src/utils/artifact_rehydration.py` only if an existing interface cannot express the frozen formal binding
- Test: `tests/test_parallel_mode_supervisor.py`
- Test: `tests/test_unified_parallel_lifecycle.py`
- Test: `tests/test_artifact_rehydration.py`

**Interfaces:**
- Consumes: `ArtifactRehydrator.freeze_binding_set()`、`resolve_bound_artifact()`、`RunRecovery.heartbeat()`、`mark_downstream_scheduling_started()`。
- Produces: attempt-scoped frozen authority、周期 lease 更新、crash-safe resume。

- [ ] **Step 1: 写失败测试，aggregate 等下游消费者启动前必须冻结 binding**

```python
def test_aggregate_cannot_start_before_binding_is_frozen(run_root):
    publish_all_bundle_candidates_without_binding(run_root)
    result = start_aggregate(run_root)
    assert result.returncode != 0
    assert not aggregate_path(run_root).exists()
```

- [ ] **Step 2: 写失败测试，heartbeat 贯穿活动任务**

fake worker 运行超过一个短测试 lease interval；测试轮询 event/lease identity，确认 token 不变且 heartbeat_at 前进。停止 heartbeat 后 resume 必须 orphan 旧 in-flight bundle。

- [ ] **Step 3: 写失败测试，rehydration fit/predict 调用数为 0**

删除一个已注册 artifact，提供匹配的 trusted replica，恢复后验证 canonical、semantic、schema 和 full identity 未改变；fake fit/predict spy 均为 0。bytes mismatch 必须走认证操作者显式恢复，不能自签新 digest。

- [ ] **Step 4: 正式 prepare/resume 接入 binding 生命周期**

生产 bundle 与下游消费之间的顺序固定为：

```text
preflight
→ create/resume attempt and receive Lease
→ launch workers with attempt_id and fencing_token
→ atomically publish or exactly reuse accepted bundles
→ validate or explicitly rehydrate every accepted run artifact
→ freeze artifact_binding_set.json containing the complete accepted set
→ mark_downstream_scheduling_started
→ mode/global aggregate resolves only through the frozen binding
```

这里的“downstream scheduling”明确指消费 accepted artifacts 的 mode/global aggregation 与 final acceptance，不指产生 bundle 的训练 worker。这样 binding 在包含完整 60-bundle 权威后一次冻结，冻结后不再增加或替换物理引用。

- [ ] **Step 5: shell 接入 heartbeat 和 fenced failure API**

shell 只能把进程状态作为本地显示；权威 task status 由 Python recovery API记录。SIGINT/SIGTERM 先停止新增调度，再终止整个活动进程组，最后使用当前 attempt-held token 发布 `partial_failed`。

- [ ] **Step 6: 运行 Gate 5 测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_parallel_mode_supervisor.py \
tests/test_unified_parallel_lifecycle.py tests/test_artifact_rehydration.py \
tests/test_run_recovery_state_machine.py -q -p no:cacheprovider \
--basetemp /tmp/d1-d6-task14-recovery
```

Expected: 全部通过；无后台 fake child 残留。

- [ ] **Step 7: shell syntax check**

```bash
bash -n scripts/parallel_mode_runner.sh
```

Expected: exit 0。

- [ ] **Step 8: 提交 Gate 5**

```bash
git add scripts/run_unified_d1_d6.py scripts/parallel_mode_runner.sh src/utils/artifact_rehydration.py tests/test_parallel_mode_supervisor.py tests/test_unified_parallel_lifecycle.py tests/test_artifact_rehydration.py
git commit -m "feat: connect formal scheduling to binding and recovery"
```

**Gate 5 acceptance:** scheduler crash/resume 不重复 accepted bundle；D5 顺序和 16-thread budget 保持；所有恢复解析经 frozen binding。

### Gate 6：接通唯一最终 sealing gate

**Files:**
- Modify: `scripts/run_unified_d1_d6.py`
- Modify: `scripts/parallel_mode_runner.sh`
- Modify: `src/utils/result_acceptance.py`
- Test: `tests/test_sealed_run_acceptance.py`
- Create: `tests/test_formal_end_to_end_sealing.py`

**Interfaces:**
- Consumes: 60 accepted bundle manifests、12 mode identities、evaluated traces、source-selection proofs、worker trace proofs、binding、fencing token。
- Produces: authoritative `results/experiment_results.csv`、sealed acceptance report、Run Manifest、最后写入的 `SEALED_SUCCESS`。

- [ ] **Step 1: 增加 `final-seal` operation 的失败测试**

缺少、重复、额外 bundle/horizon，trace metric 不匹配，worker proof 缺失，binding identity 不匹配或 fencing token 不一致时，operation 必须非零并从 `complete_unsealed` 发布 `sealed_failed`，不得写成功 marker。

- [ ] **Step 2: 实现 official final-seal operation**

CLI operation choices 增加 `final-seal`。它自行从 frozen binding 和 accepted manifests 收集 records/proofs，不允许调用者提交未认证的内存 mappings 作为权威身份。

- [ ] **Step 3: 强制成功写入顺序和 durability**

```text
validate all inputs
→ write acceptance report temp + fsync + replace + directory fsync
→ write authoritative CSV temp + fsync + replace + directory fsync
→ write Run Manifest temp + fsync + replace + directory fsync
→ append sealed_success transition/event + fsync
→ exclusive write SEALED_SUCCESS + fsync run root
```

marker 写入后不得再修改 run 内任何权威产物。

- [ ] **Step 4: shell 成功路径调用 final-seal**

`aggregate` 只产生 `complete_unsealed`；shell 随后必须调用 `--operation final-seal` 并以其退出码作为全局成功码。

- [ ] **Step 5: 运行 Gate 6 测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_sealed_run_acceptance.py \
tests/test_formal_end_to_end_sealing.py tests/test_result_acceptance_scopes.py \
tests/test_failure_exit_propagation.py -q -p no:cacheprovider \
--basetemp /tmp/d1-d6-task14-final-seal
```

Expected: 全部通过；成功 marker 最后写入；sealed failure 不可 resume。

- [ ] **Step 6: 提交 Gate 6**

```bash
git add scripts/run_unified_d1_d6.py scripts/parallel_mode_runner.sh src/utils/result_acceptance.py tests/test_sealed_run_acceptance.py tests/test_formal_end_to_end_sealing.py
git commit -m "feat: complete formal acceptance and terminal sealing"
```

**Gate 6 acceptance:** 官方成功路径能够到达 `SEALED_SUCCESS`；`finalize_sealed_run` 不再是无调用点的库函数。

### Gate 7：官方端到端和故障注入验收

**Files:**
- Modify: `tests/test_formal_end_to_end_sealing.py`
- Modify: `tests/fixtures/fake_formal_worker.py` or replace its formal role with `tests/fixtures/fake_fitted_formal_methods.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: 官方 shell supervisor、mini sealed fixture、Gate 1A、Gate 1A-T、Gate 1A-X、Gate 1A-V、Gate 1B、Gate 2–6 的全部接口。
- Produces: 一条足以证明“整台机器使用新零件”的端到端证据。

- [ ] **Step 1: 正向 60-bundle mini run**

使用 6 个 mini datasets、2 modes、5 seeds 和 fake fitted predictors 运行官方 shell；必须生成精确 60 个 accepted bundle、12 个 mode、完整 aggregate、Run Manifest 和最后 marker。

- [ ] **Step 2: 增加旁路零调用断言**

同一测试记录并断言：

```text
legacy_loader_calls == 0
legacy_per_horizon_runner_calls == 0
legacy_publisher_calls == 0
worker_truth_accesses == 0
```

- [ ] **Step 3: 增加故障矩阵**

至少覆盖：worker 非零退出、SIGTERM、heartbeat 停止、stale worker 晚发布、artifact missing、有 trusted replica 的 rehydration、artifact bytes mismatch、重复 bundle、缺 horizon、D5-with 前置失败、aggregate 后 final gate 失败。

- [ ] **Step 4: dry-run 零写入测试**

运行 Python 和 shell 两条官方 dry-run；RUN_ROOT、attempt、bundle、CSV、manifest、binding、marker 全部不存在。

- [ ] **Step 5: 更新 README**

README 仅描述已经由本 Gate 证明的命令。legacy 命令必须标记为 non-sealed compatibility；sealed data deployment、resume、rehydration、lease 和 terminal semantics 与实际 CLI 一致。

- [ ] **Step 6: 运行端到端测试**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py --timeout 180 -- \
python -m pytest tests/test_formal_end_to_end_sealing.py \
tests/test_formal_no_legacy_calls.py tests/test_formal_entry_preflight.py \
tests/test_parallel_mode_supervisor.py -q -p no:cacheprovider \
--basetemp /tmp/d1-d6-task14-e2e
```

Expected: 全部通过，且命令在 180 秒内完成；若返回 124，立即停止并交给用户手动运行，不得缩小端到端合同来追求通过。

- [ ] **Step 7: 提交 Gate 7**

```bash
git add tests/test_formal_end_to_end_sealing.py tests/fixtures/fake_formal_worker.py tests/fixtures/fake_fitted_formal_methods.py README.md
git commit -m "test: gate the complete formal sealing path"
```

**Gate 7 acceptance:** official-entry 正向、负向和恢复场景全部由同一调用链覆盖；不能通过 mock 掉被审查的 orchestration 本身。

### Gate 8：平台验证和最终只读审计准备

**Files:**
- Modify: `README.md` only if actual verified server commands differ
- Create: `docs/superpowers/reviews/d1-d6-task14-verification-record.md`

**Interfaces:**
- Consumes: Gate 0、Gate 1A、Gate 1A-T、Gate 1A-X、Gate 1A-V、Gate 1B、Gate 2–7 的固定提交序列。
- Produces: clean-server preflight/dry-run 证据和最终审计输入。

- [ ] **Step 1: 本地联合轻量套件**

运行 Task 1–13 原联合套件、Gate 1A、Gate 1A-T operator tests、Gate 1A-V authority/manifest checks、Gate 1B、Gate 2–7 新测试、compileall、`bash -n` 和 `git diff --check`。Gate 8 不重跑 Gate 1A-X 正式物化。任何 warning 必须分类记录，不得只报告退出码。

- [ ] **Step 2: Ubuntu/Python 3.10 clean checkout 验证**

在服务器 fresh checkout 固定候选 SHA，通过正式部署 manifest 安装 sealed 数据；运行 preflight 和两条 dry-run。不得复制本地 `.venv`、绝对路径、历史 `outputs/runs/` 或未注册 sidecar。

- [ ] **Step 3: 验证候选 SHA 自包含关系**

记录代码 SHA、设计 digest、部署 manifest digest、六个 dataset digests、schema registry digest、result registry digest 和 60-bundle run plan digest。

- [ ] **Step 4: 冻结候选，不再在审计中顺手修改**

最终审计开始后只读。若出现 BLOCKER/HIGH，退回对应 Gate；不得在审计过程中直接修复并继续沿用原审计结论。

- [ ] **Step 5: 提交验证记录**

```bash
git add README.md docs/superpowers/reviews/d1-d6-task14-verification-record.md
git commit -m "docs: record Task 14 formal verification"
```

**Gate 8 acceptance:** fresh-server preflight/dry-run 使用与训练相同的 resolver；工作树干净；不存在本地路径或 ignored data 的隐含依赖。

---

## 5. 进度汇报规则

为避免再次出现“局部完成被汇报为整体完成”，进度只允许使用以下状态：

| 状态 | 含义 |
|---|---|
| `NOT STARTED` | Gate 尚未开始 |
| `IMPLEMENTED, NOT GATED` | 代码已写，但失败/通过测试或审查未完成 |
| `GATE PASSED` | 本 Gate 的全部 acceptance evidence 已通过且需要的提交已形成；Gate 1A-X 以用户 exit code、execution report、manifest candidate 和新 root 取代 Git 提交 |
| `INTEGRATION COMPLETE, AUDIT PENDING` | Gate 0、Gate 1A、Gate 1A-T、Gate 1A-X、Gate 1A-V、Gate 1B、Gate 2–8 全部通过，尚未开始最终只读审计 |
| `READY` | 最终只读审计无 BLOCKER/HIGH |
| `BLOCKED AT GATE <id>` | 明确退回某个 Gate，包括 `Gate 1A-T`、`Gate 1A-X` 或 `Gate 1A-V`，不使用“基本完成”表述 |

每日报告固定为：

```text
当前 Gate：
今日完成的可验证结果：
实际执行命令与退出码：
新增或改变的正式调用边：
legacy 旁路调用数：
尚未满足的 Gate 条件：
是否出现范围变更：否 / 已停止并请求决策
下一步仅做：
```

禁止使用“代码已经差不多”“测试基本通过”“只剩收尾”等无法审计的描述。

---

## 6. 时间计划和停止条件

### 6.1 工期

在一名熟悉仓库的实施者、设计不变、无 180 秒 timeout、mini fixture 可在本地稳定运行的前提下：

| 工作 | 理想工程日 | 稳妥工程日 |
|---|---:|---:|
| Gate 0–1B：冻结、adoption proof、authority 物化、sealed input、部署合同 | 1.0–1.5 | 2.0–2.5 |
| Gate 2–3：统一 bundle worker、禁止 legacy | 1.5–2.0 | 2.0–3.0 |
| Gate 4–5：typed publication、fencing、binding、recovery | 2.0 | 3.0–4.0 |
| Gate 6–7：final seal、E2E、故障注入 | 1.5–2.0 | 2.0–3.0 |
| Gate 8：服务器验证和只读审计准备 | 0.5–1.0 | 1.0–2.0 |
| **合计** | **6.5–8.5** | **8.5–12.5** |

该时间不包括真实 D1–D6 模型训练时长，也不包括设计范围变化。

### 6.2 强制停止条件

出现以下任一情况必须停止当前 Gate，不得通过简化合同继续：

- timeout wrapper 返回 124；
- Gate 1A-T 或 Gate 1A-V 任一 Codex Python 命令返回 124；立即停止且不得拆分或重试；
- Gate 1A-X 用户 operator 命令非零；不得自动重试，先只读审查失败报告并取得新授权；
- producer 不支持单 dataset 调用，或 operator 必须进一步按 entity、batch、日期区间或文件片段拆分；
- D3、D4、D5、D6 不能在同一 private root 内全部完成并逐 dataset 独立验证；
- 需要修改冻结设计才能继续；
- Gate 1A-X/V 发现旧 root inventory 改变、D1/D2 任一 bytes 改变或 D3–D6 source/target bytes 与旧 authority 不完全相同；
- private build root 与 final deployment parent 不在同一文件系统，或无法以一次 rename 发布原本不存在的新 root；
- final deployment path 已存在、private/final path 重叠、存在 symlink 或旧 root 会被移动/覆盖；
- execution report、manifest candidate、content-set digest、规定 artifact coverage 或 proof identity 无法闭合；
- mini E2E 必须 mock 掉正式 orchestration 才能通过；
- 正式 worker 必须读取 raw/legacy 数据才能运行；
- D3–D6 无法提供可信 adoption/repair proof；
- 服务器 sealed bytes 无法与部署 manifest 对齐；
- stale token 或半发布目录无法被可靠拒绝；
- `SEALED_SUCCESS` 无法保证最后写入；
- 同一 Gate 连续两轮修改后仍出现同类旁路，必须先做根因复核，不继续堆补丁。

---

## 7. 如何降低下次审计大修概率

本计划不能保证审计永远没有发现，但可以把“最终才发现系统没接线”的概率降到最低。下次审计前必须满足：

1. **设计未漂移。** 冻结 digest 测试通过。
2. **唯一入口。** 60 个 task command 全部指向同一个 bundle worker。
3. **旁路为零。** legacy loader/runner/publisher 既有静态门禁，也有运行时 zero-call assertion。
4. **同一链路验收。** preflight、正式运行和 dry-run 共享相同 sealed resolver。
5. **真实状态接线。** scheduler 的 cell 状态、heartbeat、fencing 和 binding 均由正式 E2E 覆盖。
6. **最终门禁可达。** 官方 shell 成功路径确实写出最后 marker，而不是直接测试 finalizer 函数。
7. **故障先于审计注入。** crash、signal、stale worker、artifact corruption、D5 dependency 和 duplicate bundle 在提交审计前已验证。
8. **服务器环境先验证。** 不把 macOS 本地通过视为 Ubuntu/Python 3.10 可用证明。
9. **审计候选冻结。** 审计期间不修改候选；发现问题明确退回 Gate。
10. **完成定义唯一。** 只有最终审计无 BLOCKER/HIGH 才使用 `READY`。

这套机制的目标不是让审计“不能发现问题”，而是让问题在对应 Gate 当天暴露，而不是在 Task 1–13 全部宣布完成后集中暴露。

---

## 8. Task 1–13 到 Task 14 Gate 的覆盖映射

| 原 Task | Task 14 负责证明其正式接线的 Gate |
|---|---|
| Task 1：协议与特征冻结 | Gate 0、Gate 1A-T、Gate 1A-V、Gate 1B、Gate 2 |
| Task 2：封存数据 | Gate 1A、Gate 1A-T、Gate 1A-X、Gate 1A-V、Gate 1B、Gate 8 |
| Task 3：truth 隔离 | Gate 2、Gate 7 |
| Task 4：source 选择 | Gate 1B、Gate 2 |
| Task 5：盲滚动 | Gate 2、Gate 7 |
| Task 6：预测接口 | Gate 2 |
| Task 7：typed artifacts | Gate 4、Gate 6 |
| Task 8：恢复状态机 | Gate 4、Gate 5、Gate 7 |
| Task 9：rehydration/binding | Gate 5、Gate 7 |
| Task 10：60 seed bundles | Gate 2、Gate 3、Gate 7 |
| Task 11：调度器 | Gate 5、Gate 7 |
| Task 12：最终门禁 | Gate 6、Gate 7 |
| Task 13：正式入口 | Gate 3、Gate 6、Gate 7、Gate 8 |

任何一项原 Task 都不因“已有单测”直接标记完成；必须由上表对应 Gate 证明它已进入官方调用链。

---

## 9. 最终完成条件

Task 14 只有在以下条件同时满足时才能标记完成：

- Gate 0、Gate 1A、Gate 1A-T、Gate 1A-X、Gate 1A-V、Gate 1B、Gate 2–8 全部为 `GATE PASSED`；
- 现有权威设计 digest 未改变；
- 唯一 tracked deployment manifest 精确绑定新 immutable root 中正式 D1–D6 artifact bytes，且无 raw、legacy、旧 root fallback 或本机路径 authority；
- 官方 task matrix 精确为 60 个 bundle，且全部调用新正式 worker；
- formal E2E 中 legacy loader/runner/publisher 调用数均为 0；
- worker truth access 为 0，truth mutation 不改变 worker semantic prediction digest；
- 所有 accepted bundle 均由同一 attempt-held fencing token 通过原子目录发布；
- 恢复只使用 frozen binding 或认证 trusted replica，fit/predict 调用数为 0；
- final acceptance 独立复算 metrics/keys/counts/identity chain；
- `SEALED_SUCCESS` 是成功路径最后一次写入；
- 两条 dry-run 均零写入；
- Ubuntu/Python 3.10 fresh-server preflight/dry-run 通过；
- 最终只读全面审查结论为 `READY — NO BLOCKING FINDINGS`。

在这些条件满足前，项目状态应报告为：

```text
协议基础设施已完成；正式执行链正在 Task 14 集成，尚未获准启动正式 D1–D6。
```
