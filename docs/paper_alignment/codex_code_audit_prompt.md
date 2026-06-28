# codex_code_audit_prompt.md

你现在只做代码审计，不要修改文件。目标是检查当前仓库对论文《A multi-source multi-layer-based transfer learning approach for forecasting customer demands of newly launched products》的复现是否 paper-aligned。

请按以下规则执行：

1. 先定位主入口、配置文件、数据加载、KNN 选源、RFE、CNN、No-TL、SS-TL、MSWA-TL、MSSB-TL、MSML-TL、MSML-TL-RFE、实验 runner、结果输出、supply chain cost model。
2. 不要把工程默认值写成论文事实。每个发现必须归类为：
   - `paper_aligned`
   - `paper_missing_detail_engineering_default`
   - `paper_conflict_requires_choice`
   - `paper_deviation`
   - `possible_leakage`
3. 不要只看最终 RMSE；必须检查数据流是否泄漏 target test。
4. 审计输出必须包含文件路径、函数/类名、关键行号、当前行为、论文期望行为、风险等级。

## 重点检查清单

| 检查项 | 代码中应查什么 | 期望 paper-aligned 行为 | 风险等级 |
| --- | ------- | ------------------- | ---- |
| Dataset 1 | 数据加载与过滤 | Store 1–3、Item 1–10；Item 10 target | 高 |
| Dataset 2 | pasta 数据、promotion | Item 10 target；promotion 作为可用特征 | 高 |
| Dataset 3 | Rossmann store-level sales | Store 10 target；first 10 stores/regions 假设需记录 | 高 |
| train/val/test split | date/time step 切片 | Table 2/3 对齐 | 高 |
| KNN window | source selection 使用哪些 target 行 | 使用 target observed window/约 30 days，不使用 target test | 高 |
| KNN distance | 距离公式 | Euclidean distance | 高 |
| KNN feature space | 输入列 | all available features；字段需记录 | 高 |
| no sharing pool | source pool filter | 同 store/brand/region 内 | 高 |
| with sharing pool | source pool filter | 跨 store/brand/region | 高 |
| selected sources | source IDs/rank/distance | 可与 Table 5/6 对照 | 高 |
| weights | 权重公式 | 正文/表格 inverse distance 与 Algorithm 1 distance/sum 冲突，必须显式选择 | 高 |
| RFE | fit/transform scope | 不使用 target test；记录 estimator、n_features、selected features | 高 |
| CNN | model summary | Conv1D-MaxPool-Conv1D-MaxPool-Conv1D-Flatten-Dense | 高 |
| frozen layers | layer trainable flags | 冻结前 4 个计算层：Conv1D_1, MaxPool_1, Conv1D_2, MaxPool_2 | 高 |
| No-TL | 训练数据 | target only，无 source | 高 |
| SS-TL | KNN single source | 最邻近 source pretrain + target fine-tune | 高 |
| MSWA-TL | prediction aggregation | 3 个 source model 的预测加权平均 | 高 |
| MSSB-TL | switching | 使用 target validation RMSE 选模型，不用 test RMSE | 高 |
| MSML-TL | layer fusion | 多 source weights/biases layer-wise weighted average | 高 |
| MSML-TL-RFE | RFE + MSML | RFE 在 MSML 前，保留约 40%–60% features | 高 |
| horizon | 训练/评估循环 | 1–5 days ahead；记录 per-horizon 与平均 | 中 |
| repeats/seeds | 实验重复 | 论文未给次数/seed，标为 engineering default | 中 |
| RMSE/accuracy | metric formula | RMSE；accuracy=1/RMSE | 中 |
| cost model | SC Eq. 1–4/Table 1 | Dataset 1 only；fixed LT=5；dynamic LT=3+2 | 高 |
| output schema | CSV/MD/JSON 字段 | 至少包含 dataset, method, scenario, source_count, horizon, selected_sources, distances, weights, weight_formula, rfe_features, RMSE, metric_space, runtime, cost | 高 |

## 必须报告的潜在冲突

1. 正文/表格 inverse distance weights vs Algorithm 1 distance/sum_distance。
2. Algorithm 1 注释 “Train the new CNN network using the target domain test data” 与 line 19 Target_train_data/Target_val_data 冲突。
3. Dataset 3 特征正文与 Fig. 9 不完全一致。
4. Table 8 with-info mean RMSE 表格 0.1937 vs 正文 0.1973。
5. Table 13 9-source runtime 表格 3915 vs 正文 3519。
6. horizon 文本中 1/15/30 与实际 1–5 days 实验的关系。
7. RFE fit scope：source-only / target+source 不清楚。

## 输出格式

请输出 Markdown：

```markdown
# Paper-aligned code audit report

## 1. Executive conclusion
- PASS / PARTIAL / FAIL
- 最主要偏差：...

## 2. File map
| 模块 | 文件 | 函数/类 | 作用 |

## 3. Paper-aligned checks
| 检查项 | 当前代码行为 | 论文期望 | 证据位置 | 风险 | 结论 |

## 4. Leakage checks
| 泄漏风险 | 当前代码行为 | 是否泄漏 | 需要修改吗 |

## 5. Paper conflicts and implementation choices
| 冲突项 | 当前采用版本 | 是否显式记录 | 建议 |

## 6. Missing paper details / engineering defaults
| 参数 | 当前默认 | 是否记录 | 是否影响结果 |

## 7. Result-output completeness
| 输出字段 | 当前是否有 | 是否足够复现实验审计 |

## 8. Minimal recommended fixes
只列必须修复项，不做代码修改。
```
