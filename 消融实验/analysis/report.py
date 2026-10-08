"""Chinese report for one completed statistical run."""

from __future__ import annotations

import numpy as np
import pandas as pd


def write_report(
    path,
    *,
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    disagreement: pd.DataFrame,
    notes: dict[str, object],
    missing_k10: int,
    cell_count: int,
    planned_cell_count: int,
) -> None:
    lines = [
        "# 消融实验统计分析",
        "",
        "本报告只汇总已经完成的消融 cell。它不更新论文，也不补跑实验。",
        "",
        "## 1. 数据与核对",
        "",
        f"- 结果工作簿：`/Users/ming/Desktop/复现实验/论文文稿相关/原始实验结果/消融实验全部结果.xlsx`。",
        f"- 工作簿 SHA256：`{notes.get('workbook_sha256', '')}`。",
        f"- runs 目录：`{notes.get('runs_path', '')}`。说明页记录的位置与该目录一致。",
        f"- 计划 cell 为 {planned_cell_count} 个。工作簿和 runs 中实际完成 {cell_count} 个，状态均为 COMPLETED。",
        f"- F4/k10 按说明页记为未运行，共 {missing_k10} 个计划 cell。这些 cell 没有补值，也没有记成“候选源不足”。因此不能把 {cell_count} 条记录写成原计划全部完成。",
        f"- 每个已完成条件都有种子 42、43、44、45、46。每个 cell 的逐日预测覆盖冻结测试区间的 180 个日期，没有重复日期，权重版本为 EMA。",
        f"- 按冻结公式从逐日预测重算 RMSE 和 sMAPE 后，与逐格记录的最大绝对差为 RMSE {notes.get('max_abs_rmse_gap')}、sMAPE {notes.get('max_abs_smape_gap')}。",
        f"- 同一公式与 `sita_ablation.metrics.evaluate_original_sales` 的最大绝对差为 {notes.get('official_metric_max_abs_gap')}，核对了 {notes.get('official_metric_cells_checked')} 个 cell。",
        f"- 五种子均值使用样本标准差。逐格重算与“五种子均值”表一致。第 50 轮训练曲线的验证 RMSE、验证 sMAPE 与逐格记录一致。",
        f"- runs 中的 `predictions.parquet` 和 `cell_record.json` 与工作簿逐日预测、逐格指标一致。预测值最大绝对差为 {notes.get('max_abs_run_prediction_gap')}。",
        f"- 工作簿审计表与 runs 同级 audit 目录一致，文件为 {'、'.join(notes.get('audit_files') or [])}。这些文件里的 failed、hash_mismatch、insufficient_candidates、missing 都是空列表。phase_2 审计文件{'存在' if notes.get('phase_2_audit_present') else '不存在'}。F4/k10 的缺失来自说明页，不是审计里的候选源不足。",
        "",
        "## 2. 统计单位与公式",
        "",
        "主要指标是测试集原始销量尺度上的 RMSE 和 sMAPE。二者都是误差，数值越低越好。计算前先把负数裁成 0：",
        "",
        "```text",
        "RMSE = sqrt(mean((pred - true)^2))",
        "sMAPE = 100 * mean(2 * |pred - true| / (|true| + |pred| + 1e-8))",
        "```",
        "",
        "每个“数据集 × 预测步长”先对种子 42–46 求算术平均。统计单位是 D1、D4、D5、D6 与步长 1、3、5 组成的 12 个配对条件。5 个随机种子不是 5 个独立样本，180 个测试日也不是 180 个独立样本。",
        "",
        "条件顺序固定为 D1、D4、D5、D6，每个数据集内按步长 1、3、5。相对改善率按这个顺序计算：",
        "",
        "```text",
        "RI = (对照方案误差 - 候选方案误差) / 对照方案误差 × 100%",
        "```",
        "",
        "RI 为正表示候选方案误差更低，RI 为负表示候选方案误差更高。记号 X–Y 中，X 是候选方案，Y 是对照方案。这与执行规范里的相对改进率是同一个公式。",
        "",
        "每一项比较对 12 个条件的 RI 做双侧 Wilcoxon 符号秩检验，调用 `scipy.stats.wilcoxon(zero_method='wilcox', method='auto', alternative='two-sided')`。没有零 RI 且条件数不超过 50 时，SciPy 1.13 实际使用精确分布，否则使用正态近似。",
        "",
        "平均 RI 的 95% 区间来自 100,000 次有放回百分位 bootstrap。重采样单位是 12 个条件。每次比较都从 `numpy.random.default_rng(20261001)` 重新开始，百分位使用 NumPy 的 linear 方法，取 2.5% 和 97.5%。",
        "",
        "配对差值是对照误差减去候选误差，单位与原指标相同。交互项没有单一对照误差，它的分布就是 12 个 I_c。",
        "",
        "## 3. 比较族",
        "",
        "校正范围以 `ABLATION_EXECUTION_SPEC.md` 第 6.2 节为准，不按本轮结果增减比较。",
        "",
        "- 族 A 有 5 项：F4–F1、F3–F1、F4–F2、F4–F3、F2–F1。RMSE 与 sMAPE 分开做 Holm 校正。F3–F1 在规范中是核心证据，因此保留在族 A，不降为探索性分析。",
        "- 族 B 有 4 项：F4–P1、F4–P2、F4–P3、P3–P2。校正只在这 4 项内部进行，不与族 A 混在一起。",
        "- 交互项 I_c = RI(F4–F3) − RI(F2–F1) 单独做 Wilcoxon 和 bootstrap，不进入 Holm。",
        "- G5 是 SITA_TCN–TCN_REVIN，单独报告未校正 p 值。",
        "- S1 到 S4 是敏感性或探索性结果，报告未校正 p 值和区间。规范中的 S1 包含 F1、F2、F3、F4 的另一学习率；本轮任务清单点名了 F4。两处都保留，并且都不进入族 A 或族 B。",
        "- 不生成 F4/k10 的比较。",
        "",
        "本轮任务清单把 F4–F2、F4–P1、F4–P2、F4–P3 放在同一组。规范把 F4–F2 放在族 A，把后三项放在族 B。报告按规范分族校正。",
        "",
        "## 4. 比较结果",
        "",
        "下表的改善条件数是 12 个条件中 RI 大于 0 的个数。Holm 列为空表示该比较不在预先规定的校正族内。12 个条件的 RI 在 `paired_ri_detail.csv`。配对差值（对照误差减候选误差）的最小值、四分位数和最大值在 `comparison_results.csv`。",
        "",
        _markdown_table(summary),
        "",
        "p 值不单独作为结论。平均 RI、区间、校正后 p 值和 12 个条件的方向需要一起看。某一项校正后 p 值较大，只说明这 12 个条件没有为该项比较提供足够强的秩证据，不能写成已经证明该因素没有作用。",
        "",
        "### 4.1 分歧条件",
        "",
        f"RMSE 与 sMAPE 的 RI 符号不同的条件有 {len(disagreement)} 个，明细在 `rmse_smape_disagreement.csv`。这些条件附有两侧的平均有符号误差和零销量天数，供后续解释，不另做检验。",
        "",
        "### 4.2 学习率的验证选择",
        "",
        "对 F1、F2、F3、F4，每个条件用 5 个种子的验证 RMSE 均值选择学习率，再读取所选协议的测试误差。验证误差只用于选择，不替代主协议。两侧验证 RMSE 完全相等时记为 tie，不指定测试结果。",
        "",
        _validation_text(validation),
        "",
        "## 5. 限制",
        "",
        f"- 推断基于 12 个配对条件。区间和 p 值描述的是这 12 个条件之间的变动，不是种子之间或日期之间的抽样变动。",
        "- 工作簿没有 LSTM、LightGBM、DLinear、PatchTST、TimesNet、MSML-TL 或 SeasonalNaive 的结果。消融分支不能代替这些基准，本报告也不比较 SITA 与它们谁的误差更低。",
        "- F4/k10 缺失。K 的敏感性只包含已完成的 K=5 与主协议 K=3，不能外推到 K=10。",
        "- 敏感性分析和 G5 的 p 值没有做 Holm 校正，不能与族 A、族 B 的校正后 p 值放在同一证据等级。",
        "- 交互项、逐数据集平均 RI 和种子方向一致比例是描述或补充结果。种子方向一致比例把每个条件的 5 个种子方向与该条件均值方向比较，不把种子当作新的检验样本。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(summary: pd.DataFrame) -> str:
    columns = [
        "比较",
        "比较族",
        "指标",
        "平均RI",
        "中位RI",
        "CI下限",
        "CI上限",
        "改善条件数",
        "变差条件数",
        "持平条件数",
        "Wilcoxon_p",
        "Holm_p",
        "Wilcoxon方法",
    ]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in summary.iterrows():
        rendered = []
        for column in columns:
            rendered.append(_format(column, row[column]))
        body.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, rule, *body])


def _validation_text(validation: pd.DataFrame) -> str:
    lines = []
    for arm, block in validation.groupby("分支", sort=False):
        counts = block["选择"].value_counts().to_dict()
        chosen = block.loc[block["选择"] != "tie"]
        if chosen.empty:
            lines.append(f"- {arm}：12 个条件都没有分出验证 RMSE，未汇总选择后的测试误差。")
            continue
        lines.append(
            "- {arm}：主协议 {main} 个条件，另一学习率 {alt} 个条件，持平 {tie} 个条件。"
            "在 {used} 个作出唯一选择的条件上，选择后测试 RMSE 均值为 {rmse:.4f}，测试 sMAPE 均值为 {smape:.4f}。".format(
                arm=arm,
                main=int(counts.get("main", 0)),
                alt=int(counts.get("alt_lr", 0)),
                tie=int(counts.get("tie", 0)),
                used=int(len(chosen)),
                rmse=float(chosen["选择后测试RMSE"].mean()),
                smape=float(chosen["选择后测试sMAPE"].mean()),
            )
        )
    return "\n".join(lines)


def _format(column: str, value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if column in {"平均RI", "中位RI", "CI下限", "CI上限"}:
        return f"{float(value):.3f}"
    if column in {"Wilcoxon_p", "Holm_p"}:
        return f"{float(value):.6f}"
    if column in {"改善条件数", "变差条件数", "持平条件数"}:
        return str(int(value))
    return str(value)
