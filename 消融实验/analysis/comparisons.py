"""Frozen comparison families for the SITA ablation.

Family membership is fixed by ``ABLATION_EXECUTION_SPEC.md`` section 6.2.
Holm is applied inside family A and inside family B, separately for RMSE and
sMAPE. Interaction, the TCN contrast, and the sensitivity contrasts are
estimated but do not enter those families.

Notation ``X–Y`` means candidate X against control Y:
``RI = (E_Y - E_X) / E_Y * 100``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.stats import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    WilcoxonResult,
    five_number,
    holm,
    paired_wilcoxon,
    percentile_bootstrap,
    relative_improvement,
    win_loss_tie,
)


DATASETS = ("D1", "D4", "D5", "D6")
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
CONDITIONS = tuple((dataset, horizon) for dataset in DATASETS for horizon in HORIZONS)

METRIC_COLUMNS = {
    "RMSE": "测试RMSE",
    "sMAPE": "测试sMAPE",
}


@dataclass(frozen=True)
class Contrast:
    comparison_id: str
    candidate_arm: str
    candidate_protocol: str
    control_arm: str
    control_protocol: str
    family: str
    holm_family: str | None
    role: str
    meaning: str


def frozen_contrasts() -> tuple[Contrast, ...]:
    """Return the predeclared contrasts in family order."""
    family_a = (
        _main("F4", "F1", "A", "A", "确认性", "总效应：源信息引导初始化加 RevIN，相对于随机初始化且无 RevIN"),
        _main("F3", "F1", "A", "A", "确认性", "无 RevIN 时，源信息引导初始化相对于随机初始化"),
        _main("F4", "F2", "A", "A", "确认性", "RevIN 相同时，源信息引导初始化相对于随机初始化"),
        _main("F4", "F3", "A", "A", "确认性", "已有源信息引导初始化时，加入 RevIN 相对于不加 RevIN"),
        _main("F2", "F1", "A", "A", "确认性", "随机初始化时，加入 RevIN 相对于不加 RevIN"),
    )
    family_b = (
        _main("F4", "P1", "B", "B", "确认性", "RevIN 与 Top-3 逆距离相同时，卷积层可调整相对于卷积层冻结"),
        _main("F4", "P2", "B", "B", "确认性", "RevIN 相同时，Top-3 逆距离加权相对于 Top-1 单源"),
        _main("F4", "P3", "B", "B", "确认性", "Top-3 与 RevIN 相同时，逆距离加权相对于均匀加权"),
        _main("P3", "P2", "B", "B", "确认性", "RevIN 相同时，Top-3 均匀加权相对于 Top-1 单源"),
    )
    interaction = (
        Contrast(
            comparison_id="I(F4–F3)−(F2–F1)",
            candidate_arm="F4",
            candidate_protocol="main",
            control_arm="F1",
            control_protocol="main",
            family="交互",
            holm_family=None,
            role="补充",
            meaning="I_c = RI(F4–F3) − RI(F2–F1)，检验 RevIN 的相对改善是否随初始化方式改变",
        ),
    )
    g5 = (
        Contrast(
            comparison_id="SITA_TCN–TCN_REVIN",
            candidate_arm="SITA_TCN",
            candidate_protocol="main",
            control_arm="TCN_REVIN",
            control_protocol="main",
            family="G5",
            holm_family=None,
            role="单独报告",
            meaning="同为 TCN 且都使用 RevIN 时，源信息引导初始化相对于目标模型原生初始化",
        ),
    )
    sensitivity = (
        _protocol("F1", "alt_lr", "S1", "另一个学习率相对于主协议学习率"),
        _protocol("F2", "alt_lr", "S1", "另一个学习率相对于主协议学习率"),
        _protocol("F3", "alt_lr", "S1", "另一个学习率相对于主协议学习率"),
        _protocol("F4", "alt_lr", "S1", "另一个学习率相对于主协议学习率"),
        _protocol("F4", "random3", "S2", "Random-3 相对于 Top-3"),
        _protocol("F4", "bottom3", "S2", "Bottom-3 相对于 Top-3"),
        _protocol("F4", "k5", "S3", "K=5 相对于 K=3"),
        _protocol("F4", "ell15", "S4", "比较窗口 15 天相对于 30 天"),
    )
    return family_a + family_b + interaction + g5 + sensitivity


def _main(
    candidate: str,
    control: str,
    family: str,
    holm_family: str | None,
    role: str,
    meaning: str,
) -> Contrast:
    return Contrast(
        comparison_id=f"{candidate}–{control}",
        candidate_arm=candidate,
        candidate_protocol="main",
        control_arm=control,
        control_protocol="main",
        family=family,
        holm_family=holm_family,
        role=role,
        meaning=meaning,
    )


def _protocol(arm: str, protocol: str, family: str, meaning: str) -> Contrast:
    return Contrast(
        comparison_id=f"{arm}/{protocol}–{arm}/main",
        candidate_arm=arm,
        candidate_protocol=protocol,
        control_arm=arm,
        control_protocol="main",
        family=family,
        holm_family=None,
        role="敏感性/探索性",
        meaning=meaning,
    )


def condition_vector(cells: pd.DataFrame, arm: str, protocol: str, column: str) -> np.ndarray:
    """Return 12 seed-means in canonical dataset-by-horizon order.

    Rows may arrive in any order. Pairing is by dataset and horizon, not by
    the original row position or by seed.
    """
    subset = cells.loc[(cells["分支"] == arm) & (cells["协议"] == protocol), ["数据集", "预测步长", "种子", column]]
    if subset.empty:
        raise KeyError(f"missing cells for {arm}/{protocol}")
    grouped = subset.groupby(["数据集", "预测步长"], sort=False)[column]
    means = grouped.mean()
    counts = grouped.size()
    ordered = []
    for dataset, horizon in CONDITIONS:
        key = (dataset, horizon)
        if key not in means.index:
            raise KeyError(f"{arm}/{protocol} is missing {dataset} h={horizon}")
        if int(counts.loc[key]) != len(SEEDS):
            raise KeyError(f"{arm}/{protocol} {dataset} h={horizon} does not have five seeds")
        ordered.append(float(means.loc[key]))
    return np.asarray(ordered, dtype=np.float64)


def seed_matrix(cells: pd.DataFrame, arm: str, protocol: str, column: str) -> np.ndarray:
    """Return errors with shape ``(12, 5)`` in canonical condition and seed order."""
    ordered = []
    for dataset, horizon in CONDITIONS:
        row = []
        for seed in SEEDS:
            matched = cells.loc[
                (cells["分支"] == arm)
                & (cells["协议"] == protocol)
                & (cells["数据集"] == dataset)
                & (cells["预测步长"] == horizon)
                & (cells["种子"] == seed),
                column,
            ]
            if len(matched) != 1:
                raise KeyError(f"{arm}/{protocol} {dataset} h={horizon} seed {seed} is not unique")
            row.append(float(matched.iloc[0]))
        ordered.append(row)
    return np.asarray(ordered, dtype=np.float64)


def analyze_cells(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the comparison table, 12-condition RI detail, dataset summary, and disagreements.

    Holm multipliers use every predeclared contrast in a family. They are not
    recomputed from the subset that happens to be significant.
    """
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for contrast in frozen_contrasts():
        for metric, column in METRIC_COLUMNS.items():
            if contrast.family == "交互":
                candidate, control, ri = _interaction(cells, column)
            else:
                candidate = condition_vector(cells, contrast.candidate_arm, contrast.candidate_protocol, column)
                control = condition_vector(cells, contrast.control_arm, contrast.control_protocol, column)
                ri = relative_improvement(candidate, control)
            record = _summary_record(contrast, metric, candidate, control, ri, cells, column)
            summary_rows.append(record)
            detail_rows.extend(_detail_rows(contrast, metric, candidate, control, ri, cells, column))
    summary = pd.DataFrame(summary_rows)
    summary = assign_holm(summary)
    detail = pd.DataFrame(detail_rows)
    dataset_summary = _dataset_summary(detail)
    disagreement = _disagreement(detail, cells)
    return summary, detail, dataset_summary, disagreement


def assign_holm(summary: pd.DataFrame) -> pd.DataFrame:
    """Adjust p-values inside each ``(holm_family, metric)`` block only."""
    adjusted = np.full(len(summary), np.nan, dtype=np.float64)
    grouped = summary.groupby(["Holm族", "指标"], dropna=True, sort=False)
    for (_, _), block in grouped:
        adjusted[block.index.to_numpy()] = holm(block["Wilcoxon_p"].to_numpy(dtype=np.float64))
    output = summary.copy()
    output["Holm_p"] = adjusted
    return output


def validation_selection(cells: pd.DataFrame) -> pd.DataFrame:
    """Choose a learning rate from the five-seed mean validation RMSE.

    The choice is recorded for F1, F2, F3, and F4. It does not replace the
    main-protocol test results. Exact ties stay unresolved.
    """
    rows = []
    for arm in ("F1", "F2", "F3", "F4"):
        main_val = condition_vector(cells, arm, "main", "验证RMSE")
        alt_val = condition_vector(cells, arm, "alt_lr", "验证RMSE")
        main_rmse = condition_vector(cells, arm, "main", "测试RMSE")
        alt_rmse = condition_vector(cells, arm, "alt_lr", "测试RMSE")
        main_smape = condition_vector(cells, arm, "main", "测试sMAPE")
        alt_smape = condition_vector(cells, arm, "alt_lr", "测试sMAPE")
        for index, (dataset, horizon) in enumerate(CONDITIONS):
            if main_val[index] < alt_val[index]:
                choice = "main"
            elif alt_val[index] < main_val[index]:
                choice = "alt_lr"
            else:
                choice = "tie"
            selected_rmse = {"main": main_rmse[index], "alt_lr": alt_rmse[index]}.get(choice, np.nan)
            selected_smape = {"main": main_smape[index], "alt_lr": alt_smape[index]}.get(choice, np.nan)
            rows.append(
                {
                    "分支": arm,
                    "数据集": dataset,
                    "预测步长": horizon,
                    "主协议验证RMSE": main_val[index],
                    "另一学习率验证RMSE": alt_val[index],
                    "选择": choice,
                    "选择后测试RMSE": selected_rmse,
                    "选择后测试sMAPE": selected_smape,
                    "主协议测试RMSE": main_rmse[index],
                    "另一学习率测试RMSE": alt_rmse[index],
                    "主协议测试sMAPE": main_smape[index],
                    "另一学习率测试sMAPE": alt_smape[index],
                }
            )
    return pd.DataFrame(rows)


def descriptive_condition_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean RMSE and sMAPE for every completed arm and condition."""
    rows = []
    keys = cells.groupby(["分支", "协议", "数据集", "预测步长"], sort=False)
    for (arm, protocol, dataset, horizon), block in keys:
        rmse = block["测试RMSE"].to_numpy(dtype=np.float64)
        smape = block["测试sMAPE"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "分支": arm,
                "协议": protocol,
                "数据集": dataset,
                "预测步长": int(horizon),
                "种子数": int(len(block)),
                "测试RMSE均值": float(np.mean(rmse)),
                "测试RMSE样本标准差": float(np.std(rmse, ddof=1)),
                "测试sMAPE均值": float(np.mean(smape)),
                "测试sMAPE样本标准差": float(np.std(smape, ddof=1)),
                "有符号误差均值": float(np.mean(block["有符号误差"].to_numpy(dtype=np.float64))),
                "零销量天数均值": float(np.mean(block["零销量天数"].to_numpy(dtype=np.float64))),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["分支", "协议", "数据集", "预测步长"], kind="mergesort").reset_index(drop=True)


def _interaction(cells: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f4 = condition_vector(cells, "F4", "main", column)
    f3 = condition_vector(cells, "F3", "main", column)
    f2 = condition_vector(cells, "F2", "main", column)
    f1 = condition_vector(cells, "F1", "main", column)
    ri_with_init = relative_improvement(f4, f3)
    ri_without_init = relative_improvement(f2, f1)
    return f4, f1, ri_with_init - ri_without_init


def _summary_record(
    contrast: Contrast,
    metric: str,
    candidate: np.ndarray,
    control: np.ndarray,
    ri: np.ndarray,
    cells: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    counts = win_loss_tie(ri)
    test = paired_wilcoxon(ri)
    interval = percentile_bootstrap(ri, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    paired_diff = control - candidate if contrast.family != "交互" else ri
    diff_summary = five_number(paired_diff)
    ri_summary = five_number(ri)
    agreement = _seed_agreement(cells, contrast, column, ri) if contrast.family != "交互" else None
    return {
        "比较": contrast.comparison_id,
        "比较族": contrast.family,
        "Holm族": contrast.holm_family,
        "角色": contrast.role,
        "含义": contrast.meaning,
        "指标": metric,
        "条件数": int(ri.size),
        "平均RI": float(np.mean(ri)),
        "中位RI": float(np.median(ri)),
        "RI最小值": ri_summary["min"],
        "RI第一四分位": ri_summary["q1"],
        "RI第三四分位": ri_summary["q3"],
        "RI最大值": ri_summary["max"],
        "改善条件数": counts.wins,
        "变差条件数": counts.losses,
        "持平条件数": counts.ties,
        "配对差值最小值": diff_summary["min"],
        "配对差值第一四分位": diff_summary["q1"],
        "配对差值中位数": diff_summary["median"],
        "配对差值第三四分位": diff_summary["q3"],
        "配对差值最大值": diff_summary["max"],
        "CI下限": interval.low,
        "CI上限": interval.high,
        "bootstrap次数": interval.n_resamples,
        "bootstrap种子": interval.seed,
        "百分位方法": interval.percentile_method,
        "Wilcoxon统计量": test.statistic,
        "Wilcoxon_p": test.pvalue,
        "Wilcoxon方法": test.method,
        "Wilcoxon有效条件数": test.n,
        "剔除的零RI数": test.n_zero_discarded,
        "种子方向一致比例": agreement,
    }


def _detail_rows(
    contrast: Contrast,
    metric: str,
    candidate: np.ndarray,
    control: np.ndarray,
    ri: np.ndarray,
    cells: pd.DataFrame,
    column: str,
) -> list[dict[str, object]]:
    rows = []
    for index, (dataset, horizon) in enumerate(CONDITIONS):
        seed_note = ""
        if contrast.family != "交互":
            seed_ri = _seed_ri(cells, contrast, column, dataset, horizon)
            seed_note = ";".join(f"{seed}:{value:.10g}" for seed, value in zip(SEEDS, seed_ri))
        rows.append(
            {
                "比较": contrast.comparison_id,
                "比较族": contrast.family,
                "角色": contrast.role,
                "指标": metric,
                "数据集": dataset,
                "预测步长": horizon,
                "候选误差": float(candidate[index]) if contrast.family != "交互" else np.nan,
                "对照误差": float(control[index]) if contrast.family != "交互" else np.nan,
                "配对差值": float(control[index] - candidate[index]) if contrast.family != "交互" else float(ri[index]),
                "RI": float(ri[index]),
                "逐种子RI": seed_note,
            }
        )
    return rows


def _seed_ri(
    cells: pd.DataFrame,
    contrast: Contrast,
    column: str,
    dataset: str,
    horizon: int,
) -> np.ndarray:
    candidate = seed_matrix(cells, contrast.candidate_arm, contrast.candidate_protocol, column)
    control = seed_matrix(cells, contrast.control_arm, contrast.control_protocol, column)
    condition_index = CONDITIONS.index((dataset, horizon))
    return relative_improvement(candidate[condition_index], control[condition_index])


def _seed_agreement(
    cells: pd.DataFrame,
    contrast: Contrast,
    column: str,
    condition_ri: np.ndarray,
) -> float:
    candidate = seed_matrix(cells, contrast.candidate_arm, contrast.candidate_protocol, column)
    control = seed_matrix(cells, contrast.control_arm, contrast.control_protocol, column)
    agreements = 0
    total = 0
    for index in range(len(CONDITIONS)):
        seed_ri = relative_improvement(candidate[index], control[index])
        target = condition_ri[index]
        for value in seed_ri:
            total += 1
            same_sign = (value > 0 and target > 0) or (value < 0 and target < 0) or (value == 0 and target == 0)
            agreements += int(same_sign)
    return agreements / total


def _dataset_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordinary = detail.loc[detail["比较族"] != "交互"]
    for (comparison, metric, dataset), block in ordinary.groupby(["比较", "指标", "数据集"], sort=False):
        ri = block["RI"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "比较": comparison,
                "指标": metric,
                "数据集": dataset,
                "步长数": int(len(block)),
                "平均RI": float(np.mean(ri)),
                "改善步长数": int(np.sum(ri > 0.0)),
                "变差步长数": int(np.sum(ri < 0.0)),
                "持平步长数": int(np.sum(ri == 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _disagreement(detail: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordinary = detail.loc[detail["比较族"] != "交互"]
    for (comparison, dataset, horizon), block in ordinary.groupby(["比较", "数据集", "预测步长"], sort=False):
        rmse = float(block.loc[block["指标"] == "RMSE", "RI"].iloc[0])
        smape = float(block.loc[block["指标"] == "sMAPE", "RI"].iloc[0])
        if np.sign(rmse) == np.sign(smape):
            continue
        contrast = next(item for item in frozen_contrasts() if item.comparison_id == comparison)
        rows.append(
            {
                "比较": comparison,
                "数据集": dataset,
                "预测步长": int(horizon),
                "RMSE的RI": rmse,
                "sMAPE的RI": smape,
                "候选有符号误差均值": _condition_mean_column(cells, contrast.candidate_arm, contrast.candidate_protocol, dataset, horizon, "有符号误差"),
                "对照有符号误差均值": _condition_mean_column(cells, contrast.control_arm, contrast.control_protocol, dataset, horizon, "有符号误差"),
                "候选零销量天数均值": _condition_mean_column(cells, contrast.candidate_arm, contrast.candidate_protocol, dataset, horizon, "零销量天数"),
                "对照零销量天数均值": _condition_mean_column(cells, contrast.control_arm, contrast.control_protocol, dataset, horizon, "零销量天数"),
            }
        )
    return pd.DataFrame(rows)


def _condition_mean_column(
    cells: pd.DataFrame,
    arm: str,
    protocol: str,
    dataset: str,
    horizon: int,
    column: str,
) -> float:
    matched = cells.loc[
        (cells["分支"] == arm)
        & (cells["协议"] == protocol)
        & (cells["数据集"] == dataset)
        & (cells["预测步长"] == horizon),
        column,
    ]
    return float(np.mean(matched.to_numpy(dtype=np.float64)))


def wilcoxon_method_label(result: WilcoxonResult) -> str:
    return result.method
