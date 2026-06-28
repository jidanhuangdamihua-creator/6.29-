"""
Module 5: Similar Source Selection (KNN-style + Distance Weights)

本模块仅负责：
1. 为 target 从 source pool 中选择 top-k 相似源
2. 计算欧氏距离
3. 根据距离计算权重

不包含多源迁移训练逻辑。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from data_preprocessing import infer_source_selection_feature_columns, temporal_split_by_ratio_or_dates

try:
    from environment import setup_logging
except ImportError:
    setup_logging = None


LOGGER_NAME = "experiment"
KNN_REPRESENTATION_PAPER_OBSERVED_SEQUENCE = "paper_observed_sequence"
KNN_REPRESENTATION_ENGINEERING_SUMMARY_STATS = "engineering_summary_stats"
VALID_KNN_REPRESENTATIONS = {
    KNN_REPRESENTATION_PAPER_OBSERVED_SEQUENCE,
    KNN_REPRESENTATION_ENGINEERING_SUMMARY_STATS,
}


def _get_logger() -> logging.Logger:
    """获取项目统一日志器；若未初始化则按默认参数初始化。"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level="INFO", log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    return logger


class SourceSelector:
    """相似源选择器：构建签名、计算距离、生成 top-k 与权重。"""

    @staticmethod
    def _format_domain_key(key: Tuple) -> str:
        """将分组键格式化为可读 domain 名称。"""
        if len(key) >= 2:
            return f"entity={key[0]}|item={key[1]}"
        if len(key) == 1:
            return str(key[0])
        return "unknown"

    def _collect_domain_names(self, df: pd.DataFrame, group_cols: Tuple[str, str]) -> List[str]:
        """基于 group_cols 收集 domain 名称列表。"""
        if any(col not in df.columns for col in group_cols):
            return []
        grouped = df.groupby(list(group_cols), sort=False)
        names: List[str] = []
        for key, _ in grouped:
            tup = tuple(key) if isinstance(key, tuple) else (key,)
            names.append(self._format_domain_key(tup))
        return names

    def _log_debug_selection_details(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        group_cols: Tuple[str, str],
        include_sales_in_knn: bool,
        resolved_feature_cols: List[str],
        source_keys: List[Tuple],
        target_signature: np.ndarray,
        source_signatures: np.ndarray,
        distances: np.ndarray,
        results: List[Dict[str, object]],
        knn_representation: str,
    ) -> None:
        """打印 source selection 调试信息，帮助核对论文对齐情况。"""
        logger = _get_logger()

        dataset_name = (
            target_df.attrs.get("dataset_name")
            or source_df.attrs.get("dataset_name")
            or "unknown_dataset"
        )
        target_domains = self._collect_domain_names(target_df, group_cols)
        source_domains = self._collect_domain_names(source_df, group_cols)

        source_signature_shapes = [
            {"source_key": self._format_domain_key(k), "shape": (int(source_signatures.shape[1]),)}
            for k in source_keys
        ]
        ranking = [
            {
                "rank": i + 1,
                "source_key": self._format_domain_key(source_keys[int(idx)]),
                "distance": float(distances[int(idx)]),
            }
            for i, idx in enumerate(np.argsort(distances))
        ]
        top_k_final = [
            {
                "source_key": self._format_domain_key(tuple(r["source_key"]) if isinstance(r["source_key"], (list, tuple)) else (r["source_key"],)),
                "distance": float(r["distance"]),
                "weight": float(r["weight"]),
            }
            for r in results
        ]

        logger.info("[source_selection_debug] dataset=%s", dataset_name)
        logger.info("[source_selection_debug] target_domains=%s", target_domains)
        logger.info("[source_selection_debug] source_domains=%s", source_domains)
        logger.info("[source_selection_debug] include_sales_in_knn=%s", include_sales_in_knn)
        logger.info("[source_selection_debug] knn_representation=%s", knn_representation)
        logger.info("[source_selection_debug] knn_features=%s", resolved_feature_cols)
        logger.info("[source_selection_debug] knn_features_contains_sales=%s", "sales" in resolved_feature_cols)
        logger.info("[source_selection_debug] target_signature_shape=%s", tuple(target_signature.shape))
        logger.info("[source_selection_debug] source_signature_shapes=%s", source_signature_shapes)
        logger.info("[source_selection_debug] distance_ranking=%s", ranking)
        logger.info("[source_selection_debug] top_k_selected=%s", top_k_final)

    def _resolve_source_selection_features(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        feature_cols: Sequence[str],
        include_sales_in_knn: bool,
        knn_feature_mode: str | None = None,
    ) -> Tuple[List[str], Dict[str, object]]:
        """自动推断 source selection 特征，并返回诊断信息用于日志。"""
        requested = list(feature_cols) if feature_cols is not None else []
        info = infer_source_selection_feature_columns(
            source_df=source_df,
            target_df=target_df,
            candidate_cols=requested,
            include_sales_in_knn=include_sales_in_knn,
            knn_feature_mode=knn_feature_mode,
        )
        resolved = list(info.get("selected_features", []))
        if not resolved:
            raise ValueError("No resolved source-selection features.")
        return resolved, info

    def _validate_feature_columns(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
        """校验特征列是否存在并返回列表。"""
        cols = list(feature_cols)
        if not cols:
            raise ValueError("feature_cols must not be empty")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return cols

    def _resolve_signature_static_features(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> List[str]:
        """Resolve optional static/profile features for signature augmentation."""
        scenario = str(source_df.attrs.get("information_sharing_scenario", "")).strip()
        static_cols = source_df.attrs.get("signature_static_feature_cols", [])
        if not isinstance(static_cols, list):
            static_cols = []
        if scenario != "with_information_sharing":
            return []
        resolved: List[str] = []
        for col in static_cols:
            if col in source_df.columns and col in target_df.columns:
                resolved.append(str(col))
        return resolved

    def _normalize_knn_representation(self, knn_representation: str | None) -> str:
        """Resolve and validate the KNN vector representation."""
        resolved = str(knn_representation or KNN_REPRESENTATION_PAPER_OBSERVED_SEQUENCE).strip().lower()
        if resolved not in VALID_KNN_REPRESENTATIONS:
            raise ValueError(
                "Unsupported knn_representation. Use "
                f"'{KNN_REPRESENTATION_PAPER_OBSERVED_SEQUENCE}' or "
                f"'{KNN_REPRESENTATION_ENGINEERING_SUMMARY_STATS}'."
            )
        return resolved

    @staticmethod
    def _encode_static_scalar(series: pd.Series) -> float:
        """Encode scalar metadata deterministically for distance calculation."""
        non_na = series.dropna()
        if non_na.empty:
            return 0.0
        if pd.api.types.is_numeric_dtype(non_na):
            return float(pd.to_numeric(non_na, errors="coerce").iloc[-1])
        text = str(non_na.astype("string").iloc[-1])
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return float(int(digest, 16) / 4294967295.0)

    def _signature_from_df(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        static_feature_cols: Sequence[str] | None = None,
    ) -> np.ndarray:
        """
        从单个序列 DataFrame 构建固定维度签名。

        说明：
        - 为避免序列长度不同导致向量维度不一致，默认使用统计签名。
        - 每个特征拼接 5 个统计量: mean/std/min/max/last。
        - 后续可替换为时间展开向量或更复杂表示。
        """
        if df.empty:
            return np.zeros(len(feature_cols) * 5, dtype=np.float64)

        ordered = df.sort_values("date") if "date" in df.columns else df.copy()
        signature_parts: List[float] = []

        for col in feature_cols:
            values = pd.to_numeric(ordered[col], errors="coerce").to_numpy(dtype=np.float64)
            values = values[~np.isnan(values)]
            if values.size == 0:
                signature_parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            signature_parts.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.min(values)),
                    float(np.max(values)),
                    float(values[-1]),
                ]
            )

        for col in (static_feature_cols or []):
            signature_parts.append(self._encode_static_scalar(ordered[col]))

        return np.asarray(signature_parts, dtype=np.float64)

    def _sequence_signature_from_df(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        expected_dates: Sequence[pd.Timestamp] | None = None,
        expected_rows: int | None = None,
        static_feature_cols: Sequence[str] | None = None,
    ) -> np.ndarray:
        """
        Build the paper-aligned KNN vector by flattening observed time steps.

        The caller passes target observed dates for source domains, so the
        selector compares contemporaneous source/target rows and never needs
        target test rows.
        """
        if df.empty:
            raise ValueError("paper_observed_sequence requires non-empty source/target data.")
        if "date" not in df.columns:
            raise ValueError("paper_observed_sequence requires a date column.")

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values(["date"]).reset_index(drop=True)

        if expected_dates is not None:
            expected = pd.to_datetime(pd.Index(expected_dates)).dropna().sort_values()
            work = work[work["date"].isin(expected)].sort_values(["date"]).reset_index(drop=True)
            if int(work["date"].nunique()) != len(expected):
                raise ValueError(
                    "paper_observed_sequence requires each source to cover the target observed dates. "
                    f"expected_unique_dates={len(expected)} actual_unique_dates={int(work['date'].nunique())}."
                )

        if expected_rows is not None and len(work) != int(expected_rows):
            raise ValueError(
                "paper_observed_sequence requires source and target vectors to use the same row count. "
                f"expected_rows={int(expected_rows)} actual_rows={len(work)}."
            )

        values = work[list(feature_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        sequence_parts = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        static_parts = [self._encode_static_scalar(work[col]) for col in (static_feature_cols or [])]
        if static_parts:
            return np.concatenate([sequence_parts, np.asarray(static_parts, dtype=np.float64)])
        return sequence_parts.astype(np.float64)

    @staticmethod
    def _positive_int_attr(attrs: Dict[str, object], key: str, default: int = 0) -> int:
        try:
            value = int(attrs.get(key, default))
        except (TypeError, ValueError):
            value = default
        return value if value > 0 else default

    def _target_observed_window_for_paper_knn(self, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Return target train+validation rows for paper-aligned KNN when split metadata is available."""
        if target_df.empty:
            return target_df, {
                "target_window_source": "provided_target_df",
                "target_test_data_excluded": True,
                "date_alignment_mode": "empty_target",
            }

        attrs = target_df.attrs
        logger = _get_logger()
        if "paper_split_protocol" in attrs:
            required = ["observed_days", "test_days"]
            missing = [key for key in required if key not in attrs]
            if missing:
                raise ValueError(
                    "Missing solidified paper observed window attrs: "
                    f"dataset_id={attrs.get('dataset_name', 'unknown')} "
                    f"scenario={attrs.get('information_sharing_scenario', 'unknown')} "
                    f"method={attrs.get('method', 'unknown')} "
                    f"paper_split_protocol={attrs.get('paper_split_protocol')} "
                    f"required={required}"
                )

            observed_days = self._positive_int_attr(attrs, "observed_days")
            test_days = self._positive_int_attr(attrs, "test_days")
            if observed_days <= 0 or test_days <= 0:
                raise ValueError(
                    "Missing solidified paper observed window attrs: "
                    f"dataset_id={attrs.get('dataset_name', 'unknown')} "
                    f"scenario={attrs.get('information_sharing_scenario', 'unknown')} "
                    f"method={attrs.get('method', 'unknown')} "
                    f"paper_split_protocol={attrs.get('paper_split_protocol')} "
                    f"required={required}"
                )
            if "date" not in target_df.columns:
                raise ValueError("paper_observed_sequence requires a date column.")

            ordered = target_df.copy()
            ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
            ordered = ordered.dropna(subset=["date"]).sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
            unique_dates = ordered["date"].drop_duplicates().sort_values()
            total_days = int(observed_days + test_days)
            if len(unique_dates) < total_days:
                raise ValueError(
                    "solidified paper observed window requires enough target dates. "
                    f"dataset_id={attrs.get('dataset_name', 'unknown')} "
                    f"scenario={attrs.get('information_sharing_scenario', 'unknown')} "
                    f"method={attrs.get('method', 'unknown')} "
                    f"required_unique_dates={total_days} actual_unique_dates={len(unique_dates)}"
                )
            eval_dates = unique_dates.iloc[-total_days:]
            observed_dates = eval_dates.iloc[:observed_days]
            observed = ordered[ordered["date"].isin(observed_dates)].copy()
            observed.attrs = attrs.copy()
            return observed, {
                "target_window_source": "solidified_observed_days",
                "target_test_data_excluded": True,
                "target_observed_days_config": int(observed_days),
                "target_test_days_config": int(test_days),
                "target_observed_start_date": observed_dates.min().strftime("%Y-%m-%d"),
                "target_observed_end_date": observed_dates.max().strftime("%Y-%m-%d"),
                "date_alignment_mode": "solidified_observed_days",
                "paper_split_protocol": attrs.get("paper_split_protocol"),
            }

        split_config = target_df.attrs.get("split_config", {}) or {}
        mode = str(target_df.attrs.get("split_mode", "")).strip().lower()
        train_days = int(split_config.get("train_days", 0)) if isinstance(split_config, dict) else 0
        val_days = int(split_config.get("val_days", 0)) if isinstance(split_config, dict) else 0
        observed_days = train_days + val_days

        if mode in {"days", "actual_time_steps"} and observed_days > 0 and "date" in target_df.columns:
            unique_days = int(pd.to_datetime(target_df["date"], errors="coerce").dropna().nunique())
            if unique_days > observed_days:
                target_train, target_val, _ = temporal_split_by_ratio_or_dates(target_df)
                observed = pd.concat([target_train, target_val], axis=0, ignore_index=True)
                observed = observed.sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
                observed.attrs = target_df.attrs.copy()
                return observed, {
                    "target_window_source": "derived_train_val_split",
                    "target_test_data_excluded": True,
                    "target_observed_days_config": int(observed_days),
                    "target_observed_start_date": pd.Timestamp(observed["date"].min()).strftime("%Y-%m-%d"),
                    "target_observed_end_date": pd.Timestamp(observed["date"].max()).strftime("%Y-%m-%d"),
                    "date_alignment_mode": "derived_train_val_split",
                }

        logger.warning(
            "WARNING: observed_days/test_days not found in target attrs; falling back to full target sequence"
        )
        return target_df, {
            "target_window_source": "provided_target_df",
            "target_test_data_excluded": True,
            "date_alignment_mode": "legacy_full_target_sequence",
        }

    def _source_date_alignment_record(
        self,
        source_key: Tuple,
        group: pd.DataFrame,
        target_dates: pd.Series,
        date_alignment_mode: str,
        kept_or_skipped: str,
        skip_reason: str,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> Dict[str, object]:
        work = group.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        dates = work["date"].dropna().drop_duplicates().sort_values()
        overlap_dates = int(dates.isin(target_dates).sum())
        return {
            "dataset_id": target_df.attrs.get("dataset_name") or source_df.attrs.get("dataset_name") or "unknown",
            "scenario": target_df.attrs.get("information_sharing_scenario") or source_df.attrs.get("information_sharing_scenario") or "unknown",
            "method": target_df.attrs.get("method") or source_df.attrs.get("method") or "unknown",
            "target_unique_dates": int(len(target_dates)),
            "source_entity_id": source_key[0] if source_key else "unknown",
            "source_unique_dates": int(len(dates)),
            "min_date": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
            "max_date": dates.max().strftime("%Y-%m-%d") if len(dates) else "",
            "overlap_dates": int(overlap_dates),
            "missing_dates": int(len(target_dates) - overlap_dates),
            "date_alignment_mode": str(date_alignment_mode),
            "source_kept_or_skipped": kept_or_skipped,
            "skip_reason": skip_reason,
        }

    def _resolve_target_series_for_paper_knn(
        self,
        observed_target_df: pd.DataFrame,
        group_cols: Tuple[str, str],
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Reduce a possibly multi-entity target window to one entity-item series.

        paper_observed_sequence compares one flattened source series against one
        flattened target series over the same observed calendar dates.
        """
        for g in group_cols:
            if g not in observed_target_df.columns:
                raise ValueError(f"Missing group column in target: {g}")

        work = observed_target_df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values(list(group_cols) + ["date"]).reset_index(drop=True)
        if work.empty:
            raise ValueError("paper_observed_sequence requires a non-empty target observed window.")

        unique_groups = work[list(group_cols)].drop_duplicates()
        selection_meta: Dict[str, object] = {
            "target_group_count": int(len(unique_groups)),
        }

        if len(unique_groups) == 1:
            series_df = work.copy()
            selection_meta["target_series_mode"] = "single_group"
            selection_meta["target_series_key"] = tuple(unique_groups.iloc[0])
        else:
            attrs = work.attrs
            target_entity_id = attrs.get("target_entity_id")
            target_item_id = attrs.get("target_item_id")
            if target_entity_id is not None:
                mask = work[group_cols[0]].astype(str) == str(target_entity_id)
                if target_item_id is not None:
                    mask &= work[group_cols[1]].astype(str) == str(target_item_id)
                series_df = work[mask].copy()
                if series_df.empty:
                    raise ValueError(
                        "paper_observed_sequence could not resolve target series from attrs "
                        f"target_entity_id={target_entity_id} target_item_id={target_item_id}."
                    )
                selection_meta["target_series_mode"] = "attrs_target_entity"
                first_row = series_df.iloc[0]
                selection_meta["target_series_key"] = tuple(first_row[g] for g in group_cols)
            else:
                first = unique_groups.sort_values(list(group_cols)).iloc[0]
                series_df = work[
                    (work[group_cols[0]] == first[group_cols[0]])
                    & (work[group_cols[1]] == first[group_cols[1]])
                ].copy()
                selection_meta["target_series_mode"] = "first_group"
                selection_meta["target_series_key"] = tuple(first[g] for g in group_cols)

        series_df = series_df.sort_values(["date"]).reset_index(drop=True)
        series_df.attrs = work.attrs.copy()
        return series_df, selection_meta

    def _build_observed_sequence_signatures(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        feature_cols: Sequence[str],
        group_cols: Tuple[str, str],
        static_feature_cols: Sequence[str] | None = None,
        requested_k: int | None = None,
    ) -> Tuple[np.ndarray, List[Tuple], np.ndarray, Dict[str, object]]:
        """Build target/source signatures by flattening the target observed window."""
        logger = _get_logger()
        observed_target_df, target_window_metadata = self._target_observed_window_for_paper_knn(target_df)
        cols = self._validate_feature_columns(observed_target_df, feature_cols)
        self._validate_feature_columns(source_df, cols)

        target_series_df, target_series_metadata = self._resolve_target_series_for_paper_knn(
            observed_target_df,
            group_cols=group_cols,
        )
        target_dates = target_series_df["date"].drop_duplicates().sort_values()
        signature_row_count = int(len(target_series_df))
        target_signature = self._sequence_signature_from_df(
            target_series_df,
            cols,
            expected_rows=signature_row_count,
            static_feature_cols=static_feature_cols,
        )

        for g in group_cols:
            if g not in source_df.columns:
                raise ValueError(f"Missing group column: {g}")

        source_keys: List[Tuple] = []
        signatures: List[np.ndarray] = []
        kept_records: List[Dict[str, object]] = []
        skipped_records: List[Dict[str, object]] = []
        date_alignment_mode = str(target_window_metadata.get("date_alignment_mode", "target_observed_dates"))
        grouped = source_df.groupby(list(group_cols), sort=False)
        for key, group in grouped:
            source_key = tuple(key) if isinstance(key, tuple) else (key,)
            record = self._source_date_alignment_record(
                source_key=source_key,
                group=group,
                target_dates=target_dates,
                date_alignment_mode=date_alignment_mode,
                kept_or_skipped="kept",
                skip_reason="",
                target_df=target_df,
                source_df=source_df,
            )
            if int(record["missing_dates"]) != 0:
                record["source_kept_or_skipped"] = "skipped"
                record["skip_reason"] = "source_missing_target_observed_dates"
                skipped_records.append(record)
                logger.warning("[date-alignment-diag] %s", json.dumps(record, ensure_ascii=True, sort_keys=True))
                continue

            source_keys.append(source_key)
            signatures.append(
                self._sequence_signature_from_df(
                    group,
                    cols,
                    expected_dates=target_dates,
                    expected_rows=signature_row_count,
                    static_feature_cols=static_feature_cols,
                )
            )
            kept_records.append(record)

        if not signatures:
            dataset_id = target_df.attrs.get("dataset_name") or source_df.attrs.get("dataset_name") or "unknown"
            scenario = target_df.attrs.get("information_sharing_scenario") or source_df.attrs.get("information_sharing_scenario") or "unknown"
            method = target_df.attrs.get("method") or source_df.attrs.get("method") or "unknown"
            raise ValueError(
                "No valid sources after paper_observed_sequence alignment: "
                f"dataset_id={dataset_id} scenario={scenario} method={method} "
                f"requested_k={int(requested_k or 0)} skipped_source_count={len(skipped_records)}"
            )

        source_signatures = np.vstack(signatures).astype(np.float64) if signatures else np.empty((0, target_signature.shape[0]))
        metadata = {
            "observed_window_rows": signature_row_count,
            "observed_window_unique_dates": int(len(target_dates)),
            "source_window_alignment": "target_observed_dates",
            "valid_source_count": int(len(source_keys)),
            "skipped_source_count": int(len(skipped_records)),
            "date_alignment_diagnostics": {
                "kept_sources": kept_records,
                "skipped_sources": skipped_records,
            },
            **target_window_metadata,
            **target_series_metadata,
        }
        return target_signature, source_keys, source_signatures, metadata

    def build_target_signature(
        self,
        target_df: pd.DataFrame,
        feature_cols: Sequence[str],
        static_feature_cols: Sequence[str] | None = None,
    ) -> np.ndarray:
        """
        从 target 的短期历史构造相似性签名。

        Args:
            target_df: 某个 target 序列的 DataFrame。
            feature_cols: 用于构造签名的特征列。

        Returns:
            一维 numpy 向量。
        """
        logger = _get_logger()
        cols = self._validate_feature_columns(target_df, feature_cols)
        logger.info("[build_target_signature] Start. rows=%d features=%s", len(target_df), cols)

        signature = self._signature_from_df(target_df, cols, static_feature_cols=static_feature_cols)

        logger.info("[build_target_signature] Finished. signature_dim=%d", signature.shape[0])
        return signature

    def build_source_signatures(
        self,
        source_df: pd.DataFrame,
        feature_cols: Sequence[str],
        group_cols: Tuple[str, str] = ("entity_id", "item_id"),
        static_feature_cols: Sequence[str] | None = None,
    ) -> Tuple[List[Tuple], np.ndarray]:
        """
        对 source pool 按 group_cols 分组并构造签名。

        Args:
            source_df: source pool DataFrame。
            feature_cols: 用于构造签名的特征列。
            group_cols: source 序列分组键。

        Returns:
            source_keys: 每个 source 序列的键。
            source_signatures: shape=(num_sources, signature_dim) 的签名矩阵。
        """
        logger = _get_logger()
        cols = self._validate_feature_columns(source_df, feature_cols)
        for g in group_cols:
            if g not in source_df.columns:
                raise ValueError(f"Missing group column: {g}")

        logger.info(
            "[build_source_signatures] Start. rows=%d group_cols=%s features=%s",
            len(source_df),
            group_cols,
            cols,
        )

        source_keys: List[Tuple] = []
        signatures: List[np.ndarray] = []

        grouped = source_df.groupby(list(group_cols), sort=False)
        for key, group in grouped:
            source_keys.append(tuple(key) if isinstance(key, tuple) else (key,))
            signatures.append(self._signature_from_df(group, cols, static_feature_cols=static_feature_cols))

        if signatures:
            source_signatures = np.vstack(signatures).astype(np.float64)
        else:
            source_signatures = np.empty((0, (len(cols) * 5) + len(static_feature_cols or [])), dtype=np.float64)

        logger.info(
            "[build_source_signatures] Finished. num_sources=%d signature_dim=%d",
            len(source_keys),
            source_signatures.shape[1] if source_signatures.ndim == 2 else 0,
        )
        return source_keys, source_signatures

    def compute_euclidean_distances(
        self,
        target_signature: np.ndarray,
        source_signatures: np.ndarray,
    ) -> np.ndarray:
        """
        计算 target 与所有 source 的欧氏距离。

        Args:
            target_signature: shape=(signature_dim,) 的 target 签名。
            source_signatures: shape=(num_sources, signature_dim) 的 source 签名矩阵。

        Returns:
            shape=(num_sources,) 的距离数组。
        """
        logger = _get_logger()

        tgt = np.asarray(target_signature, dtype=np.float64).reshape(-1)
        src = np.asarray(source_signatures, dtype=np.float64)

        if src.ndim != 2:
            raise ValueError("source_signatures must be a 2D array")
        if src.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        if src.shape[1] != tgt.shape[0]:
            raise ValueError(
                "Signature dimension mismatch: "
                f"target_dim={tgt.shape[0]} source_dim={src.shape[1]}"
            )

        logger.info("[compute_euclidean_distances] Start. num_sources=%d", src.shape[0])
        distances = np.linalg.norm(src - tgt, axis=1)
        logger.info("[compute_euclidean_distances] Finished.")
        return distances

    def compute_source_weights(
        self,
        distances: np.ndarray,
        mode: str = "inverse_distance",
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        根据距离计算权重。

        支持模式：
        - inverse_distance:
          w_i = (1 / (d_i + eps)) / sum_j (1 / (d_j + eps))
        logger.info("[source_selection_debug] include_sales_in_knn=%s", include_sales_in_knn)
        - raw_distance:
          w_i = d_i / sum_j d_j

        注意：raw_distance 模式中，距离越大权重越大，这通常与
        “越相似权重越大”的直觉相反，仅用于对照实验或兼容论文特定设定。

        Args:
            distances: 一维距离数组。
            mode: 权重模式。
            eps: 数值稳定项。

        Returns:
            一维权重数组。
        """
        logger = _get_logger()
        d = np.asarray(distances, dtype=np.float64).reshape(-1)

        if d.size == 0:
            return np.empty((0,), dtype=np.float64)
        if np.any(d < 0):
            raise ValueError("Distances must be non-negative")

        logger.info("[compute_source_weights] Start. mode=%s num_sources=%d", mode, d.size)

        if mode == "inverse_distance":
            scores = 1.0 / (d + eps)
            denom = float(np.sum(scores))
            if denom <= 0:
                weights = np.full_like(d, fill_value=1.0 / d.size)
            else:
                weights = scores / denom
        elif mode == "raw_distance":
            denom = float(np.sum(d))
            if denom <= 0:
                weights = np.full_like(d, fill_value=1.0 / d.size)
            else:
                weights = d / denom
        else:
            raise ValueError("Unsupported mode. Use 'inverse_distance' or 'raw_distance'.")

        logger.info("[compute_source_weights] Finished. weight_sum=%.8f", float(np.sum(weights)))
        return weights

    def _adaptive_source_selection(
        self,
        distances: np.ndarray,
        source_keys: List[Tuple],
        min_sources: int = 1,
        max_sources: int = 3,
        distance_jump_threshold: float = 0.5,
        distance_ratio_threshold: float | None = None,
    ) -> Tuple[np.ndarray, List[int], Dict[str, object]]:
        """
        自适应选择源数量，基于距离分布动态决定。

        策略：
        1. 按距离排序所有源
        2. 从最近的源开始，检查相邻源之间的距离跳跃
        3. 如果距离增量超过阈值，停止添加更多源
        4. 确保源数量在 [min_sources, max_sources] 范围内

        Args:
            distances: 所有源到 target 的距离数组
            source_keys: 源标识列表
            min_sources: 最小源数量
            max_sources: 最大源数量
            distance_jump_threshold: 相邻源距离增量阈值（相对于前一个距离的比例）
                例如 0.5 表示如果下一个源的距离比当前源增加 50% 以上，则停止
            distance_ratio_threshold: 可选的绝对距离比例阈值
                如果提供，选择距离小于 min_distance * (1 + ratio) 的源

        Returns:
            selected_indices: 选中的源索引
            adaptive_meta: 自适应选择的元信息
        """
        logger = _get_logger()

        if len(distances) == 0:
            return np.array([], dtype=np.int64), [], {"adaptive_selection_used": False}

        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        selected_count = min_sources
        min_distance = sorted_distances[0] if len(sorted_distances) > 0 else 0.0

        # 方法1: 基于距离跳跃阈值
        if distance_jump_threshold is not None and distance_jump_threshold > 0:
            for i in range(min_sources, min(max_sources, len(sorted_distances))):
                prev_dist = sorted_distances[i - 1]
                curr_dist = sorted_distances[i]

                # 计算相对增量
                if prev_dist > 0:
                    relative_jump = (curr_dist - prev_dist) / prev_dist
                else:
                    relative_jump = curr_dist if curr_dist > 0 else 0.0

                if relative_jump > distance_jump_threshold:
                    logger.info(
                        "[adaptive_source_selection] Stopping at source %d: "
                        "distance_jump=%.2f%% > threshold=%.2f%%",
                        i + 1, relative_jump * 100, distance_jump_threshold * 100
                    )
                    break
                selected_count = i + 1

        # 方法2: 基于绝对距离比例阈值
        if distance_ratio_threshold is not None and distance_ratio_threshold > 0:
            threshold_distance = min_distance * (1 + distance_ratio_threshold)
            for i in range(min_sources, min(max_sources, len(sorted_distances))):
                if sorted_distances[i] > threshold_distance:
                    logger.info(
                        "[adaptive_source_selection] Stopping at source %d: "
                        "distance=%.4f > threshold=%.4f",
                        i + 1, sorted_distances[i], threshold_distance
                    )
                    break
                selected_count = i + 1

        # 确保在范围内
        selected_count = max(min_sources, min(selected_count, max_sources, len(sorted_distances)))
        selected_indices = sorted_indices[:selected_count]

        adaptive_meta = {
            "adaptive_selection_used": True,
            "min_sources": min_sources,
            "max_sources": max_sources,
            "selected_source_count": int(selected_count),
            "distance_jump_threshold": distance_jump_threshold,
            "distance_ratio_threshold": distance_ratio_threshold,
            "min_distance": float(min_distance),
            "selected_distances": [float(sorted_distances[i]) for i in range(selected_count)],
            "all_distances_sorted": [float(d) for d in sorted_distances[:max_sources + 2]],
        }

        logger.info(
            "[adaptive_source_selection] Selected %d sources (min=%d, max=%d). "
            "distances=%s",
            selected_count, min_sources, max_sources,
            adaptive_meta["selected_distances"]
        )

        return selected_indices, list(range(selected_count)), adaptive_meta

    def select_top_k_sources(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        feature_cols: Sequence[str],
        k: int = 3,
        group_cols: Tuple[str, str] = ("entity_id", "item_id"),
        weight_mode: str = "inverse_distance",
        debug_mode: bool = False,
        include_sales_in_knn: bool = True,
        knn_representation: str | None = None,
        knn_feature_mode: str | None = None,
        adaptive_source_selection: bool = False,
        min_sources: int = 1,
        max_sources: int | None = None,
        distance_jump_threshold: float = 0.5,
        distance_ratio_threshold: float | None = None,
    ) -> Dict[str, object]:
        """
        选择 top-k 相似 source，并返回距离与权重。

        Args:
            target_df: target 序列 DataFrame。
            source_df: source pool DataFrame。
            feature_cols: 签名特征列。
            k: 选择的 source 数量。
            group_cols: source 分组键。
            weight_mode: 权重模式，见 compute_source_weights。
            debug_mode: 是否输出调试/验证日志。
            knn_representation: KNN 向量表示。paper_observed_sequence 为论文对齐默认；
                engineering_summary_stats 保留当前 mean/std/min/max/last 工程摘要。
            knn_feature_mode: KNN 特征模式。
                "paper_available_features_no_ids" / "sales_only_sequence" /
                "engineering_all_numeric"。

        Returns:
                        结构化结果（sources 按距离升序）：
                        {
                            "meta": {
                                "weight_mode": str,
                                "target_signature_dim": int,
                                "feature_cols": list[str],
                                "adaptive_selection_info": dict (if adaptive_source_selection=True)
                            },
                            "sources": [
                                {
                                    "source_key": (...),
                                    "distance": float,
                                    "weight": float
                                },
                                ...
                            ]
                        }

                        新增返回字段用于后续多源迁移模块的调试和追踪：
                        - meta.weight_mode
                        - meta.target_signature_dim
                        - meta.feature_cols
                        - meta.adaptive_selection_info (自适应源选择信息)
        """
        # 处理自适应源选择参数
        effective_max_sources = max_sources if max_sources is not None else k
        logger = _get_logger()
        resolved_knn_representation = self._normalize_knn_representation(knn_representation)
        logger.info(
            "[select_top_k_sources] Start. k=%d weight_mode=%s debug_mode=%s include_sales_in_knn=%s knn_representation=%s knn_feature_mode=%s",
            k,
            weight_mode,
            debug_mode,
            include_sales_in_knn,
            resolved_knn_representation,
            knn_feature_mode,
        )

        if k <= 0:
            raise ValueError("k must be positive")

        resolved_feature_cols, feature_info = self._resolve_source_selection_features(
            source_df=source_df,
            target_df=target_df,
            feature_cols=feature_cols,
            include_sales_in_knn=include_sales_in_knn,
            knn_feature_mode=knn_feature_mode,
        )
        static_feature_cols = self._resolve_signature_static_features(source_df=source_df, target_df=target_df)

        logger.info(
            "[source_selection_features] include_sales_in_knn=%s requested=%s resolved=%s feature_dim=%d contains_sales=%s",
            include_sales_in_knn,
            list(feature_cols),
            resolved_feature_cols,
            len(resolved_feature_cols),
            "sales" in resolved_feature_cols,
        )
        logger.info(
            "[source_selection_features] missing_in_source=%s missing_in_target=%s excluded_by_rule=%s",
            feature_info.get("missing_in_source", []),
            feature_info.get("missing_in_target", []),
            feature_info.get("excluded_by_rule", []),
        )

        representation_metadata: Dict[str, object] = {}
        if resolved_knn_representation == KNN_REPRESENTATION_PAPER_OBSERVED_SEQUENCE:
            target_signature, source_keys, source_signatures, representation_metadata = (
                self._build_observed_sequence_signatures(
                    target_df=target_df,
                    source_df=source_df,
                    feature_cols=resolved_feature_cols,
                    group_cols=group_cols,
                    static_feature_cols=static_feature_cols,
                    requested_k=k,
                )
            )
            signature_component_breakdown = {
                "time_series_sequence": list(resolved_feature_cols),
                "static_metadata": list(static_feature_cols),
            }
        else:
            target_signature = self.build_target_signature(
                target_df,
                resolved_feature_cols,
                static_feature_cols=static_feature_cols,
            )
            source_keys, source_signatures = self.build_source_signatures(
                source_df=source_df,
                feature_cols=resolved_feature_cols,
                group_cols=group_cols,
                static_feature_cols=static_feature_cols,
            )
            signature_component_breakdown = {
                "time_series_stats": [f"{col}:mean/std/min/max/last" for col in resolved_feature_cols],
                "static_metadata": list(static_feature_cols),
            }

        logger.info(
            "[source_selection_features] knn_representation=%s target_signature_dim=%d source_signature_dim=%d",
            resolved_knn_representation,
            int(target_signature.shape[0]),
            int(source_signatures.shape[1]) if source_signatures.ndim == 2 else 0,
        )

        result_payload: Dict[str, object] = {
            "meta": {
                "weight_mode": weight_mode,
                "distance_metric": "euclidean",
                "knn_representation": resolved_knn_representation,
                "target_signature_dim": int(target_signature.shape[0]),
                "feature_cols": list(resolved_feature_cols),
                "signature_static_feature_cols": list(static_feature_cols),
                "signature_component_breakdown": signature_component_breakdown,
                **representation_metadata,
                "include_sales_in_knn": bool(include_sales_in_knn),
                "contains_sales": bool("sales" in resolved_feature_cols),
                "knn_feature_mode": str(feature_info.get("knn_feature_mode", "unknown")),
                "requested_feature_cols": list(feature_cols),
                "missing_in_source": list(feature_info.get("missing_in_source", [])),
                "missing_in_target": list(feature_info.get("missing_in_target", [])),
                "excluded_by_rule": list(feature_info.get("excluded_by_rule", [])),
            },
            "sources": [],
        }

        if len(source_keys) == 0:
            logger.info("[select_top_k_sources] No source sequences found. Return empty result.")
            return result_payload

        distances = self.compute_euclidean_distances(target_signature, source_signatures)

        # 自适应源选择或固定 top-k 选择
        adaptive_selection_info: Dict[str, object] = {"adaptive_selection_used": False}

        if adaptive_source_selection:
            selected_indices, _, adaptive_selection_info = self._adaptive_source_selection(
                distances=distances,
                source_keys=source_keys,
                min_sources=min_sources,
                max_sources=effective_max_sources,
                distance_jump_threshold=distance_jump_threshold,
                distance_ratio_threshold=distance_ratio_threshold,
            )
            logger.info(
                "[select_top_k_sources] Adaptive selection: selected %d sources",
                len(selected_indices)
            )
        else:
            top_k = min(k, len(source_keys))
            sorted_indices = np.argsort(distances)
            selected_indices = sorted_indices[:top_k]

        selected_distances = distances[selected_indices]
        selected_weights = self.compute_source_weights(selected_distances, mode=weight_mode)

        # 添加自适应选择信息到元数据
        result_payload["meta"]["adaptive_selection_info"] = adaptive_selection_info
        result_payload["meta"]["requested_k"] = int(k)
        result_payload["meta"]["effective_k"] = int(len(selected_indices))
        result_payload["meta"]["valid_source_count"] = int(len(source_keys))
        result_payload["meta"]["skipped_source_count"] = int(
            representation_metadata.get("skipped_source_count", 0)
        )
        result_payload["meta"]["date_alignment_mode"] = str(
            representation_metadata.get("date_alignment_mode", "")
        )
        if len(selected_indices) < int(k):
            diag = {
                "dataset_id": target_df.attrs.get("dataset_name") or source_df.attrs.get("dataset_name") or "unknown",
                "scenario": target_df.attrs.get("information_sharing_scenario") or source_df.attrs.get("information_sharing_scenario") or "unknown",
                "method": target_df.attrs.get("method") or source_df.attrs.get("method") or "unknown",
                "requested_k": int(k),
                "effective_k": int(len(selected_indices)),
                "valid_source_count": int(len(source_keys)),
                "skipped_source_count": int(representation_metadata.get("skipped_source_count", 0)),
                "reason": "valid sources fewer than requested k after paper_observed_sequence alignment",
            }
            logger.warning("[date-alignment-diag] %s", json.dumps(diag, ensure_ascii=True, sort_keys=True))

        results: List[Dict[str, object]] = []
        for rank, (idx, w) in enumerate(zip(selected_indices, selected_weights), start=1):
            results.append(
                {
                    "source_rank": int(rank),
                    "source_key": source_keys[int(idx)],
                    "distance": float(distances[int(idx)]),
                    "weight": float(w),
                    "knn_representation": resolved_knn_representation,
                }
            )

        # 按距离从小到大排序，确保输出稳定。
        results = sorted(results, key=lambda x: x["distance"])
        for rank, row in enumerate(results, start=1):
            row["source_rank"] = int(rank)

        if debug_mode:
            self._log_debug_selection_details(
                target_df=target_df,
                source_df=source_df,
                group_cols=group_cols,
                include_sales_in_knn=include_sales_in_knn,
                resolved_feature_cols=resolved_feature_cols,
                source_keys=source_keys,
                target_signature=target_signature,
                source_signatures=source_signatures,
                distances=distances,
                results=results,
                knn_representation=resolved_knn_representation,
            )

        logger.info(
            "[select_top_k_sources] Finished. selected=%d weight_sum=%.8f",
            len(results),
            float(sum(float(r["weight"]) for r in results)),
        )
        result_payload["sources"] = results
        return result_payload
