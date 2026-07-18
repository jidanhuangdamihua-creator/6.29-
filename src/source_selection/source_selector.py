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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_processing.data_preprocessing import infer_source_selection_feature_columns
from src.constants import D4_D6_RUNTIME_KNN_PROTOCOL_VERSION
from src.protocols.candidate_pool import (
    build_consumer_fingerprint,
    build_source_pool_fingerprint,
    select_daily_sequence_sources,
)
from src.protocols.d2_source_calendarization import build_d2_sealed_identity
from src.protocols.experiment_protocol import (
    PROTOCOL_VERSION,
    ProtocolViolation,
    get_experiment_protocol,
)
from src.protocols.knn_frames import get_configured_knn_frame
from src.protocols.provenance import (
    build_cnn_tensor_provenance,
    extract_selected_source_slices,
    validate_cnn_tensor_provenance,
)

try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None


LOGGER_NAME = "experiment"


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
    def _declares_d1_d6(frame: pd.DataFrame) -> bool:
        dataset_name = str(frame.attrs.get("dataset_name", "")).strip().lower()
        return dataset_name in {f"dataset{number}" for number in range(1, 7)} or bool(
            frame.attrs.get("protocol_dataset_id")
        )

    def _select_with_shared_protocol(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        *,
        k: int,
        group_cols: Sequence[str],
        model_feature_cols: Sequence[str],
        weight_mode: str,
        include_sales_in_knn: bool,
    ) -> Dict[str, object]:
        if weight_mode != "inverse_distance":
            raise ProtocolViolation("formal D1-D6 selection requires inverse_distance weights")
        if not include_sales_in_knn:
            raise ProtocolViolation("formal D1-D6 selection requires the daily sales sequence")
        required_attrs = (
            "protocol_dataset_id",
            "protocol_scenario",
            "protocol_target_key",
            "protocol_candidate_keys",
            "protocol_group_cols",
            "knn_observed_start",
            "knn_observed_end",
            "source_observation_cutoff",
        )
        for role, frame in (("target", target_df), ("source", source_df)):
            missing = [name for name in required_attrs if name not in frame.attrs]
            if missing:
                raise ProtocolViolation(
                    f"{role} is missing shared protocol metadata: {missing}"
                )
            if frame.attrs.get("protocol_version") != PROTOCOL_VERSION:
                raise ProtocolViolation(
                    f"{role} protocol_version must be {PROTOCOL_VERSION}"
                )
        for name in required_attrs:
            if target_df.attrs[name] != source_df.attrs[name]:
                raise ProtocolViolation(f"target/source shared protocol metadata mismatch: {name}")
        configured_group_cols = tuple(target_df.attrs["protocol_group_cols"])
        if tuple(group_cols) != configured_group_cols:
            raise ProtocolViolation(
                f"selector group_cols differ from protocol: {tuple(group_cols)!r} != {configured_group_cols!r}"
            )
        protocol = get_experiment_protocol(target_df.attrs["protocol_dataset_id"])
        d2_identity_fields = (
            "d2_source_calendarization_rule_version",
            "d2_source_authority_digest",
            "d2_consumer_frame_fingerprint",
        )
        d2_sealed_identity = None
        if protocol.dataset_id == "D2":
            missing_d2_identity = [
                name
                for name in d2_identity_fields
                if name not in source_df.attrs or name not in target_df.attrs
            ]
            if missing_d2_identity:
                raise ProtocolViolation(
                    "D2 selector is missing calendarization identity: "
                    f"{missing_d2_identity!r}"
                )
            for name in d2_identity_fields:
                if source_df.attrs[name] != target_df.attrs[name]:
                    raise ProtocolViolation(
                        f"D2 source/target calendarization identity mismatch: {name}"
                    )
        knn_target_df = get_configured_knn_frame(target_df, "target")
        knn_source_df = get_configured_knn_frame(source_df, "source")
        for identity_field in d2_identity_fields:
            if identity_field in source_df.attrs:
                knn_source_df.attrs[identity_field] = source_df.attrs[identity_field]
            if identity_field in target_df.attrs:
                knn_target_df.attrs[identity_field] = target_df.attrs[identity_field]
        observed_start = pd.Timestamp(target_df.attrs["knn_observed_start"]).normalize()
        observed_end = pd.Timestamp(target_df.attrs["knn_observed_end"]).normalize()
        for role, frame in (("target", knn_target_df), ("source", knn_source_df)):
            dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            if dates.isna().any():
                raise ProtocolViolation(f"{role} configured KNN frame contains invalid dates")
            if not dates.between(observed_start, observed_end, inclusive="both").all():
                raise ProtocolViolation(
                    f"{role} configured KNN frame contains dates outside the protocol window"
                )
        prepared_pool = source_df.attrs.get("prepared_daily_sequence_pool")
        result = select_daily_sequence_sources(
            target_df=knn_target_df,
            source_df=knn_source_df,
            prepared_pool=prepared_pool,
            protocol=protocol,
            scenario=target_df.attrs["protocol_scenario"],
            target_key=target_df.attrs["protocol_target_key"],
            candidate_keys=target_df.attrs["protocol_candidate_keys"],
            group_cols=configured_group_cols,
            observed_start=target_df.attrs["knn_observed_start"],
            feature_cols=("sales",),
            k=k,
        )
        provenance_source_df = source_df
        if prepared_pool is not None:
            if tuple(model_feature_cols) != ("sales",):
                raise ProtocolViolation(
                    "prepared-pool source provenance currently supports sales-only preflight"
                )
            provenance_source_df = prepared_pool.selected_sales_frame(
                result.ordered_source_keys
            )
        source_slices = extract_selected_source_slices(
            result,
            provenance_source_df,
            training_start=result.observed_start,
            training_end=result.source_observation_cutoff,
            model_feature_cols=model_feature_cols,
        )
        tensor_provenance = tuple(
            build_cnn_tensor_provenance(
                source_slice,
                window_size=int(target_df.attrs.get("model_window_size", 10)),
                horizon=int(target_df.attrs.get("model_horizon", 1)),
                label_col="sales",
            )
            for source_slice in source_slices
        )
        for provenance in tensor_provenance:
            validate_cnn_tensor_provenance(
                provenance,
                provenance_source_df,
                group_cols=configured_group_cols,
            )
        if protocol.dataset_id == "D2":
            d2_sealed_identity = build_d2_sealed_identity(
                rule_version=source_df.attrs["d2_source_calendarization_rule_version"],
                source_authority_digest=source_df.attrs["d2_source_authority_digest"],
                consumer_frame_fingerprint=source_df.attrs[
                    "d2_consumer_frame_fingerprint"
                ],
                candidate_pool_digest=result.candidate_pool_digest,
                selection_result_digest=result.selection_result_digest,
            )
        sources = [
            {
                "source_rank": entry.rank,
                "source_key": entry.source_key,
                "distance": entry.distance,
                "weight": entry.weight,
                "tie_group": entry.tie_group,
                "date_start": entry.observed_start,
                "date_end": entry.observed_end,
            }
            for entry in result.entries
        ]
        excluded = [dict(item) for item in result.excluded_candidates]
        eligible_candidate_keys = list(target_df.attrs["protocol_candidate_keys"])
        excluded_candidate_keys = {
            tuple(item["source_key"])
            for item in excluded
        }
        valid_30d_candidate_keys = [
            key
            for key in eligible_candidate_keys
            if tuple(key) not in excluded_candidate_keys
        ]
        source_pool_fingerprint = build_source_pool_fingerprint(
            protocol_version=protocol.protocol_version,
            dataset_id=protocol.dataset_id,
            scenario=target_df.attrs["protocol_scenario"],
            target_key=target_df.attrs["protocol_target_key"],
            group_cols=configured_group_cols,
            candidate_keys=eligible_candidate_keys,
        )
        consumer_fingerprint = build_consumer_fingerprint(
            protocol_version=protocol.protocol_version,
            dataset_id=protocol.dataset_id,
            scenario=target_df.attrs["protocol_scenario"],
            target_key=target_df.attrs["protocol_target_key"],
            source_pool_fingerprint=source_pool_fingerprint,
            candidate_pool_digest=result.candidate_pool_digest,
            selection_result_digest=result.selection_result_digest,
            ordered_top_k=sources,
        )
        candidate_digest_input = dict(result.candidate_pool_digest_input)
        meta = {
            "selection_authority": "shared_protocol",
            "selection_path": "shared_protocol",
            "protocol_track": protocol.track,
            "protocol_version": protocol.protocol_version,
            "weight_mode": protocol.weight_mode,
            "distance_metric": "euclidean",
            "group_cols": list(configured_group_cols),
            "target_signature_dim": 30,
            "feature_cols": ["sales"],
            "requested_feature_cols": ["sales"],
            "representation": protocol.knn_representation,
            "knn_representation": protocol.knn_representation,
            "scaling": "global_minmax_legal_observed_values",
            "scaler_fit_scope": "target_and_candidate_legal_observed_values",
            "knn_observed_start": result.observed_start,
            "knn_observed_end": result.observed_end,
            "origin": result.observed_end,
            "protocol_observed_start": target_df.attrs["protocol_observed_start"],
            "protocol_observed_days": target_df.attrs["protocol_observed_days"],
            "observed_days": target_df.attrs["protocol_observed_days"],
            "boundary": "inclusive",
            "target_observed_start": result.observed_start,
            "target_observed_end": result.observed_end,
            "source_observation_cutoff": result.source_observation_cutoff,
            "target_test_excluded": True,
            "source_future_excluded": True,
            "source_alignment_mode": "exact_knn_observed_dates",
            "candidate_pool_digest": result.candidate_pool_digest,
            "candidate_pool_digest_input": candidate_digest_input,
            "selection_result_digest": result.selection_result_digest,
            "selection_digest": result.selection_result_digest,
            "source_frame_min_date": source_df.attrs["source_frame_min_date"],
            "source_frame_max_date": source_df.attrs["source_frame_max_date"],
            "target_frame_min_date": target_df.attrs["target_frame_min_date"],
            "target_frame_max_date": target_df.attrs["target_frame_max_date"],
            "source_frame_digest": candidate_digest_input.get(
                "source_frame_digest",
                source_df.attrs.get("source_frame_digest", ""),
            ),
            "target_frame_digest": candidate_digest_input.get(
                "target_frame_digest",
                target_df.attrs.get("target_frame_digest", ""),
            ),
            "source_pool_fingerprint": source_pool_fingerprint,
            "consumer_fingerprint": consumer_fingerprint,
            "selected_sources_runtime": list(sources),
            "source_skip_diagnostics": excluded,
            "candidate_source_count": len(target_df.attrs["protocol_candidate_keys"]),
            "valid_source_count": len(target_df.attrs["protocol_candidate_keys"]) - len(excluded),
            "eligible_candidate_keys": eligible_candidate_keys,
            "valid_30d_candidate_keys": valid_30d_candidate_keys,
            "eligible_candidate_count": len(eligible_candidate_keys),
            "valid_30d_candidate_count": len(valid_30d_candidate_keys),
            "selected_count": len(sources),
            "observed_days": len(result.entries[0].raw_vector),
            "skipped_source_count": len(excluded),
            "requested_k": int(k),
            "effective_k": int(k),
            "scaler_min": result.scaler_min,
            "scaler_max": result.scaler_max,
            "cnn_provenance_validated": True,
            "cnn_provenance_source_keys": [item.source_key for item in source_slices],
            "cnn_provenance_sample_counts": [
                int(item.input_tensor.shape[0]) for item in tensor_provenance
            ],
        }
        if protocol.dataset_id == "D2":
            meta.update(
                {
                    "d2_source_calendarization_rule_version": source_df.attrs[
                        "d2_source_calendarization_rule_version"
                    ],
                    "d2_source_authority_digest": source_df.attrs[
                        "d2_source_authority_digest"
                    ],
                    "d2_consumer_frame_fingerprint": source_df.attrs[
                        "d2_consumer_frame_fingerprint"
                    ],
                    "d2_sealed_identity": d2_sealed_identity,
                }
            )
        return {"meta": meta, "sources": sources}

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
    ) -> Tuple[List[str], Dict[str, object]]:
        """Resolve source-selection features without widening explicit feature lists."""
        requested = list(feature_cols) if feature_cols is not None else []
        if requested:
            resolved = self._validate_feature_columns(source_df, requested)
            self._validate_feature_columns(target_df, resolved)
            info: Dict[str, object] = {
                "selected_features": list(resolved),
                "requested_feature_cols": list(requested),
                "missing_in_source": [],
                "missing_in_target": [],
                "excluded_by_rule": [],
                "include_sales_in_knn": bool(include_sales_in_knn),
                "knn_feature_mode": "explicit_feature_cols",
                "feature_resolution_source": "explicit_feature_cols",
            }
            try:
                runtime_info = infer_source_selection_feature_columns(
                    source_df=source_df,
                    target_df=target_df,
                    candidate_cols=requested,
                    include_sales_in_knn=include_sales_in_knn,
                )
                info["runtime_inferred_features"] = list(runtime_info.get("selected_features", []))
                info["runtime_knn_feature_mode"] = runtime_info.get("knn_feature_mode", "")
                info["runtime_excluded_by_rule"] = list(runtime_info.get("excluded_by_rule", []))
            except Exception as exc:  # diagnostics only; explicit features remain authoritative
                info["runtime_infer_error"] = f"{type(exc).__name__}: {exc}"
            return resolved, info

        info = infer_source_selection_feature_columns(
            source_df=source_df,
            target_df=target_df,
            candidate_cols=[],
            include_sales_in_knn=include_sales_in_knn,
        )
        resolved = list(info.get("selected_features", []))
        if not resolved:
            raise ValueError("No resolved source-selection features.")
        info["feature_resolution_source"] = "runtime_infer"
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

    @staticmethod
    def _runtime_protocol_requested(target_df: pd.DataFrame, source_df: pd.DataFrame) -> bool:
        """Return whether either frame declares the D4-D6 runtime KNN protocol."""
        attrs = (target_df.attrs, source_df.attrs)
        return any(
            frame_attrs.get("selection_authority") == "runtime"
            or frame_attrs.get("protocol_version") == D4_D6_RUNTIME_KNN_PROTOCOL_VERSION
            for frame_attrs in attrs
        )

    @staticmethod
    def _strict_digest(payload: object) -> str:
        """Return a deterministic SHA-256 digest for JSON-compatible diagnostics."""
        def default(value: object) -> object:
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.bool_):
                return bool(value)
            raise TypeError(f"Unsupported digest value: {type(value).__name__}")

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=default,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _runtime_aligned_signatures(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        feature_cols: Sequence[str],
        group_cols: Tuple[str, str],
        static_feature_cols: Sequence[str] | None,
    ) -> Tuple[np.ndarray, List[Tuple], np.ndarray, Dict[str, Any]]:
        """Build D4-D6 signatures on the exact shared 30-day observed calendar."""
        required_attrs = (
            "selection_authority",
            "protocol_version",
            "target_observed_start",
            "target_observed_end",
            "source_history_start",
            "source_history_end",
            "target_test_excluded",
            "source_future_excluded",
            "source_alignment_mode",
            "representation",
            "scaling",
            "scaler_fit_scope",
        )
        for role, frame in (("target", target_df), ("source", source_df)):
            missing = [key for key in required_attrs if key not in frame.attrs]
            if missing:
                raise ValueError(
                    f"Missing D4-D6 runtime KNN metadata on {role}: {missing}"
                )
            if "date" not in frame.columns:
                raise ValueError(f"D4-D6 runtime KNN {role} dataframe requires date column")

        for key in required_attrs:
            if target_df.attrs[key] != source_df.attrs[key]:
                raise ValueError(f"D4-D6 runtime KNN metadata mismatch for {key}")
        if target_df.attrs["selection_authority"] != "runtime":
            raise ValueError("D4-D6 runtime KNN selection_authority must be 'runtime'")
        if target_df.attrs["protocol_version"] != D4_D6_RUNTIME_KNN_PROTOCOL_VERSION:
            raise ValueError(
                "Unsupported D4-D6 runtime KNN protocol_version: "
                f"{target_df.attrs['protocol_version']!r}"
            )

        target_observed_start = pd.Timestamp(target_df.attrs["target_observed_start"]).normalize()
        target_observed_end = pd.Timestamp(target_df.attrs["target_observed_end"]).normalize()
        source_history_start = pd.Timestamp(source_df.attrs["source_history_start"]).normalize()
        source_history_end = pd.Timestamp(source_df.attrs["source_history_end"]).normalize()
        if (target_observed_end - target_observed_start).days != 29:
            raise ValueError("D4-D6 runtime KNN observed window must contain exactly 30 calendar days")
        if source_history_end != target_observed_end:
            raise ValueError("D4-D6 runtime KNN source_history_end must equal target_observed_end")
        if (source_history_end - source_history_start).days != 299:
            raise ValueError("D4-D6 runtime KNN source history must contain exactly 300 calendar days")

        required_dates = pd.DatetimeIndex(
            pd.date_range(target_observed_start, target_observed_end, freq="D")
        )
        target = target_df.copy()
        target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
        if target["date"].isna().any():
            raise ValueError("D4-D6 runtime KNN target dataframe contains invalid date values")
        target_observed = target[target["date"].isin(required_dates)].sort_values("date")
        if target_observed["date"].duplicated().any():
            raise ValueError("D4-D6 runtime KNN target observed window contains duplicate dates")
        target_dates = pd.DatetimeIndex(target_observed["date"])
        if not target_dates.equals(required_dates):
            missing = required_dates.difference(target_dates).strftime("%Y-%m-%d").tolist()
            raise ValueError(
                "D4-D6 runtime KNN target does not cover all observed dates: "
                f"missing_dates={missing}"
            )

        target_signature = self._signature_from_df(
            target_observed,
            feature_cols,
            static_feature_cols=static_feature_cols,
        )
        source_keys: List[Tuple] = []
        signatures: List[np.ndarray] = []
        skipped: List[Dict[str, Any]] = []

        grouped = source_df.groupby(list(group_cols), sort=False)
        for raw_key, raw_group in grouped:
            source_key = tuple(raw_key) if isinstance(raw_key, tuple) else (raw_key,)
            group = raw_group.copy()
            group["date"] = pd.to_datetime(group["date"], errors="coerce").dt.normalize()
            if group["date"].isna().any():
                skipped.append(
                    {"source_key": source_key, "reason": "invalid_date_values", "missing_dates": []}
                )
                continue
            history = group[
                group["date"].between(source_history_start, source_history_end, inclusive="both")
            ]
            aligned = history[history["date"].isin(required_dates)].sort_values("date")
            aligned_dates = pd.DatetimeIndex(aligned["date"].drop_duplicates())
            missing_dates = required_dates.difference(aligned_dates).strftime("%Y-%m-%d").tolist()
            if missing_dates:
                skipped.append(
                    {
                        "source_key": source_key,
                        "reason": "missing_target_observed_dates",
                        "missing_dates": missing_dates,
                    }
                )
                continue
            if aligned["date"].duplicated().any():
                skipped.append(
                    {
                        "source_key": source_key,
                        "reason": "duplicate_target_observed_dates",
                        "missing_dates": [],
                    }
                )
                continue
            source_keys.append(source_key)
            signatures.append(
                self._signature_from_df(
                    aligned,
                    feature_cols,
                    static_feature_cols=static_feature_cols,
                )
            )

        if not signatures:
            raise ValueError(
                "No valid sources after exact_target_observed_dates alignment: "
                f"source_skip_diagnostics={skipped}"
            )
        source_signatures = np.vstack(signatures).astype(np.float64)
        digest_keys = sorted(source_keys, key=lambda key: json.dumps(key, default=str))
        metadata = {
            key: target_df.attrs[key]
            for key in required_attrs
        }
        for key in (
            "target_observed_start",
            "target_observed_end",
            "source_history_start",
            "source_history_end",
        ):
            metadata[key] = pd.Timestamp(metadata[key]).strftime("%Y-%m-%d")
        metadata.update(
            {
                "source_skip_diagnostics": skipped,
                "candidate_source_count": int(len(source_keys)),
                "skipped_source_count": int(len(skipped)),
                "candidate_pool_digest": self._strict_digest(digest_keys),
            }
        )
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

        Returns:
                        结构化结果（sources 按距离升序）：
                        {
                            "meta": {
                                "weight_mode": str,
                                "target_signature_dim": int,
                                "feature_cols": list[str]
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
        """
        logger = _get_logger()
        logger.info(
            "[select_top_k_sources] Start. k=%d weight_mode=%s debug_mode=%s include_sales_in_knn=%s",
            k,
            weight_mode,
            debug_mode,
            include_sales_in_knn,
        )

        if k <= 0:
            raise ValueError("k must be positive")

        shared_protocol = (
            target_df.attrs.get("protocol_version") == PROTOCOL_VERSION
            or source_df.attrs.get("protocol_version") == PROTOCOL_VERSION
        )
        if shared_protocol:
            return self._select_with_shared_protocol(
                target_df,
                source_df,
                k=k,
                group_cols=group_cols,
                model_feature_cols=feature_cols,
                weight_mode=weight_mode,
                include_sales_in_knn=include_sales_in_knn,
            )
        if self._declares_d1_d6(target_df) or self._declares_d1_d6(source_df):
            raise ProtocolViolation(
                "D1-D6 selection requires shared protocol metadata; legacy fallback is forbidden"
            )

        resolved_feature_cols, feature_info = self._resolve_source_selection_features(
            source_df=source_df,
            target_df=target_df,
            feature_cols=feature_cols,
            include_sales_in_knn=include_sales_in_knn,
        )
        static_feature_cols = self._resolve_signature_static_features(source_df=source_df, target_df=target_df)

        logger.info(
            "[source_selection_features] include_sales_in_knn=%s requested=%s resolved=%s feature_dim=%d signature_dim=%d contains_sales=%s",
            include_sales_in_knn,
            list(feature_cols),
            resolved_feature_cols,
            len(resolved_feature_cols),
            len(resolved_feature_cols) * 5,
            "sales" in resolved_feature_cols,
        )
        logger.info(
            "[source_selection_features] missing_in_source=%s missing_in_target=%s excluded_by_rule=%s",
            feature_info.get("missing_in_source", []),
            feature_info.get("missing_in_target", []),
            feature_info.get("excluded_by_rule", []),
        )

        runtime_metadata: Dict[str, Any] = {}
        if self._runtime_protocol_requested(target_df, source_df):
            target_signature, source_keys, source_signatures, runtime_metadata = (
                self._runtime_aligned_signatures(
                    target_df=target_df,
                    source_df=source_df,
                    feature_cols=resolved_feature_cols,
                    group_cols=group_cols,
                    static_feature_cols=static_feature_cols,
                )
            )
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

        result_payload: Dict[str, object] = {
            "meta": {
                "weight_mode": weight_mode,
                "distance_metric": "euclidean",
                "group_cols": list(group_cols),
                "target_signature_dim": int(target_signature.shape[0]),
                "feature_cols": list(resolved_feature_cols),
                "signature_static_feature_cols": list(static_feature_cols),
                "signature_component_breakdown": {
                    "time_series_stats": [f"{col}:mean/std/min/max/last" for col in resolved_feature_cols],
                    "static_metadata": list(static_feature_cols),
                },
                "include_sales_in_knn": bool(include_sales_in_knn),
                "contains_sales": bool("sales" in resolved_feature_cols),
                "knn_feature_mode": str(feature_info.get("knn_feature_mode", "")),
                "feature_resolution_source": str(feature_info.get("feature_resolution_source", "")),
                "runtime_inferred_features": list(feature_info.get("runtime_inferred_features", [])),
                "runtime_knn_feature_mode": str(feature_info.get("runtime_knn_feature_mode", "")),
                "runtime_excluded_by_rule": list(feature_info.get("runtime_excluded_by_rule", [])),
                "runtime_infer_error": str(feature_info.get("runtime_infer_error", "")),
                "requested_feature_cols": list(feature_cols),
                "missing_in_source": list(feature_info.get("missing_in_source", [])),
                "missing_in_target": list(feature_info.get("missing_in_target", [])),
                "excluded_by_rule": list(feature_info.get("excluded_by_rule", [])),
                **runtime_metadata,
            },
            "sources": [],
        }

        if len(source_keys) == 0:
            logger.info("[select_top_k_sources] No source sequences found. Return empty result.")
            return result_payload

        distances = self.compute_euclidean_distances(target_signature, source_signatures)

        top_k = min(k, len(source_keys))
        sorted_indices = np.argsort(distances)
        selected_indices = sorted_indices[:top_k]

        selected_distances = distances[selected_indices]
        selected_weights = self.compute_source_weights(selected_distances, mode=weight_mode)

        results: List[Dict[str, object]] = []
        for rank, (idx, w) in enumerate(zip(selected_indices, selected_weights), start=1):
            results.append(
                {
                    "source_rank": int(rank),
                    "source_key": source_keys[int(idx)],
                    "distance": float(distances[int(idx)]),
                    "weight": float(w),
                }
            )

        # 按距离从小到大排序，确保输出稳定。
        results = sorted(results, key=lambda x: x["distance"])
        for rank, row in enumerate(results, start=1):
            row["source_rank"] = int(rank)

        if runtime_metadata:
            result_payload["meta"]["selected_sources_runtime"] = list(results)
            result_payload["meta"]["selection_result_digest"] = self._strict_digest(results)

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
            )

        logger.info(
            "[select_top_k_sources] Finished. selected=%d weight_sum=%.8f",
            len(results),
            float(sum(float(r["weight"]) for r in results)),
        )
        result_payload["sources"] = results
        return result_payload
