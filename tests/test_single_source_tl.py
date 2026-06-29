"""
Test script for Module 4: Single-Source Transfer Learning (SS-TL)

最小可运行测试：使用 Dataset1 验证 SS-TL 完整流程。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import Config
from src.utils.environment import setup_logging

from src.data_processing.data_preprocessing import (
    build_source_target_split,
    build_tabular_sequence,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.transfer_methods.single_source_tl import (
    build_target_model_from_source,
    evaluate_regression_model,
    fine_tune_target_model,
    train_source_model,
)


def main() -> None:
    """运行 SS-TL 最小流程测试。"""
    setup_logging(log_level="INFO", log_file=None)

    # ---- 1. 加载数据 ----
    config = Config(config_file="config.yaml", supply_chain_file="supply_chain.yaml", verbose=True)

    csv_path = str(Path(str(config.dataset.path)) / "train.csv")
    df = load_dataset(dataset_name="Dataset1", data_path=csv_path)
    df = extract_datetime_features(df)

    source_df, target_df = build_source_target_split(df, config)

    window_size = 10
    horizon = 1

    # ---- 2. 准备 source 数据（取第一个 source item） ----
    first_source_item = sorted(source_df["item_id"].unique())[0]
    single_source_df = source_df[source_df["item_id"] == first_source_item].copy()
    single_source_df.attrs["split_role"] = "source"
    single_source_df.attrs["split_mode"] = "ratio"
    single_source_df.attrs["split_config"] = {
        "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
    }

    src_train, src_val, src_test = temporal_split_by_ratio_or_dates(single_source_df)
    src_train, src_val, src_test, _, src_feat_cols = normalize_features(src_train, src_val, src_test)

    X_source, y_source = build_tabular_sequence(src_train, horizon=horizon, window_size=window_size)
    X_source = to_cnn_tensor(X_source)

    # ---- 3. 准备 target 数据 ----
    tgt_train, tgt_val, tgt_test = temporal_split_by_ratio_or_dates(target_df)
    tgt_train, tgt_val, tgt_test, _, tgt_feat_cols = normalize_features(tgt_train, tgt_val, tgt_test)

    X_target_train, y_target_train = build_tabular_sequence(tgt_train, horizon=horizon, window_size=window_size)
    X_target_val, y_target_val = build_tabular_sequence(tgt_val, horizon=horizon, window_size=window_size)
    X_target_test, y_target_test = build_tabular_sequence(tgt_test, horizon=horizon, window_size=window_size)

    X_target_train = to_cnn_tensor(X_target_train)
    X_target_val = to_cnn_tensor(X_target_val)
    X_target_test = to_cnn_tensor(X_target_test)

    # 确保 source 与 target 特征维度一致
    assert X_source.shape[1:] == X_target_train.shape[1:], (
        f"Shape mismatch: source {X_source.shape[1:]} vs target {X_target_train.shape[1:]}"
    )
    input_shape = X_source.shape[1:]  # (window_size, num_features)

    # ---- 4. SS-TL 流程 ----
    # 4a. 训练 source 模型
    source_model = train_source_model(
        X_source, y_source, input_shape=input_shape, epochs=3, batch_size=16,
    )
    print("Source Model Trained Successfully")

    # 4b. 构建 target 模型（迁移权重 + 冻结层）
    target_model, frozen_names = build_target_model_from_source(
        source_model, input_shape=input_shape, freeze_first_n_layers=4,
    )
    print("Target Model Built Successfully")
    print(f"Frozen Layers: {frozen_names}")

    # 4c. 微调 target 模型
    target_model = fine_tune_target_model(
        target_model,
        X_target_train, y_target_train,
        X_target_val=X_target_val, y_target_val=y_target_val,
        epochs=3, batch_size=16,
    )
    print("Fine-tuning Completed")

    # 4d. 评估
    results = evaluate_regression_model(target_model, X_target_test, y_target_test)
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Prediction Shape: {results['y_pred_shape']}")


if __name__ == "__main__":
    main()
