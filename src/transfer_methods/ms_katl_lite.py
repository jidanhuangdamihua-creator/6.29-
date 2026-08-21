"""
MS-KATL-Lite: Multi-Source KNN-prior Attentive Transfer Learning (Lite).

Extended-track method (v2.2 revision). Not part of FORMAL_METHODS.

与 MSML-TL 的区别：
1. 表示空间融合——K 个 source CNN 全程冻结为特征专家，不做参数平均后丢弃；
2. KNN 先验注意力——每个样本学习一组融合权重 alpha，KNN 距离作为 softmax
   先验 bias 注入（gamma 控制强度）；
3. 标量残差门控——逐样本决定接受多少迁移，可退化为 target-only；
4. 重要性加权逐源对抗对齐——L_adv = sum_i sg(alpha_bar_i) * L_domain_i，
   GRL 仅取反（-I），lambda 调度只在总损失处乘一次。

冻结实现决策（v2.2，针对超过 LSTM 的冷启动修正）：
- Target CNN E_T 复用 MSML 融合初始化，但 Stage B **解冻 conv**，与 No-TL 一样可训练；
  source CNN 仍全部冻结，其 target/source 特征可预计算；
- 预测为显式残差：y = Head_target(z_T) + g * Head_transfer(z_F)，g→0 时退化为
  融合初始化后的 target CNN 线性头（对齐 No-TL/MSML 底座）；
- 门控 bias 初始化为 -2（g≈0.12），训练从倾向 target-only 出发；
- 主实验默认 λ=0（对抗作为可选消融，不进入 5 样本主路径）；
- 每个 source 专家使用自己 Stage A 拟合的 MinMaxScaler 变换 target 输入；
- 无早停，跑满 epochs（与 MSML 微调一致）。
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None

from src.models.cnn_model import build_base_cnn
from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.source_selection.source_selector import SourceSelector, TargetK3SelectionContext
from src.utils.runtime_control import keras_verbose
from src.evaluation.metrics import smape
from src.utils.finite_diagnostics import (
    NonFiniteArrayError,
    summarize_model_weights,
    validate_finite_array,
)
from src.utils.source_fillna import fill_source_numeric_na
from src.transfer_methods.source_failure_tolerance import (
    AllSourcesFailedError,
    SOURCE_LEVEL_EXCEPTIONS,
    make_failed_source,
    normalize_successful_source_weights,
    enforce_formal_source_success,
    runtime_selection_meta,
    should_skip_source_exception,
    source_failure_meta,
)
from src.transfer_methods.msml_tl import (
    train_source_cnn_for_msml,
    get_transferable_layer_names,
    fuse_source_models_layerwise,
    load_fused_params_into_target_model,
)
from src.protocols.runner_adapter import source_key_mask
from src.protocols.raw_preprocessing import TargetK3RawSourceContext
from src.protocols.provenance import bind_actual_cnn_source_frame

from src.transfer_methods.revin import RevIN

if TYPE_CHECKING:
    from src.protocols.transformation_reuse import TargetTransformationReuseContext


LOGGER_NAME = "experiment"

METHOD_NAME = "MS-KATL-Lite"

# v2.2 冻结超参（Lite 化；为 1–5 个 target 窗口再缩一档）
D_ADAPTER = 8
D_K = 8
DISC_HIDDEN = 8
DEFAULT_GAMMA = 0.5
DEFAULT_TAU_K = 1.0
FEATURE_DROPOUT = 0.1
GATE_BIAS_INIT = -2.0  # sigmoid(-2) ≈ 0.12，倾向 target-only
DEFAULT_USE_ADVERSARIAL = False
VALID_GATE_MODES = ("learnable", "zero", "one")
# Stage-B training recipe (v2 screening arm). Defaults reproduce the frozen
# v1 behaviour exactly: pure MSE, constant LR, no weight averaging.
VALID_LOSS_NAMES = ("mse", "huber")
VALID_LR_SCHEDULES = ("constant", "cosine")
DEFAULT_LOSS_NAME = "mse"
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_USE_EMA = False
DEFAULT_EMA_MOMENTUM = 0.99
DEFAULT_USE_REVIN = False
CONV_LAYER_NAMES = ("conv1", "conv2", "conv3")
DECOMPOSITION_TOLERANCE = 1e-5
WEIGHT_DRIFT_EPS = 1e-12
TIE_ABS_EPS = 1e-12


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level="INFO", log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    return logger


# -------------------------------------------------------------------
# 1. KNN prior
# -------------------------------------------------------------------

def compute_knn_prior(distances: Sequence[float], tau_k: float = DEFAULT_TAU_K) -> np.ndarray:
    """将 KNN 欧氏距离转化为 softmax 先验分布 pi_i = softmax(-d_i / tau_k)。"""
    d = np.asarray(list(distances), dtype=np.float64)
    if d.size == 0:
        raise ValueError("compute_knn_prior requires at least one distance (K=0 must fail closed)")
    if not np.all(np.isfinite(d)):
        raise ValueError(f"compute_knn_prior received non-finite distances: {d!r}")
    if float(tau_k) <= 0:
        raise ValueError(f"tau_k must be positive, got {tau_k!r}")
    logits = -d / float(tau_k)
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def lambda_schedule(progress: float) -> float:
    """DANN 对抗系数调度：lambda(p) = 2/(1+exp(-10p)) - 1，p in [0,1]。"""
    p = min(max(float(progress), 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# -------------------------------------------------------------------
# 2. Layers
# -------------------------------------------------------------------

@tf.custom_gradient
def gradient_reversal(x):
    """GRL：前向恒等，反向梯度取反（不含 lambda，v2 修正）。"""
    def grad(dy):
        return -dy
    return tf.identity(x), grad


class GradientReversalLayer(tf.keras.layers.Layer):
    """Gradient Reversal Layer (Ganin & Lempitsky 2015)。"""

    def call(self, x):
        return gradient_reversal(x)


class SourceAdapter(tf.keras.layers.Layer):
    """每个 source 专属的单层投影 adapter（Lite: 256->8）。"""

    def __init__(self, adapter_dim: int = D_ADAPTER, **kwargs):
        super().__init__(**kwargs)
        self.projection = tf.keras.layers.Dense(adapter_dim, activation="relu")

    def call(self, x):
        return self.projection(x)


class KNNPriorCrossAttention(tf.keras.layers.Layer):
    """带 KNN 先验 bias 的 single-head cross-attention（Lite: d_k=8）。"""

    def __init__(self, d_k: int = D_K, gamma: float = DEFAULT_GAMMA,
                 epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.d_k = int(d_k)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.W_q = tf.keras.layers.Dense(self.d_k, use_bias=False)
        self.W_k = tf.keras.layers.Dense(self.d_k, use_bias=False)
        self.W_v = tf.keras.layers.Dense(self.d_k, use_bias=False)

    def call(self, z_T, source_tokens, knn_prior, source_mask=None):
        """
        Args:
            z_T: (batch, d_model)
            source_tokens: (batch, K, d_adapter)
            knn_prior: (batch, K)
            source_mask: (batch, K) bool，True=有效
        Returns:
            z_F: (batch, d_k), alpha: (batch, K)
        """
        q = tf.expand_dims(self.W_q(z_T), axis=1)          # (batch, 1, d_k)
        k = self.W_k(source_tokens)                        # (batch, K, d_k)
        v = self.W_v(source_tokens)                        # (batch, K, d_k)

        scores = tf.matmul(q, k, transpose_b=True)         # (batch, 1, K)
        scores = scores / tf.sqrt(tf.cast(self.d_k, tf.float32))
        scores = tf.squeeze(scores, axis=1)                # (batch, K)

        prior_bias = self.gamma * tf.math.log(
            tf.cast(knn_prior, tf.float32) + self.epsilon
        )
        scores = scores + prior_bias

        if source_mask is not None:
            scores = tf.where(
                tf.cast(source_mask, tf.bool),
                scores,
                tf.fill(tf.shape(scores), tf.constant(-1e9, tf.float32)),
            )

        alpha = tf.nn.softmax(scores, axis=-1)             # (batch, K)
        z_F = tf.reduce_sum(tf.expand_dims(alpha, -1) * v, axis=1)  # (batch, d_k)
        return z_F, alpha


class ResidualGate(tf.keras.layers.Layer):
    """负迁移防护标量门控。g→0 时预测退化为 Head_target(z_T)。"""

    def __init__(self, bias_init: float = GATE_BIAS_INIT, **kwargs):
        super().__init__(**kwargs)
        self.gate_proj = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(float(bias_init)),
        )

    def call(self, z_T, z_F):
        concat = tf.concat([z_T, z_F], axis=-1)
        return self.gate_proj(concat)  # (batch, 1)


class DomainDiscriminator(tf.keras.layers.Layer):
    """每个 source 专属的域判别器（Lite: d_adapter→8→1）。"""

    def __init__(self, hidden_dim: int = DISC_HIDDEN, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        return self.dense2(self.dense1(x))


def validate_gate_mode(gate_mode: str) -> str:
    mode = str(gate_mode).strip().lower()
    if mode not in VALID_GATE_MODES:
        raise ValueError(
            f"Unknown gate_mode={gate_mode!r}. Expected one of: {VALID_GATE_MODES}"
        )
    return mode


class MSKATLLite(tf.keras.Model):
    """MS-KATL-Lite 头部模型（输入为预计算的冻结 CNN 特征）。"""

    def __init__(self, d_model: int = 256, d_adapter: int = D_ADAPTER,
                 d_k: int = D_K, num_sources: int = 3,
                 gamma: float = DEFAULT_GAMMA, dropout: float = FEATURE_DROPOUT,
                 transfer_branch_enabled: bool = True,
                 gate_mode: str = "learnable",
                 **kwargs):
        super().__init__(**kwargs)
        if int(num_sources) < 1:
            raise ValueError(
                f"MS-KATL-Lite requires 1 <= K, got K={num_sources} (K=0 must fail closed)"
            )
        if int(num_sources) > 3:
            raise ValueError(f"MS-KATL-Lite requires K <= 3, got K={num_sources}")
        self.num_sources = int(num_sources)
        self.d_model = int(d_model)
        self.transfer_branch_enabled = bool(transfer_branch_enabled)
        self.gate_mode = validate_gate_mode(gate_mode)
        if not self.transfer_branch_enabled and self.gate_mode != "zero":
            raise ValueError(
                "transfer_branch_enabled=False requires gate_mode='zero' (fail closed)"
            )
        if self.gate_mode == "zero" and self.transfer_branch_enabled:
            raise ValueError(
                "gate_mode='zero' requires transfer_branch_enabled=False (fail closed)"
            )
        self.feature_dropout = tf.keras.layers.Dropout(float(dropout))
        self.adapters = [
            SourceAdapter(adapter_dim=d_adapter, name=f"adapter_{i}")
            for i in range(self.num_sources)
        ]
        self.cross_attention = KNNPriorCrossAttention(d_k=d_k, gamma=gamma)
        self.residual_gate = ResidualGate()
        self.head_target = tf.keras.layers.Dense(1, name="head_target")
        self.head_transfer = tf.keras.layers.Dense(1, name="head_transfer")
        self.grl = GradientReversalLayer()
        self.discriminators = [
            DomainDiscriminator(name=f"discriminator_{i}")
            for i in range(self.num_sources)
        ]

    def forward_decomposed(
        self,
        z_T,
        source_features,
        knn_prior,
        source_mask=None,
        training=False,
    ) -> Dict[str, tf.Tensor]:
        """Return residual prediction parts without changing default call outputs."""
        if len(source_features) != self.num_sources:
            raise ValueError(
                f"expected {self.num_sources} source feature tensors, got {len(source_features)}"
            )
        z_T = self.feature_dropout(z_T, training=training)
        target_prediction = self.head_target(z_T)
        batch = tf.shape(z_T)[0]
        if not self.transfer_branch_enabled:
            zeros_alpha = tf.zeros((batch, self.num_sources), dtype=z_T.dtype)
            zeros_gate = tf.zeros((batch, 1), dtype=z_T.dtype)
            zeros_transfer = tf.zeros((batch, 1), dtype=z_T.dtype)
            return {
                "y_pred": target_prediction,
                "alpha": zeros_alpha,
                "gate": zeros_gate,
                "target_prediction": target_prediction,
                "transfer_raw": zeros_transfer,
                "transfer_gated": zeros_transfer,
                "z_F": tf.zeros((batch, int(self.cross_attention.d_k)), dtype=z_T.dtype),
            }
        adapted = [self.adapters[i](feat) for i, feat in enumerate(source_features)]
        source_tokens = tf.stack(adapted, axis=1)
        z_F, alpha = self.cross_attention(z_T, source_tokens, knn_prior, source_mask)
        transfer_raw = self.head_transfer(z_F)
        if self.gate_mode == "one":
            gate = tf.ones((batch, 1), dtype=z_T.dtype)
        elif self.gate_mode == "learnable":
            gate = self.residual_gate(z_T, z_F)
        else:
            raise ValueError(f"unreachable gate_mode={self.gate_mode!r} with transfer enabled")
        transfer_gated = gate * transfer_raw
        return {
            "y_pred": target_prediction + transfer_gated,
            "alpha": alpha,
            "gate": gate,
            "target_prediction": target_prediction,
            "transfer_raw": transfer_raw,
            "transfer_gated": transfer_gated,
            "z_F": z_F,
        }

    def call(self, z_T, source_features, knn_prior, source_mask=None, training=False):
        """
        推理/预测前向传播（不含域对抗；对抗仅出现在训练循环）。

        Args:
            z_T: (batch, d_model) 预计算 target 特征
            source_features: list[K] of (batch, d_model) 预计算 source 专家特征
            knn_prior: (batch, K)
        Returns:
            y_pred: (batch, 1), alpha: (batch, K), g: (batch, 1)
        """
        decomposed = self.forward_decomposed(
            z_T, source_features, knn_prior, source_mask=source_mask, training=training
        )
        return decomposed["y_pred"], decomposed["alpha"], decomposed["gate"]


# -------------------------------------------------------------------
# 3. Frozen feature extraction helpers
# -------------------------------------------------------------------

def freeze_source_cnn(source_model):
    """冻结整个 source CNN（Stage A 完成后调用）。"""
    for layer in source_model.layers:
        layer.trainable = False
    return source_model


def build_flatten_feature_extractor(model, name: str = "feature_extractor"):
    """截取 CNN flatten 层输出作为特征提取器。"""
    return tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("flatten").output,
        name=name,
    )


def _strip_attrs_copy(df: pd.DataFrame) -> pd.DataFrame:
    """浅拷贝并清空 attrs，避免自定义 scaler 变换污染 reuse 缓存身份。"""
    out = df.copy()
    out.attrs = {}
    return out


def _transform_with_scaler(
    df: pd.DataFrame,
    scaler,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """用外部（source-i）scaler 变换 target 帧的特征列。"""
    out = _strip_attrs_copy(df)
    cols = [str(c) for c in feature_columns]
    out[cols] = scaler.transform(out[cols].astype(np.float64))
    return out


def _source_scaler_and_train_tensor(
    source_sequence_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int,
    window_size: int,
    transformation_reuse_context: "TargetTransformationReuseContext | None" = None,
) -> Tuple[object, np.ndarray, List[str]]:
    """
    重建 source-i 的 Stage A MinMaxScaler 与训练窗口张量。

    与 train_source_cnn_for_msml 内部完全相同的 split/normalize/windowing 调用，
    传入同一 reuse 上下文时命中缓存，保证 scaler 与 Stage A 训练时一致。
    """
    working = fill_source_numeric_na(source_sequence_df, feature_columns=feature_cols)
    working = working.copy()
    working.attrs["split_role"] = "source"
    working.attrs["split_mode"] = "ratio"
    working.attrs["split_config"] = {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }
    src_train, src_val, src_test = temporal_split_by_ratio_or_dates(working)
    src_train, src_val, src_test, source_scaler, source_feature_columns = normalize_features(
        src_train, src_val, src_test, feature_columns=feature_cols,
        reuse_context=transformation_reuse_context,
    )
    X_source, y_source = build_tabular_sequence(
        src_train, horizon=horizon, window_size=window_size, feature_columns=feature_cols,
        reuse_context=transformation_reuse_context,
    )
    if len(y_source) == 0:
        raise ValueError("source split produced zero training windows for MS-KATL")
    return source_scaler, to_cnn_tensor(X_source), list(source_feature_columns)


# -------------------------------------------------------------------
# 4. Stage B custom training loop (on precomputed frozen features)
# -------------------------------------------------------------------

def train_ms_katl_head(
    model: MSKATLLite,
    x_train: np.ndarray,
    f_t_train: List[np.ndarray],
    y_train: np.ndarray,
    f_s_pools: List[np.ndarray],
    knn_prior: np.ndarray,
    target_extractor=None,
    x_val: Optional[np.ndarray] = None,
    f_t_val: Optional[List[np.ndarray]] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    seed: int = 42,
    use_adversarial: bool = DEFAULT_USE_ADVERSARIAL,
    trainable_variables: Optional[Sequence[tf.Variable]] = None,
    collect_gate_grads: bool = False,
    loss_name: str = DEFAULT_LOSS_NAME,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    lr_schedule: str = DEFAULT_LR_SCHEDULE,
    use_ema: bool = DEFAULT_USE_EMA,
    ema_momentum: float = DEFAULT_EMA_MOMENTUM,
    revin: Optional[RevIN] = None,
    revin_include_mean: bool = True,
) -> Dict[str, object]:
    """
    MS-KATL Stage B 训练循环：
    - source 专家特征预计算；target 表示可由冻结特征或可训练 extractor 得到；
    - 主实验默认 use_adversarial=False（λ=0），对抗仅作消融；
    - 无早停，跑满 epochs；
    - 可选训练配方（v2 screening 臂）：Huber 损失 / CosineDecay 学习率 /
      EMA 权重平均。默认值复现 v1 行为（MSE + 常数 LR + 无 EMA）。
    - 可选 RevIN（v3 screening 臂）：对输入窗口做实例归一化，预测后再反归一化
      回到 MinMax 空间。默认关闭，V1/V2 行为不变。
    - revin_include_mean 默认 True，保持 V3 location+scale 反变换；False 为
      residual-aware 的 scale-only 反变换（V3.2）。默认值不得改变 V3 数值路径。
    """
    logger = _get_logger()
    resolved_loss = str(loss_name).strip().lower()
    if resolved_loss not in VALID_LOSS_NAMES:
        raise ValueError(f"Unknown loss_name={loss_name!r}. Expected one of: {VALID_LOSS_NAMES}")
    resolved_schedule = str(lr_schedule).strip().lower()
    if resolved_schedule not in VALID_LR_SCHEDULES:
        raise ValueError(
            f"Unknown lr_schedule={lr_schedule!r}. Expected one of: {VALID_LR_SCHEDULES}"
        )
    if float(huber_delta) <= 0.0:
        raise ValueError(f"huber_delta must be positive, got {huber_delta!r}")
    if not (0.0 < float(ema_momentum) < 1.0):
        raise ValueError(f"ema_momentum must be in (0, 1), got {ema_momentum!r}")
    if revin is not None and not isinstance(revin, RevIN):
        raise TypeError(f"revin must be a RevIN layer or None, got {type(revin)!r}")
    include_mean = bool(revin_include_mean)
    k = model.num_sources
    n = int(x_train.shape[0])
    if n == 0:
        raise ValueError("MS-KATL received zero target training samples")
    rng = np.random.default_rng(int(seed))
    tf.random.set_seed(int(seed))

    steps_per_epoch = max(1, math.ceil(n / int(batch_size)))
    total_steps = max(1, int(epochs) * steps_per_epoch)

    if resolved_schedule == "cosine":
        lr_arg = tf.keras.optimizers.schedules.CosineDecay(
            float(learning_rate), decay_steps=int(total_steps)
        )
    else:
        lr_arg = float(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_arg)
    bce = tf.keras.losses.BinaryCrossentropy()
    huber_loss = (
        tf.keras.losses.Huber(delta=float(huber_delta)) if resolved_loss == "huber" else None
    )

    x_train_tf = tf.constant(x_train, dtype=tf.float32)
    f_t_train_tf = [tf.constant(f, dtype=tf.float32) for f in f_t_train]
    y_train_tf = tf.constant(np.asarray(y_train, dtype=np.float32).reshape(-1, 1))
    prior_row = np.asarray(knn_prior, dtype=np.float32).reshape(1, k)

    probe_x = x_train_tf[:1]
    if revin is not None:
        probe_x, _, _ = revin.normalize(probe_x)
    probe_z = target_extractor(probe_x, training=False) if target_extractor is not None else probe_x
    probe_out = model(
        probe_z,
        [f[:1] for f in f_t_train_tf],
        tf.constant(np.repeat(prior_row, 1, axis=0)),
        training=False,
    )
    if trainable_variables is not None:
        trainable_vars = list(trainable_variables)
    else:
        trainable_vars = list(model.trainable_variables)
        if target_extractor is not None:
            extractor_ids = {id(v) for v in model.trainable_variables}
            extractor_vars = [
                v for v in target_extractor.trainable_variables if id(v) not in extractor_ids
            ]
            trainable_vars = extractor_vars + trainable_vars
        if revin is not None:
            revin_ids = {id(v) for v in trainable_vars}
            trainable_vars = trainable_vars + [
                v for v in revin.trainable_variables if id(v) not in revin_ids
            ]
    if not trainable_vars:
        raise ValueError("MS-KATL Stage B has no trainable variables")
    ema_values: List[tf.Tensor] = (
        [tf.identity(tf.convert_to_tensor(v)) for v in trainable_vars] if use_ema else []
    )
    gate_var_ids = {
        id(v) for v in model.residual_gate.trainable_variables
    } if model.gate_mode == "learnable" and model.residual_gate.built else set()
    initial_gate_mean = float(tf.reduce_mean(probe_out[2]).numpy())
    initial_gate_std = float(tf.math.reduce_std(probe_out[2]).numpy())
    initial_gate_bias = None
    if model.gate_mode == "learnable" and model.residual_gate.built:
        initial_gate_bias = float(model.residual_gate.gate_proj.bias.numpy().reshape(-1)[0])
    gate_grad_norms: List[float] = []

    history: Dict[str, List[float]] = {
        "forecast_loss": [], "adv_loss": [], "total_loss": [],
        "lambda": [], "gate_mean": [], "gate_std": [],
        "val_rmse": [],
    }
    history.update({f"alpha_bar_{i}": [] for i in range(k)})
    history.update({f"disc_{i}_acc": [] for i in range(k)})

    global_step = 0
    for epoch in range(int(epochs)):
        order = rng.permutation(n)
        epoch_stats: Dict[str, List[float]] = {key: [] for key in history if key != "val_rmse"}
        for step in range(steps_per_epoch):
            batch_idx = order[step * batch_size:(step + 1) * batch_size]
            if batch_idx.size == 0:
                batch_idx = order
            b = int(batch_idx.size)
            lambda_val = lambda_schedule(global_step / total_steps) if use_adversarial else 0.0

            xb = tf.gather(x_train_tf, batch_idx)
            fb = [tf.gather(f, batch_idx) for f in f_t_train_tf]
            yb = tf.gather(y_train_tf, batch_idx)
            prior_b = tf.constant(np.repeat(prior_row, b, axis=0))

            with tf.GradientTape() as tape:
                xb_in = xb
                revin_mean = None
                revin_stdev = None
                if revin is not None:
                    xb_in, revin_mean, revin_stdev = revin.normalize(xb)
                z_b = target_extractor(xb_in, training=True) if target_extractor is not None else xb_in
                y_pred, alpha, g = model(z_b, fb, prior_b, training=True)
                if revin is not None:
                    y_pred = revin.denormalize(
                        y_pred, revin_mean, revin_stdev, include_mean=include_mean
                    )
                if huber_loss is not None:
                    loss_forecast = huber_loss(yb, y_pred)
                else:
                    loss_forecast = tf.reduce_mean(tf.square(yb - y_pred))

                alpha_bar = tf.reduce_mean(alpha, axis=0)
                loss_adv = tf.constant(0.0, tf.float32)
                disc_accs: List[tf.Tensor] = [
                    tf.constant(float("nan"), tf.float32) for _ in range(k)
                ]
                if use_adversarial:
                    sb = [
                        tf.constant(
                            pool[rng.integers(0, pool.shape[0], size=b)], dtype=tf.float32
                        )
                        for pool in f_s_pools
                    ]
                    disc_accs = []
                    for i in range(k):
                        u_i_t = model.adapters[i](fb[i])
                        u_i_s = model.adapters[i](sb[i])
                        pred_t = model.discriminators[i](model.grl(u_i_t))
                        pred_s = model.discriminators[i](model.grl(u_i_s))
                        loss_domain_i = (
                            bce(tf.zeros_like(pred_s), pred_s)
                            + bce(tf.ones_like(pred_t), pred_t)
                        )
                        loss_adv += tf.stop_gradient(alpha_bar[i]) * loss_domain_i
                        acc = (
                            tf.reduce_mean(tf.cast(pred_s < 0.5, tf.float32))
                            + tf.reduce_mean(tf.cast(pred_t >= 0.5, tf.float32))
                        ) / 2.0
                        disc_accs.append(acc)

                total_loss = loss_forecast + lambda_val * loss_adv

            grads = tape.gradient(total_loss, trainable_vars)
            finite_grads = []
            step_gate_sq = 0.0
            saw_gate_grad = False
            for grad, var in zip(grads, trainable_vars):
                if grad is None:
                    continue
                if not bool(tf.reduce_all(tf.math.is_finite(grad)).numpy()):
                    raise NonFiniteArrayError(
                        f"MS-KATL gradient for {var.name} contains non-finite values "
                        f"at step={global_step}",
                        diagnostics={"variable": var.name, "step": int(global_step)},
                    )
                finite_grads.append((grad, var))
                if collect_gate_grads and id(var) in gate_var_ids:
                    saw_gate_grad = True
                    step_gate_sq += float(tf.reduce_sum(tf.square(grad)).numpy())
            if collect_gate_grads and gate_var_ids:
                gate_grad_norms.append(float(math.sqrt(step_gate_sq)) if saw_gate_grad else 0.0)
            optimizer.apply_gradients(finite_grads)
            if use_ema:
                momentum = float(ema_momentum)
                ema_values = [
                    momentum * ema + (1.0 - momentum) * tf.convert_to_tensor(var)
                    for ema, var in zip(ema_values, trainable_vars)
                ]
            global_step += 1

            epoch_stats["forecast_loss"].append(float(loss_forecast.numpy()))
            epoch_stats["adv_loss"].append(float(loss_adv.numpy()))
            epoch_stats["total_loss"].append(float(total_loss.numpy()))
            epoch_stats["lambda"].append(float(lambda_val))
            epoch_stats["gate_mean"].append(float(tf.reduce_mean(g).numpy()))
            epoch_stats["gate_std"].append(float(tf.math.reduce_std(g).numpy()))
            for i in range(k):
                epoch_stats[f"alpha_bar_{i}"].append(float(alpha_bar[i].numpy()))
                epoch_stats[f"disc_{i}_acc"].append(float(disc_accs[i].numpy()))

        for key, values in epoch_stats.items():
            history[key].append(float(np.mean(values)) if values else float("nan"))

        if x_val is not None and y_val is not None and len(y_val) > 0 and f_t_val is not None:
            x_val_tf = tf.constant(x_val, dtype=tf.float32)
            if revin is not None:
                x_val_tf, val_mean, val_stdev = revin.normalize(x_val_tf)
            else:
                val_mean = val_stdev = None
            if target_extractor is not None:
                z_val = target_extractor(x_val_tf, training=False).numpy()
            else:
                z_val = x_val_tf.numpy()
            val_pred = predict_ms_katl(model, z_val, f_t_val, knn_prior)
            if revin is not None:
                val_pred = revin.denormalize_numpy(
                    val_pred,
                    val_mean.numpy(),
                    val_stdev.numpy(),
                    include_mean=include_mean,
                )
            val_rmse = float(np.sqrt(np.mean((val_pred.flatten() - np.asarray(y_val).flatten()) ** 2)))
            history["val_rmse"].append(val_rmse)
        else:
            history["val_rmse"].append(float("nan"))

    ema_applied = False
    if use_ema:
        for var, ema in zip(trainable_vars, ema_values):
            var.assign(ema)
        ema_applied = True

    if resolved_schedule == "cosine":
        final_learning_rate = float(lr_arg(int(total_steps)).numpy())
    else:
        final_learning_rate = float(learning_rate)

    logger.info(
        "[train_ms_katl_head] Finished. epochs=%d steps=%d adversarial=%s final_forecast_loss=%.6f",
        int(epochs), global_step, bool(use_adversarial),
        history["forecast_loss"][-1] if history["forecast_loss"] else float("nan"),
    )
    final_gate_bias = None
    if model.gate_mode == "learnable" and model.residual_gate.built:
        final_gate_bias = float(model.residual_gate.gate_proj.bias.numpy().reshape(-1)[0])
    return {
        "history": history,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "training_recipe": {
            "loss_name": resolved_loss,
            "huber_delta": float(huber_delta) if resolved_loss == "huber" else None,
            "lr_schedule": resolved_schedule,
            "initial_learning_rate": float(learning_rate),
            "final_learning_rate": final_learning_rate,
            "use_ema": bool(use_ema),
            "ema_momentum": float(ema_momentum) if use_ema else None,
            "ema_applied": ema_applied,
            "revin_enabled": revin is not None,
            "revin_affine": bool(revin.affine) if revin is not None else None,
            "revin_eps": float(revin.eps) if revin is not None else None,
            "revin_target_channel": int(revin.target_channel) if revin is not None else None,
            "revin_include_mean": bool(include_mean) if revin is not None else None,
        },
        "gate_diagnostics": {
            "gate_mode": str(model.gate_mode),
            "gate_bias_initial": initial_gate_bias,
            "gate_bias_final": final_gate_bias,
            "gate_mean_initial": initial_gate_mean,
            "gate_mean_final": float(history["gate_mean"][-1]) if history["gate_mean"] else float("nan"),
            "gate_std_final": float(history["gate_std"][-1]) if history["gate_std"] else float("nan"),
            "gate_grad_norm_mean": (
                float(np.mean(gate_grad_norms)) if gate_grad_norms else float("nan")
            ),
            "gate_grad_norm_max": (
                float(np.max(gate_grad_norms)) if gate_grad_norms else float("nan")
            ),
            "gate_grad_steps": int(len(gate_grad_norms)),
        },
    }


def predict_ms_katl(
    model: MSKATLLite,
    z_t: np.ndarray,
    f_t: List[np.ndarray],
    knn_prior: np.ndarray,
) -> np.ndarray:
    """推理路径：无 GRL、无判别器、无 source batch。"""
    n = int(z_t.shape[0])
    prior = tf.constant(
        np.repeat(np.asarray(knn_prior, dtype=np.float32).reshape(1, -1), n, axis=0)
    )
    y_pred, _, _ = model(
        tf.constant(z_t, dtype=tf.float32),
        [tf.constant(f, dtype=tf.float32) for f in f_t],
        prior,
        training=False,
    )
    return y_pred.numpy()


def predict_ms_katl_decomposed(
    model: MSKATLLite,
    z_t: np.ndarray,
    f_t: List[np.ndarray],
    knn_prior: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Diagnostic forward that returns residual prediction parts."""
    n = int(z_t.shape[0])
    prior = tf.constant(
        np.repeat(np.asarray(knn_prior, dtype=np.float32).reshape(1, -1), n, axis=0)
    )
    decomposed = model.forward_decomposed(
        tf.constant(z_t, dtype=tf.float32),
        [tf.constant(f, dtype=tf.float32) for f in f_t],
        prior,
        training=False,
    )
    return {key: value.numpy() for key, value in decomposed.items()}


def extract_features_maybe_revin(
    target_extractor,
    x: np.ndarray,
    revin: Optional[RevIN] = None,
    *,
    training: bool = False,
):
    """Extract CNN features, optionally wrapping the window tensor with RevIN.

    Returns (z, mean, stdev). mean/stdev are None when RevIN is disabled.
    """
    x_tf = tf.constant(x, dtype=tf.float32)
    mean = stdev = None
    if revin is not None:
        x_tf, mean, stdev = revin.normalize(x_tf)
    if target_extractor is not None:
        z = target_extractor(x_tf, training=training)
        z_np = z.numpy() if hasattr(z, "numpy") else z
    else:
        z_np = x_tf.numpy()
    return z_np, mean, stdev


def digest_weight_arrays(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def model_weight_digest(model) -> str:
    return digest_weight_arrays(model.get_weights())


def snapshot_model_weights(model) -> List[np.ndarray]:
    return [np.array(weight, copy=True) for weight in model.get_weights()]


def snapshot_named_layer_weights(model, layer_names: Sequence[str]) -> Dict[str, List[np.ndarray]]:
    layer_map = {layer.name: layer for layer in model.layers}
    snapshots: Dict[str, List[np.ndarray]] = {}
    for name in layer_names:
        if name not in layer_map:
            raise ValueError(f"Layer {name!r} not found for weight snapshot")
        snapshots[str(name)] = [np.array(weight, copy=True) for weight in layer_map[name].get_weights()]
    return snapshots


def relative_weight_drift(
    initial_weights: Sequence[np.ndarray],
    final_weights: Sequence[np.ndarray],
    eps: float = WEIGHT_DRIFT_EPS,
) -> float:
    if len(initial_weights) != len(final_weights):
        raise ValueError(
            "relative_weight_drift requires matching snapshot lengths: "
            f"{len(initial_weights)} vs {len(final_weights)}"
        )
    delta_sq = 0.0
    init_sq = 0.0
    for initial, final in zip(initial_weights, final_weights):
        initial_arr = np.asarray(initial, dtype=np.float64)
        final_arr = np.asarray(final, dtype=np.float64)
        delta = final_arr - initial_arr
        delta_sq += float(np.sum(delta * delta))
        init_sq += float(np.sum(initial_arr * initial_arr))
    return float(math.sqrt(delta_sq) / (math.sqrt(init_sq) + float(eps)))


def scaler_digest(scaler) -> str:
    parts: List[np.ndarray] = []
    for attr in ("data_min_", "data_max_", "scale_", "min_"):
        if hasattr(scaler, attr):
            parts.append(np.asarray(getattr(scaler, attr)))
    if not parts:
        raise ValueError("scaler_digest requires a fitted scaler exposing min/max/scale")
    return digest_weight_arrays(parts)


def fused_params_digest(fused_params: Dict[str, Sequence[np.ndarray]]) -> str:
    parts: List[np.ndarray] = []
    for name in sorted(fused_params):
        for array in fused_params[name]:
            parts.append(np.asarray(array))
    return digest_weight_arrays(parts)


@dataclass
class MSKATLStageAArtifacts:
    """Frozen Stage-A artifacts shared by every mechanism-ablation arm."""

    selection_result: Dict[str, object]
    selected_sources: List[Dict[str, object]]
    source_models: List[object]
    source_scalers: List[object]
    source_train_tensors: List[np.ndarray]
    source_weights: List[float]
    source_distances: List[float]
    source_models_info: List[Dict[str, object]]
    source_identities: List[Tuple[object, ...]]
    knn_prior: np.ndarray
    input_shape: Tuple[int, ...]
    fused_params: Dict[str, List[np.ndarray]]
    layer_names: List[str]
    source_weight_snapshots: List[List[np.ndarray]]
    source_weight_digests: List[str]
    source_scaler_digests: List[str]
    fused_params_digest: str
    failed_sources: List[Dict[str, object]]
    effective_k: int
    tau_k: float
    source_extractors: List[object] = field(default_factory=list)


def prepare_ms_katl_stage_a(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    batch_size: int = 16,
    tau_k: float = DEFAULT_TAU_K,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
    k3_selection_context: TargetK3SelectionContext | None = None,
    k3_raw_source_context: TargetK3RawSourceContext | None = None,
    transformation_reuse_context: "TargetTransformationReuseContext | None" = None,
) -> MSKATLStageAArtifacts:
    """Train/load Stage A once: KNN sources, frozen source CNNs, scalers, fused conv params."""
    logger = _get_logger()
    resolved_group_cols = tuple(group_cols)

    if k3_selection_context is None:
        selector = SourceSelector()
        selection_result = selector.select_top_k_sources(
            target_df=target_df,
            source_df=source_df,
            feature_cols=feature_cols,
            k=k,
            group_cols=resolved_group_cols,
            weight_mode=weight_mode,
        )
    else:
        evidence = k3_selection_context.selection_for_method(
            target_df=target_df,
            source_df=source_df,
            feature_cols=feature_cols,
            group_cols=resolved_group_cols,
            k=k,
            weight_mode=weight_mode,
        )
        selection_result = evidence.method_wrapper(
            lifecycle_identity=k3_selection_context.lifecycle_identity,
            target_df=target_df,
            source_df=source_df,
            feature_cols=feature_cols,
            group_cols=resolved_group_cols,
            k=k,
            weight_mode=weight_mode,
        )
    selected_sources = (
        selection_result.get("sources", [])
        if isinstance(selection_result, dict)
        else selection_result
    )
    if not selected_sources:
        raise ValueError("No source selected from source pool (K=0 must fail closed).")

    source_models: List = []
    source_scalers: List[object] = []
    source_train_tensors: List[np.ndarray] = []
    source_weights: List[float] = []
    source_distances: List[float] = []
    source_models_info: List[Dict[str, object]] = []
    source_identities: List[Tuple[object, ...]] = []
    failed_sources: List[Dict[str, object]] = []
    input_shape_ref: Optional[Tuple[int, ...]] = None

    for selected in selected_sources:
        source_key = (
            tuple(selected["source_key"])
            if isinstance(selected["source_key"], (list, tuple))
            else (selected["source_key"],)
        )
        if len(source_key) != len(resolved_group_cols):
            raise ValueError(f"Invalid source_key format: {source_key}")

        if k3_raw_source_context is None:
            mask = source_key_mask(source_df, resolved_group_cols, source_key)
            source_sequence_df = source_df[mask].copy()
        else:
            if k3_selection_context is None:
                raise ValueError("K3 raw source reuse requires the shared K3 selection context")
            source_sequence_df = k3_raw_source_context.working_source(
                selection_result=selection_result,
                source_df=source_df,
                source_key=source_key,
                group_cols=resolved_group_cols,
                model_feature_cols=feature_cols,
            )
        if source_sequence_df.empty:
            raise ValueError(f"Selected source_key not found in source_df: {source_key}")
        bind_actual_cnn_source_frame(
            source_sequence_df,
            source_key=source_key,
            group_cols=resolved_group_cols,
            feature_cols=feature_cols,
        )

        try:
            train_result = train_source_cnn_for_msml(
                source_sequence_df=source_sequence_df,
                feature_cols=feature_cols,
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                source_epochs=source_epochs,
                batch_size=batch_size,
                source_key=source_key,
                transformation_reuse_context=transformation_reuse_context,
            )
            weight_diagnostics = summarize_model_weights(train_result["model"])
            if weight_diagnostics["model_weight_nan_count"] or weight_diagnostics["model_weight_inf_count"]:
                raise NonFiniteArrayError(
                    "source model weights contain non-finite values: "
                    f"nan_count={weight_diagnostics['model_weight_nan_count']} "
                    f"inf_count={weight_diagnostics['model_weight_inf_count']}",
                    diagnostics=weight_diagnostics,
                )
            source_scaler, source_train_tensor, _ = _source_scaler_and_train_tensor(
                source_sequence_df,
                feature_cols=feature_cols,
                horizon=horizon,
                window_size=window_size,
                transformation_reuse_context=transformation_reuse_context,
            )
        except SOURCE_LEVEL_EXCEPTIONS as exc:
            enforce_formal_source_success(source_df, source_key, exc)
            if not should_skip_source_exception(exc):
                raise
            failed_source = make_failed_source(source_key, exc)
            failed_sources.append(failed_source)
            logger.warning(
                "[prepare_ms_katl_stage_a] Skipping failed source_key=%s exception_type=%s message=%s",
                source_key,
                failed_source["exception_type"],
                failed_source["exception_message"],
            )
            continue

        if input_shape_ref is None:
            input_shape_ref = train_result["input_shape"]
        elif train_result["input_shape"] != input_shape_ref:
            raise ValueError(
                f"Input shape mismatch: source_key={source_key} "
                f"shape={train_result['input_shape']} expected={input_shape_ref}"
            )

        frozen = freeze_source_cnn(train_result["model"])
        source_models.append(frozen)
        source_scalers.append(source_scaler)
        source_train_tensors.append(source_train_tensor)
        source_weights.append(float(selected["weight"]))
        source_distances.append(float(selected["distance"]))
        source_identities.append(tuple(source_key))
        source_models_info.append({
            "source_key": source_key,
            "distance": float(selected["distance"]),
            "weight": float(selected["weight"]),
            "num_samples": int(train_result["num_samples"]),
        })

    if not source_models:
        raise AllSourcesFailedError(
            METHOD_NAME,
            failed_sources,
            selected_sources=selected_sources,
            selection_meta=runtime_selection_meta(selection_result),
        )

    effective_k = len(source_models)
    source_weights = normalize_successful_source_weights(source_weights)
    for info, normalized_weight in zip(source_models_info, source_weights):
        info["weight"] = float(normalized_weight)

    knn_prior = compute_knn_prior(source_distances, tau_k=tau_k)
    layer_names = get_transferable_layer_names(source_models[0])
    fused_params = fuse_source_models_layerwise(source_models, source_weights, layer_names)
    source_weight_snapshots = [snapshot_model_weights(model) for model in source_models]
    source_weight_digests = [digest_weight_arrays(snapshot) for snapshot in source_weight_snapshots]
    source_scaler_digests = [scaler_digest(scaler) for scaler in source_scalers]
    source_extractors = [
        build_flatten_feature_extractor(model, name=f"source_feature_extractor_{i}")
        for i, model in enumerate(source_models)
    ]
    logger.info(
        "[prepare_ms_katl_stage_a] Frozen K=%d identities=%s digests=%s",
        effective_k,
        source_identities,
        source_weight_digests,
    )
    return MSKATLStageAArtifacts(
        selection_result=selection_result if isinstance(selection_result, dict) else {"sources": selected_sources},
        selected_sources=list(selected_sources),
        source_models=source_models,
        source_scalers=source_scalers,
        source_train_tensors=source_train_tensors,
        source_weights=source_weights,
        source_distances=source_distances,
        source_models_info=source_models_info,
        source_identities=source_identities,
        knn_prior=np.asarray(knn_prior, dtype=np.float64),
        input_shape=tuple(input_shape_ref),
        fused_params=fused_params,
        layer_names=list(layer_names),
        source_weight_snapshots=source_weight_snapshots,
        source_weight_digests=source_weight_digests,
        source_scaler_digests=source_scaler_digests,
        fused_params_digest=fused_params_digest(fused_params),
        failed_sources=failed_sources,
        effective_k=int(effective_k),
        tau_k=float(tau_k),
        source_extractors=source_extractors,
    )


def clone_cnn_from_snapshot(
    input_shape: Tuple[int, ...],
    learning_rate: float,
    snapshot: Sequence[np.ndarray],
    *,
    target_backbone_trainable: bool,
    name: str = "target_feature_extractor",
):
    """Rebuild a target CNN from a frozen weight snapshot."""
    target_cnn = build_base_cnn(input_shape, learning_rate=learning_rate)
    target_cnn.set_weights([np.array(weight, copy=True) for weight in snapshot])
    if target_backbone_trainable:
        for layer in target_cnn.layers:
            layer.trainable = layer.name not in {"dense_out", "dropout"}
    else:
        for layer in target_cnn.layers:
            layer.trainable = False
    extractor = build_flatten_feature_extractor(target_cnn, name=name)
    return target_cnn, extractor


def raw_target_init_snapshot(
    input_shape: Tuple[int, ...],
    learning_rate: float,
    seed: int,
) -> List[np.ndarray]:
    """No-TL-style random CNN initialization under a fixed seed."""
    tf.random.set_seed(int(seed))
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(int(seed))
    model = build_base_cnn(input_shape, learning_rate=learning_rate)
    return snapshot_model_weights(model)


def fused_target_init_snapshot(
    input_shape: Tuple[int, ...],
    learning_rate: float,
    fused_params: Dict[str, List[np.ndarray]],
    seed: int,
) -> List[np.ndarray]:
    """MSML fused conv initialization plus seeded non-fused layers."""
    tf.random.set_seed(int(seed))
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(int(seed))
    model = build_base_cnn(input_shape, learning_rate=learning_rate)
    model = load_fused_params_into_target_model(model, fused_params)
    return snapshot_model_weights(model)


# -------------------------------------------------------------------
# 5. Full pipeline
# -------------------------------------------------------------------

def run_ms_katl_lite(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    gamma: float = DEFAULT_GAMMA,
    tau_k: float = DEFAULT_TAU_K,
    seed: int = 42,
    use_adversarial: bool = DEFAULT_USE_ADVERSARIAL,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
    metric_identity: Optional[Dict[str, object]] = None,
    k3_selection_context: TargetK3SelectionContext | None = None,
    k3_raw_source_context: TargetK3RawSourceContext | None = None,
    transformation_reuse_context: "TargetTransformationReuseContext | None" = None,
) -> Dict[str, object]:
    """
    运行 MS-KATL-Lite 完整流程。

    Stage A: KNN 选源 + 逐源 CNN 预训练（复用 train_source_cnn_for_msml）+ source 全冻结；
    Stage B: 融合初始化 target CNN 并解冻 conv → source 特征预计算 → 训练 E_T + 注意力残差头
             （默认 λ=0；use_adversarial=True 时才开逐源对抗）；
    评估: target test 上预测，payload 对齐现有 fused_result 合同。
    """
    logger = _get_logger()
    logger.info("[run_ms_katl_lite] Start. k=%d gamma=%.2f tau_k=%.2f seed=%d adversarial=%s",
                k, gamma, tau_k, seed, bool(use_adversarial))

    stage_a = prepare_ms_katl_stage_a(
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        k=k,
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        batch_size=batch_size,
        tau_k=tau_k,
        group_cols=group_cols,
        k3_selection_context=k3_selection_context,
        k3_raw_source_context=k3_raw_source_context,
        transformation_reuse_context=transformation_reuse_context,
    )
    selection_result = stage_a.selection_result
    selected_sources = stage_a.selected_sources
    source_models_info = stage_a.source_models_info
    source_scalers = stage_a.source_scalers
    source_train_tensors = stage_a.source_train_tensors
    knn_prior = stage_a.knn_prior
    input_shape_ref = stage_a.input_shape
    fused_params = stage_a.fused_params
    layer_names = stage_a.layer_names
    failed_sources = stage_a.failed_sources
    effective_k = stage_a.effective_k
    source_extractors = stage_a.source_extractors

    # --- Step 3: 融合初始化 target CNN，解冻 conv（对齐 No-TL 可训练底座） ---
    target_cnn = build_base_cnn(input_shape_ref, learning_rate=learning_rate)
    target_cnn = load_fused_params_into_target_model(target_cnn, fused_params)
    for layer in target_cnn.layers:
        layer.trainable = layer.name not in {"dense_out", "dropout"}
    target_extractor = build_flatten_feature_extractor(target_cnn, name="target_feature_extractor")

    # --- Step 4: target split + 归一化（保留 raw 帧供 per-source scaler 变换） ---
    raw_train_df, raw_val_df, raw_test_df = temporal_split_by_ratio_or_dates(target_df)
    raw_frames = {
        "train": raw_train_df.copy(),
        "val": raw_val_df.copy(),
        "test": raw_test_df.copy(),
    }
    target_train_df, target_val_df, target_test_df, target_scaler, target_feature_columns = normalize_features(
        raw_train_df, raw_val_df, raw_test_df, feature_columns=feature_cols,
        reuse_context=transformation_reuse_context,
    )

    def _windows(df: pd.DataFrame, use_reuse: bool) -> Tuple[np.ndarray, np.ndarray]:
        X, y = build_tabular_sequence(
            df, horizon=horizon, window_size=window_size, feature_columns=feature_cols,
            reuse_context=transformation_reuse_context if use_reuse else None,
        )
        return X, y

    X_train, y_train = _windows(target_train_df, use_reuse=True)
    X_val, y_val = _windows(target_val_df, use_reuse=True)
    X_test, y_test = _windows(target_test_df, use_reuse=True)
    if len(y_train) == 0:
        raise ValueError("Target train split produced zero windows; adjust window_size/horizon.")
    if len(y_test) == 0:
        raise ValueError("Target test split produced zero windows; adjust window_size/horizon.")
    X_train, X_val, X_test = to_cnn_tensor(X_train), (
        to_cnn_tensor(X_val) if len(y_val) else X_val
    ), to_cnn_tensor(X_test)

    # --- Step 5: source 专家特征一次性预计算；z_T 由可训练 E_T 在循环内计算 ---
    def _extract(extractor, tensor: np.ndarray) -> np.ndarray:
        return extractor(tf.constant(tensor, dtype=tf.float32), training=False).numpy()

    probe = _extract(target_extractor, X_train[:1])
    d_model = int(probe.shape[-1])

    f_t: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    for i, (scaler_i, extractor_i) in enumerate(zip(source_scalers, source_extractors)):
        for split_name in ("train", "val", "test"):
            raw_frame = raw_frames[split_name]
            scaled = _transform_with_scaler(raw_frame, scaler_i, target_feature_columns)
            X_scaled, y_scaled = build_tabular_sequence(
                scaled, horizon=horizon, window_size=window_size,
                feature_columns=feature_cols, reuse_context=None,
            )
            if split_name == "val" and len(y_scaled) == 0:
                f_t[split_name].append(None)
                continue
            f_t[split_name].append(_extract(extractor_i, to_cnn_tensor(X_scaled)))

    f_s_pools = [
        _extract(extractor_i, tensor_i)
        for extractor_i, tensor_i in zip(source_extractors, source_train_tensors)
    ]

    # --- Step 6: Stage B 训练 ---
    head_model = MSKATLLite(
        d_model=d_model,
        num_sources=effective_k,
        gamma=gamma,
        name="ms_katl_lite",
    )
    has_val = len(y_val) > 0 and all(f is not None for f in f_t["val"])
    training_summary = train_ms_katl_head(
        head_model,
        x_train=X_train,
        f_t_train=f_t["train"],
        y_train=y_train,
        f_s_pools=f_s_pools,
        knn_prior=knn_prior,
        target_extractor=target_extractor,
        x_val=X_val if has_val else None,
        f_t_val=f_t["val"] if has_val else None,
        y_val=y_val if has_val else None,
        epochs=target_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        use_adversarial=use_adversarial,
    )

    # --- Step 7: 评估（推理路径无 GRL/判别器） ---
    diagnostics: Dict[str, object] = {}
    diagnostics.update(validate_finite_array(X_test, name="X_test"))
    diagnostics.update(validate_finite_array(y_test, name="y_true", context=diagnostics))
    z_t_test = _extract(target_extractor, X_test)
    weight_diagnostics = summarize_model_weights(head_model)
    diagnostics.update(weight_diagnostics)
    extractor_weight_diagnostics = summarize_model_weights(target_extractor)
    if extractor_weight_diagnostics["model_weight_nan_count"] or extractor_weight_diagnostics["model_weight_inf_count"]:
        raise NonFiniteArrayError(
            "MS-KATL target extractor weights contain non-finite values: "
            f"nan_count={extractor_weight_diagnostics['model_weight_nan_count']} "
            f"inf_count={extractor_weight_diagnostics['model_weight_inf_count']}",
            diagnostics={**diagnostics, **extractor_weight_diagnostics},
        )
    if weight_diagnostics["model_weight_nan_count"] or weight_diagnostics["model_weight_inf_count"]:
        raise NonFiniteArrayError(
            "MS-KATL head weights contain non-finite values: "
            f"nan_count={weight_diagnostics['model_weight_nan_count']} "
            f"inf_count={weight_diagnostics['model_weight_inf_count']}",
            diagnostics=diagnostics,
        )

    y_pred = predict_ms_katl(head_model, z_t_test, f_t["test"], knn_prior)
    diagnostics.update(validate_finite_array(y_pred, name="y_pred", context=diagnostics))
    y_pred_flat = y_pred.flatten()
    y_true = np.asarray(y_test).flatten()
    eps = 1e-8
    rmse = float(np.sqrt(np.mean((y_pred_flat - y_true) ** 2)))
    accuracy = float(1.0 / (rmse + eps))
    smape_value = float(smape(y_true, y_pred_flat, epsilon=eps))

    _, alpha_test, g_test = head_model(
        tf.constant(z_t_test, dtype=tf.float32),
        [tf.constant(f, dtype=tf.float32) for f in f_t["test"]],
        tf.constant(np.repeat(knn_prior.reshape(1, -1), len(y_true), axis=0), dtype=tf.float32),
        training=False,
    )
    history = training_summary["history"]
    ms_katl_diagnostics = {
        "knn_prior": [float(v) for v in knn_prior],
        "gamma": float(gamma),
        "tau_k": float(tau_k),
        "effective_k": int(effective_k),
        "steps_per_epoch": int(training_summary["steps_per_epoch"]),
        "total_steps": int(training_summary["total_steps"]),
        "alpha_test_mean": [float(v) for v in tf.reduce_mean(alpha_test, axis=0).numpy()],
        "alpha_test_std": [float(v) for v in tf.math.reduce_std(alpha_test, axis=0).numpy()],
        "gate_test_mean": float(tf.reduce_mean(g_test).numpy()),
        "gate_test_std": float(tf.math.reduce_std(g_test).numpy()),
        "final_disc_acc": [
            history[f"disc_{i}_acc"][-1] if history[f"disc_{i}_acc"] else float("nan")
            for i in range(effective_k)
        ],
        "final_lambda": history["lambda"][-1] if history["lambda"] else float("nan"),
        "use_adversarial": bool(use_adversarial),
        "target_cnn_trainable": True,
        "final_forecast_loss": history["forecast_loss"][-1] if history["forecast_loss"] else float("nan"),
        "final_adv_loss": history["adv_loss"][-1] if history["adv_loss"] else float("nan"),
        "val_rmse_trajectory": history["val_rmse"],
        "disc_acc_trajectory": {
            f"disc_{i}": history[f"disc_{i}_acc"] for i in range(effective_k)
        },
        "alpha_bar_trajectory": {
            f"alpha_bar_{i}": history[f"alpha_bar_{i}"] for i in range(effective_k)
        },
        "gate_mean_trajectory": history["gate_mean"],
    }

    logger.info(
        "[run_ms_katl_lite] Finished. RMSE=%.4f sMAPE=%.4f gate_mean=%.3f alpha=%s",
        rmse, smape_value,
        ms_katl_diagnostics["gate_test_mean"],
        ms_katl_diagnostics["alpha_test_mean"],
    )

    return {
        "meta": {
            "method": METHOD_NAME,
            "k": int(k),
            "weight_mode": weight_mode,
            "feature_cols": list(feature_cols),
            "knn_feature_mode": (selection_result.get("meta", {}) or {}).get("knn_feature_mode", ""),
            "selected_sources": selected_sources,
            **runtime_selection_meta(selection_result),
            "fused_layers": list(layer_names),
            "ms_katl_diagnostics": ms_katl_diagnostics,
            **source_failure_meta(
                requested_k=k,
                selected_sources=selected_sources,
                valid_source_count=len(source_models_info),
                failed_sources=failed_sources,
            ),
        },
        "source_models_info": source_models_info,
        "fused_result": {
            "rmse": rmse,
            "accuracy": accuracy,
            "smape": smape_value,
            "y_true": y_true,
            "y_pred": y_pred,
            "prediction_shape": tuple(y_pred.shape),
            "sales_scaler": target_scaler,
            "feature_columns": target_feature_columns,
            **dict(metric_identity or {}),
            **{
                key: value
                for key, value in diagnostics.items()
                if key.endswith("_nan_count")
                or key.endswith("_inf_count")
                or key in {"X_test_shape", "y_true_shape", "y_pred_shape"}
            },
        },
        "frozen_layers": [l.name for l in target_cnn.layers if not l.trainable],
    }
