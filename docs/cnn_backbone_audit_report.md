# CNN 主干代码审查报告

审查日期：2026-06-15

审查范围：

- `cnn_model.py`
- `src/models/cnn_model.py`
- `src/models/no_tl_model.py`
- `msml_tl.py`
- `msml_tl_rfe.py`
- `single_source_tl.py`
- `mssb_tl.py`
- `mswa_tl.py`
- 相关调用检索：`build_base_cnn`、`build_cnn_ablation_variant`、`set_trainable_layers`、权重复制/加载、冻结逻辑

## 结论摘要

当前两份 CNN 主干文件内容完全一致，`diff -u cnn_model.py src/models/cnn_model.py` 无输出，SHA-256 也一致：

```text
55111b76052f859249b0f4392a5594a09497f1b25a93f074f951cd1b90b0b368  cnn_model.py
55111b76052f859249b0f4392a5594a09497f1b25a93f074f951cd1b90b0b368  src/models/cnn_model.py
```

因此，当前工作区不存在“同一次实验中不同方法实际使用了不同 CNN 架构”的已发生问题。但存在维护风险：No-TL 入口使用 `src.models.cnn_model`，SS-TL/MSWA-TL/MSSB-TL/MSML-TL/MSML-TL-RFE 使用根目录 `cnn_model`。只要未来只修改其中一份，实验会静默分叉。

模型架构本身可运行，`(10, 7)` 输入下实际层输出为：

```text
input_layer: (None, 10, 7)
conv1:       (None, 10, 32)
pool1:       (None, 5, 32)
conv2:       (None, 5, 64)
pool2:       (None, 2, 64)
conv3:       (None, 2, 128)
flatten:     (None, 256)
dense_out:   (None, 1)
```

`kernel_size=3` 的第三层卷积在长度 2 上使用 `padding="same"` 不会报错，但大部分卷积窗口依赖 padding，时序表达已经很短。对只有 15-30 个目标训练样本的场景，当前约 31k-32k 参数、无 Dropout/BatchNorm/权重正则、默认 100 epochs/lr=0.001 的组合有明显过拟合和训练不稳定风险。

## 问题清单

### Critical

无已确认 Critical。未发现当前两份 CNN 文件内容不一致、层名冻结立即失效、或指定调用点直接调用错误版本导致当前实验必然不可用的问题。

### Warning

1. 代码重复导致未来实验静默分叉风险

   - `src/models/no_tl_model.py:9` 从 `src.models.cnn_model` 导入。
   - `single_source_tl.py:18`、`msml_tl.py:30`、`msml_tl_rfe.py:33` 从根目录 `cnn_model` 导入。
   - `mswa_tl.py` 和 `mssb_tl.py` 不直接导入 `build_base_cnn`，而是通过 `single_source_tl.py` 间接使用根目录版本。
   - 当前两份文件完全一致，所以现在没有实际分叉；但这是高风险维护模式。

2. 小样本下模型容量偏大，过拟合风险高

   - `build_base_cnn` 包含三层卷积和一个全连接输出层。
   - 参数量公式约为 `31201 + 96 * num_features`。当 `num_features=3-12` 时，参数量约为 31,489 到 32,353。
   - 目标训练样本只有 15-30 个时，参数/样本比极高。
   - 主路径没有 Dropout、BatchNorm、L2、EarlyStopping、学习率衰减或梯度裁剪。

3. 两次 MaxPool 后第三层卷积的时序长度过短

   - 时间维度从 `10 -> 5 -> 2`。
   - `Conv1D(kernel_size=3, padding="same")` 在长度 2 的序列上合法，但时序上下文很有限，padding 影响较大。
   - 这不是运行时 bug，但作为时序主干偏激进，尤其不适合用来支撑所有迁移方法的可信结论。

4. 线性输出可能产生负值或超过归一化范围

   - `Dense(1, name="dense_out")` 无激活函数。
   - 回归任务使用线性输出本身合理，尤其销量反归一化评估时可以避免 ReLU 在 0 附近截断梯度。
   - 但如果 `sales` 标签经 MinMax 到 `[0,1]`，线性输出会允许预测小于 0 或大于 1，反归一化后可能产生负销量或超出训练范围的销量。
   - 建议不要盲目改成 ReLU；更稳妥是保留线性输出并在评估/业务输出阶段做 clipping，或实验比较 `sigmoid`/clipped prediction 是否改善。

5. 主路径 Adam 没有梯度裁剪，且学习率固定为 0.001

   - `build_base_cnn` 使用 `Adam(learning_rate=learning_rate)`，默认 `0.001`，无 `clipnorm`。
   - 小样本、短序列、不同数据集销量量纲差异大的场景中，如果标签没有被稳定归一化，MSE 梯度会受量纲强烈影响。
   - 目前预处理路径会对选中的数值列做 MinMax；如果 `sales` 包含在特征列中，标签也来自归一化后的 `sales`，风险降低，但仍建议在主路径暴露 `clipnorm` 和较小 lr 的配置。

6. MSML 系列权重融合要求输入特征数一致

   - `conv1` 权重形状是 `(3, num_features, 32)`，不同特征数之间不能直接 `set_weights` 或加权平均。
   - `msml_tl.py:701-707` 检查 source 模型之间的 `input_shape` 一致。
   - `msml_tl.py:724` 用 source 的 `input_shape_ref` 构建 target model，但没有在构建前显式检查 target tensor shape 是否等于 `input_shape_ref`；若目标特征列实际变化，错误会延迟到 fine-tune 或 set_weights。
   - `msml_tl_rfe.py` 通过统一 `selected_feature_cols` 降低了该风险，但仍依赖特征投影逻辑正确。

7. Ablation 变体含义部分是元数据/训练配置，不是模型结构

   - `change1_batch_size_1` 在 `build_cnn_ablation_variant` 中返回与 original 相同的模型；它只有在调用方使用 `resolve_cnn_ablation_training_config(...).effective_batch_size` 传给 `fit` 时才生效。
   - `src/experiment/run_no_tl_experiment.py:56-70` 正确使用了 `effective_batch_size`。
   - 若其他调用方只调用 `build_cnn_ablation_variant` 而不使用 training config，`change1_batch_size_1` 不会有实际作用。
   - `change2_no_batch_norm` 与 original 结构等价，因为当前主干本来没有 BatchNorm；它是历史/审计占位，不是有效结构消融。

8. `change3_low_lr_clipnorm` 分支重复 compile

   - `cnn_model.py:167` 先调用 `build_base_cnn`，该函数内部已经 compile。
   - `cnn_model.py:168-172` 立即再次 compile 为 `Adam(learning_rate=1e-4, clipnorm=1.0)`。
   - 对新建模型而言通常无害，因为还没有训练状态；但这是冗余逻辑，容易让后续维护误解。

### Info

1. `loss="mse", metrics=["mae"]` 对回归任务是合理默认

   - MSE 对大误差敏感，适合作为训练损失。
   - MAE 作为指标可解释性好。
   - 若论文或业务主要报告 RMSE/MAE/MAPE，应保证评估空间一致：归一化空间 vs 原始销量空间。

2. Python 版本类型标注目前可兼容 Python 3.9+

   - 两份 CNN 文件均有 `from __future__ import annotations`。
   - `clipnorm: float | None` 不会在 Python 3.9 中因注解求值时报错。

3. `get_model_summary_dict` 未被主流程调用

   - 全仓检索只发现定义，没有调用。
   - `tf.size(w).numpy()` 在 eager mode 下正常；若未来在 graph mode 中调用，可能需要改成纯 Keras/NumPy 参数统计。

## 逐项检查详情

### 一、代码重复问题

两份文件完全一致：

- `cnn_model.py`
- `src/models/cnn_model.py`

`diff -u` 无差异，SHA-256 相同。当前没有 diff 可列。

导入路径如下：

| 模块 | 导入/调用路径 | 实际 CNN 版本 |
| --- | --- | --- |
| No-TL | `src/models/no_tl_model.py:9` -> `src.models.cnn_model` | `src/models/cnn_model.py` |
| SS-TL | `single_source_tl.py:18` -> `cnn_model` | 根目录 `cnn_model.py` |
| MSWA-TL | `mswa_tl.py:32-37` -> `single_source_tl` -> `cnn_model` | 根目录 `cnn_model.py` |
| MSSB-TL | `mssb_tl.py:32-36` -> `single_source_tl` -> `cnn_model` | 根目录 `cnn_model.py` |
| MSML-TL | `msml_tl.py:30` -> `cnn_model` | 根目录 `cnn_model.py` |
| MSML-TL-RFE | `msml_tl_rfe.py:33` -> `cnn_model` | 根目录 `cnn_model.py` |

判断：当前不存在不同 CNN 版本导致的实验结果分叉；未来存在明显风险。建议只保留一个权威实现，另一处改成兼容转发。

### 二、模型架构检查

当前 `build_base_cnn`：

```text
Input
-> Conv1D(32, kernel_size=3, padding="same", activation="relu", name="conv1")
-> MaxPooling1D(pool_size=2, name="pool1")
-> Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv2")
-> MaxPooling1D(pool_size=2, name="pool2")
-> Conv1D(128, kernel_size=3, padding="same", activation="relu", name="conv3")
-> Flatten(name="flatten")
-> Dense(1, name="dense_out")
```

对 `lookback_window=10`：

- `pool1`: `10 -> 5`
- `pool2`: `5 -> 2`
- `conv3`: 长度保持 `2`
- `flatten`: `2 * 128 = 256`

该结构可以运行，但对短序列和极小训练集偏重。第三层卷积合法，不是 bug；问题是时间维度压缩过快，模型容量相对样本数过大，缺少正则化。

最后一层线性输出：

- 优点：标准回归输出，梯度不受非负激活截断。
- 风险：可能输出负销量或超出 `[0,1]` 的归一化销量。
- 建议：研究实验中优先保留线性输出以避免改变基线定义；如果需要业务约束，用预测后 clipping 或单独 ablation 比较 `sigmoid`/`softplus`/`relu`。

损失和指标：

- `mse` 作为训练损失合理。
- `mae` 作为训练监控合理。
- 建议同时在评估报告中固定输出 RMSE/MAE，并明确是在归一化空间还是原始销量空间。

### 三、数值稳定性检查

学习率：

- `0.001` 是 Adam 常用默认值，但对 15-30 个样本、100 epochs、约 31k 参数的模型偏激进。
- 建议把主路径学习率纳入实验网格：`1e-3`, `3e-4`, `1e-4`，并记录随机种子重复实验的均值/方差。

梯度裁剪：

- 主路径无 `clipnorm`。
- 只有 `_build_current_3layer_cnn_no_batch_norm(..., clipnorm=None)` 支持裁剪，`change3_low_lr_clipnorm` 用了 `clipnorm=1.0`。
- 建议让 `build_base_cnn` 支持可选 `clipnorm`，默认仍为 `None` 以保持兼容。

权重初始化：

- Keras 默认 `glorot_uniform` 对 ReLU Conv/Dense 不是最优但可用。
- 对 ReLU 卷积，`he_normal`/`he_uniform` 往往更匹配；不过小样本下初始化差异可能造成方差，需要多 seed 验证。

输入归一化：

- `normalize_features` 对数值特征做 MinMax，fit 于 train+val，transform train/val/test。
- `build_tabular_sequence` 的 `y` 来自当前 DataFrame 的 `sales` 列，因此如果 `sales` 在归一化特征列中，标签也是归一化后的销量。
- 如果未来某条路径绕过归一化，MSE 和 Adam 梯度会直接受销量量纲影响，Dataset3 几千级销量会主导训练动态。

### 四、与 MSML 系列的集成检查

层名匹配：

- 实际层名：`conv1`, `pool1`, `conv2`, `pool2`, `conv3`, `flatten`, `dense_out`。
- `msml_tl.py:45-46` 定义：
  - `_DEFAULT_TRANSFERABLE_LAYERS = ["conv1", "pool1", "conv2", "pool2"]`
  - `_DEFAULT_FUSION_LAYERS = ["conv1", "conv2"]`
- `msml_tl_rfe.py:56-57` 同样定义。
- 名称匹配，不会因名字不一致导致冻结失效。

冻结实现：

- `set_trainable_layers(model, trainable_layer_names)` 在两份 CNN 文件中定义，但全仓没有调用。
- MSML 实际使用 `freeze_fused_layers(target_model, freeze_layer_names)`，把传入名字设置为 `trainable=False`，其他层设为 `True`。
- SS-TL/MSWA/MSSB 通过 `freeze_first_n_layers=4` 冻结前四个非 Input 层，即 `conv1`, `pool1`, `conv2`, `pool2`，与 MSML 默认一致。

权重加载/复制：

- 当前主路径没有使用 Keras `model.load_weights`。
- SS-TL 使用 `zip(source_model.layers, target_model.layers)` 后逐层 `set_weights`。
- MSML 使用 `extract_layer_params` + 加权平均 + 按层名 `set_weights`。
- 这要求对应层结构和权重 shape 一致。特别是 `conv1` 依赖 `num_features`，source/target 特征数不同会失败。

重复编译：

- `build_cnn_ablation_variant("change3_low_lr_clipnorm")` 的重复 compile 无直接训练错误，但应清理为单次 compile，避免维护误解。

### 五、Ablation 变体逻辑检查

`change1_batch_size_1`：

- 模型结构不变。
- 是否生效取决于 fit 时是否使用 `effective_batch_size=1`。
- `src/experiment/run_no_tl_experiment.py:56-70` 当前正确使用。

`change2_no_batch_norm`：

- 当前 CNN 原本没有 BatchNorm。
- 该变体与 original 结构等价，只能作为历史审计标签，不是有效消融。

`change3_low_lr_clipnorm`：

- 实际改变学习率到 `1e-4` 并加 `clipnorm=1.0`。
- 当前通过重复 compile 实现，逻辑有效但冗余。

`change123_all`：

- 元数据上组合了 batch size 1、无 BatchNorm、低 lr/clipnorm。
- 模型构建使用 `_build_current_3layer_cnn_no_batch_norm(..., learning_rate=1e-4, clipnorm=1.0)`，训练 batch size 仍需调用方使用 `effective_batch_size=1`。
- 在 No-TL 当前 runner 中能生效；在只构建模型的调用中 batch size 不会体现。

### 六、潜在 Bug 检查

类型标注：

- 两份 CNN 文件都有 `from __future__ import annotations`。
- `clipnorm: float | None` 在 Python 3.9 环境中不会因注解求值失败。

`get_model_summary_dict`：

- 当前未被主流程调用。
- eager mode 下 `tf.size(w).numpy()` 正常。
- 若未来在 graph mode、`tf.function`、或禁用 eager 的环境中调用，建议改成：

```python
trainable_params = int(
    sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
)
```

## 修复建议

### 1. 消除双实现分叉

推荐保留 `src/models/cnn_model.py` 为唯一权威实现，根目录 `cnn_model.py` 改成兼容转发：

```python
"""Compatibility wrapper for the canonical CNN implementation."""

from src.models.cnn_model import (
    CNN_ABLATION_VARIANTS,
    CnnAblationTrainingConfig,
    build_base_cnn,
    build_cnn_ablation_variant,
    get_model_summary_dict,
    resolve_cnn_ablation_training_config,
    set_trainable_layers,
)
```

这样老代码 `from cnn_model import build_base_cnn` 仍可运行，但不会出现两份实现漂移。

### 2. 给主路径增加可选数值稳定参数

保持默认行为不变，但允许实验显式打开：

```python
def build_base_cnn(input_shape, learning_rate=0.001, clipnorm=None):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = MaxPooling1D(2, name="pool1")(x)
    x = Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = MaxPooling1D(2, name="pool2")(x)
    x = Conv1D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = Flatten(name="flatten")(x)
    outputs = Dense(1, name="dense_out")(x)

    optimizer_kwargs = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(**optimizer_kwargs),
        loss="mse",
        metrics=["mae"],
    )
    return model
```

### 3. 为小样本建立更轻量的正式对照主干

建议新增一个明确命名的轻量变体，而不是直接替换当前基线：

```text
Input
-> Conv1D(16, kernel_size=3, padding="same", activation="relu")
-> GlobalAveragePooling1D()
-> Dense(16, activation="relu", kernel_regularizer=l2(1e-4))
-> Dropout(0.1)
-> Dense(1)
```

原因：

- 避免两次池化把长度 10 压到 2。
- 用 `GlobalAveragePooling1D` 替代 `Flatten`，减少参数量。
- 用少量 L2/Dropout 降低目标小样本过拟合。
- 保留线性输出，预测后再按需要 clipping。

### 4. 在 MSML 构建 target 前显式检查 target shape

在 `msml_tl.py` 中，构建 target model 前可从 target train 构造一次 tensor 并检查：

```python
target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)
target_train_df, target_val_df, target_test_df, target_scaler, target_feature_columns = normalize_features(
    target_train_df, target_val_df, target_test_df,
)
X_target_train, _ = build_tabular_sequence(target_train_df, horizon=horizon, window_size=window_size)
X_target_train = to_cnn_tensor(X_target_train)
if tuple(X_target_train.shape[1:]) != tuple(input_shape_ref):
    raise ValueError(
        f"Target input shape mismatch: target={X_target_train.shape[1:]} source_ref={input_shape_ref}"
    )
target_model = build_base_cnn(input_shape_ref, learning_rate=learning_rate)
```

这能把特征数不一致的问题提前变成清晰错误。

### 5. 清理 ablation 语义

建议：

- 在文档和结果表中明确 `change1_batch_size_1` 是训练配置消融，不是模型结构消融。
- 将 `change2_no_batch_norm` 标记为 `no_op_current_backbone` 或移除，避免误导为有效结构变化。
- 将 `change3_low_lr_clipnorm` 改成调用带 `clipnorm` 参数的 `build_base_cnn`，避免重复 compile。

## 与 MSML 集成风险结论

当前没有“不同模块用了不同 CNN 版本”的实际结果风险，因为根目录和 `src/models` 两份 `cnn_model.py` 完全一致。

但存在明确的维护风险：

- No-TL 和审计脚本偏向 `src.models.cnn_model`。
- SS-TL、MSWA-TL、MSSB-TL、MSML-TL、MSML-TL-RFE 偏向根目录 `cnn_model`。
- 如果后续只修改其中一份，No-TL 与迁移学习方法会立刻使用不同主干，且不会自动报错。

最高优先级修复是统一 CNN 主干来源，并加一个测试断言根目录兼容包装与 `src.models.cnn_model` 指向同一实现。

