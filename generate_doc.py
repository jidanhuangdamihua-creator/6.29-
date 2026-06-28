content = """
================================================================================
数据处理主流程调用顺序说明文档
生成日期：2026年4月6日
================================================================================

【步骤1】程序入口 / 解析命令行参数
- 文件：run_main_experiment.py
- 函数：main() → parse_args()
- 作用：程序唯一入口。解析命令行参数（--dataset、--model、--mode、--debug），
        然后创建 ExperimentRunner 实例并调用 run()。
- 输入：命令行参数，例如：
        --dataset dataset1 --model lstm --mode train
- 输出：argparse.Namespace 对象（args），含 dataset/model/mode/debug 四个字段
- 下一步调用：ExperimentRunner.__init__() → ExperimentRunner.run()

────────────────────────────────────────────────────────────────────────────────

【步骤2】初始化实验运行器 + 加载配置
- 文件：experiment_runner.py
- 函数：ExperimentRunner.__init__()
- 作用：保存实验参数，并通过 get_config() 加载对应数据集的配置字典。
- 输入：dataset_name（str）、model_type（str）、mode（str）、debug（bool）
- 输出：初始化完成的 ExperimentRunner 对象（含 self.config）
- 下一步调用：get_config(dataset_name)

────────────────────────────────────────────────────────────────────────────────

【步骤3】读取数据集配置字典
- 文件：config.py
- 函数：get_config(dataset_name)
- 作用：根据 dataset_name 从 CONFIGS 字典中取出对应配置，包含数据路径、
        分割方式、归一化方法、序列窗口参数、训练超参数等。
- 输入：dataset_name（str），可选 'dataset1' / 'dataset2' / 'dataset3'
- 输出：config 字典（dict），含 data / rename_map / model / training /
        features / normalization / sequence 等子键
- 下一步调用：（返回给 ExperimentRunner.__init__，后续交给 run() 使用）

────────────────────────────────────────────────────────────────────────────────

【步骤4】执行实验主流程
- 文件：experiment_runner.py
- 函数：ExperimentRunner.run()
- 作用：顺序执行三大步骤：① 数据加载与预处理，② 模型构建，③ 训练或评估。
- 输入：self（含 dataset_name、model_type、mode、config）
- 输出：无直接返回值（训练后保存模型文件，评估后打印指标）
- 下一步调用：
    load_dataset()          ← 数据加载主入口
    get_model()             ← 模型构建
    Trainer.fit() 或 Evaluator.evaluate()  ← 训练或评估

────────────────────────────────────────────────────────────────────────────────

【步骤5】统一数据加载入口（分发路由）
- 文件：data_preprocessing.py
- 函数：load_dataset(dataset_name, config)
- 作用：根据 dataset_name 将请求分发给三个私有加载函数之一：
        dataset1 → _load_pasta_dataset
        dataset2 → _load_event_dataset
        dataset3 → _load_sensor_dataset
- 输入：dataset_name（str）、config（dict）
- 输出：(train_loader, val_loader, test_loader, meta)
        meta 含 input_dim / output_dim / scaler
- 下一步调用（分支）：
    若 dataset_name == 'dataset1' → _load_pasta_dataset(config)
    若 dataset_name == 'dataset2' → _load_event_dataset(config)
    若 dataset_name == 'dataset3' → _load_sensor_dataset(config)

════════════════════════════════════════════════════════════════════════════════
                      ▼ Dataset1 专属分支
════════════════════════════════════════════════════════════════════════════════

【步骤6-A】Dataset1：Pasta 工厂数据加载主函数
- 文件：data_preprocessing.py
- 函数：_load_pasta_dataset(config)
- 作用：协调 Dataset1 的完整预处理流程：读取 CSV → 标准化列名 →
        确保基础列存在 → 提取时间特征 → 拆分数据集 → 归一化 → 构建序列 → 封装 DataLoader。
- 输入：config（dict），含 data.raw_path 等
- 输出：(train_loader, val_loader, test_loader, meta)
- 下一步调用（顺序）：
    1. pd.read_csv(raw_path)
    2. _standardize_pasta_dataset(df, config)
    3. _ensure_base_columns(df)
    4. extract_datetime_features(df)
    5. build_source_target_split(df, config)
    6. temporal_split_by_ratio_or_dates(train_df, config)
    7. normalize_features(train_df, val_df, test_df, config)
    8. build_tabular_sequence(...)  ×3
    9. _make_loader(...)  ×3

【步骤7-A】Dataset1：列名标准化
- 文件：data_preprocessing.py
- 函数：_standardize_pasta_dataset(df, config)
- 作用：按 config['rename_map'] 对原始列名进行重命名，
        将 time→timestamp、machine→machine_id、label→target。
- 输入：df（原始 DataFrame）、config（dict）
- 输出：列名已统一的 DataFrame
- 下一步调用：（返回 df，交回 _load_pasta_dataset）

【步骤8-A】Dataset1：确保基础列存在
- 文件：data_preprocessing.py
- 函数：_ensure_base_columns(df)
- 作用：检查 timestamp / machine_id / target 三列是否存在，
        缺失列用 NaN 填充，防止下游函数报错。
- 输入：df（DataFrame）
- 输出：补全缺失列后的 DataFrame
- 下一步调用：（返回 df，交回 _load_pasta_dataset）

════════════════════════════════════════════════════════════════════════════════
                      ▼ Dataset2 专属分支
════════════════════════════════════════════════════════════════════════════════

【步骤6-B】Dataset2：事件日志数据加载主函数
- 文件：data_preprocessing.py
- 函数：_load_event_dataset(config)
- 作用：协调 Dataset2 的完整预处理流程：读取 CSV → 标准化列名 →
        过滤无效事件 → 提取时间特征 → 拆分数据集 → 归一化 → 构建序列 → 封装 DataLoader。
- 输入：config（dict）
- 输出：(train_loader, val_loader, test_loader, meta)
- 下一步调用（顺序）：
    1. pd.read_csv(raw_path)
    2. _standardize_event_dataset(df, config)
    3. _filter_valid_events(df)
    4. extract_datetime_features(df)
    5. build_source_target_split(df, config)
    6. temporal_split_by_ratio_or_dates(train_df, config)
    7. normalize_features(train_df, val_df, test_df, config)
    8. build_tabular_sequence(...)  ×3
    9. _make_loader(...)  ×3

【步骤7-B】Dataset2：列名标准化
- 文件：data_preprocessing.py
- 函数：_standardize_event_dataset(df, config)
- 作用：按 config['rename_map'] 重命名，将 evt_time→timestamp、
        evt_type→event_type、result→target。
- 输入：df（原始 DataFrame）、config（dict）
- 输出：列名已统一的 DataFrame
- 下一步调用：（返回 df）

【步骤8-B】Dataset2：过滤无效事件行
- 文件：data_preprocessing.py
- 函数：_filter_valid_events(df)
- 作用：删除 event_type 或 timestamp 为空的行，保证后续时间解析不出错。
- 输入：df（DataFrame）
- 输出：过滤后的 DataFrame
- 下一步调用：（返回 df）

════════════════════════════════════════════════════════════════════════════════
                      ▼ Dataset3 专属分支
════════════════════════════════════════════════════════════════════════════════

【步骤6-C】Dataset3：传感器时序数据加载主函数
- 文件：data_preprocessing.py
- 函数：_load_sensor_dataset(config)
- 作用：协调 Dataset3 的完整预处理流程：读取 CSV → 标准化列名 →
        时间重采样 → 提取时间特征 → 拆分数据集 → 归一化 → 构建序列 → 封装 DataLoader。
- 输入：config（dict）
- 输出：(train_loader, val_loader, test_loader, meta)
- 下一步调用（顺序）：
    1. pd.read_csv(raw_path)
    2. _standardize_sensor_dataset(df, config)
    3. _resample_sensor_data(df, config)
    4. extract_datetime_features(df)
    5. build_source_target_split(df, config)
    6. temporal_split_by_ratio_or_dates(train_df, config)
    7. normalize_features(train_df, val_df, test_df, config)
    8. build_tabular_sequence(...)  ×3
    9. _make_loader(...)  ×3

【步骤7-C】Dataset3：列名标准化
- 文件：data_preprocessing.py
- 函数：_standardize_sensor_dataset(df, config)
- 作用：按 config['rename_map'] 重命名，将 ts→timestamp、
        sensor_label→target。
- 输入：df（原始 DataFrame）、config（dict）
- 输出：列名已统一的 DataFrame
- 下一步调用：（返回 df）

【步骤8-C】Dataset3：传感器数据时间重采样
- 文件：data_preprocessing.py
- 函数：_resample_sensor_data(df, config)
- 作用：将 timestamp 列转为 datetime 类型，然后以 config['data']['resample_freq']
        指定的频率（Dataset3 为 '5T' = 5分钟）进行重采样，
        用前向填充（ffill）补全缺失时间戳对应的传感器值。
- 输入：df（DataFrame）、config（dict）
- 输出：时间对齐后的 DataFrame（行数可能增多）
- 下一步调用：（返回 df）

════════════════════════════════════════════════════════════════════════════════
                      ▼ 三条分支在此汇合（共用步骤）
════════════════════════════════════════════════════════════════════════════════

【步骤9】提取时间特征（三条分支共用）
- 文件：utils/feature_engineering.py
- 函数：extract_datetime_features(df)
- 作用：从 timestamp 列派生出五个时间特征列：
        hour（小时）、minute（分钟）、dayofweek（星期几，0=周一）、
        month（月份）、is_weekend（是否周末，1/0整数）。
- 输入：df（含 timestamp 列的 DataFrame）
- 输出：新增五列后的 DataFrame（copy，不修改原始对象）
- 下一步调用：（返回 df，交回各 _load_xxx_dataset）

────────────────────────────────────────────────────────────────────────────────

【步骤10】划分训练+验证集 / 测试集（三条分支共用）
- 文件：utils/split_utils.py
- 函数：build_source_target_split(df, config)
- 作用：将全量数据划分为"训练+验证"和"测试"两部分。
        分支逻辑由 config['data']['split_method'] 控制：
        - 'ratio' 模式（Dataset1、Dataset3）：
            按 test_ratio 从末尾截取，保持时序顺序。
            Dataset1: test_ratio=0.2（末尾20%为测试集）
            Dataset3: test_ratio=0.15（末尾15%为测试集）
        - 'date' 模式（Dataset2）：
            test_start_date='2023-10-01' 之后的数据为测试集。
- 输入：df（DataFrame）、config（dict）
- 输出：(train_val_df, test_df) 两个 DataFrame
- 下一步调用：temporal_split_by_ratio_or_dates(train_val_df, config)

────────────────────────────────────────────────────────────────────────────────

【步骤11】从训练+验证集中再划分出验证集（三条分支共用）
- 文件：utils/split_utils.py
- 函数：temporal_split_by_ratio_or_dates(df, config)
- 作用：将 train+val 数据进一步拆分为 train 和 val。
        分支逻辑同上，由 split_method 控制：
        - 'ratio' 模式（Dataset1、Dataset3）：
            Dataset1: val_ratio=0.1（剩余数据末尾10%为验证集）
            Dataset3: val_ratio=0.1
        - 'date' 模式（Dataset2）：
            val_start_date='2023-09-01' 之后、test_start 之前为验证集。
- 输入：df（train+val DataFrame）、config（dict）
- 输出：(train_df, val_df) 两个 DataFrame
- 下一步调用：normalize_features(train_df, val_df, test_df, config)

────────────────────────────────────────────────────────────────────────────────

【步骤12】数值特征归一化（三条分支共用）
- 文件：utils/normalization.py
- 函数：normalize_features(train_df, val_df, test_df, config)
- 作用：对 config['features']['numeric'] 中指定的数值特征列进行归一化。
        关键：scaler 只在训练集上 fit_transform，
        验证集和测试集只做 transform（防止数据泄露）。
        归一化方法由 config['normalization']['method'] 控制：
        - 'standard'（Dataset1、Dataset3）：StandardScaler，零均值单位方差
        - 'minmax'（Dataset2）：MinMaxScaler，缩放到 [0, 1]
        归一化的列：
        Dataset1: ['temp', 'pressure', 'vibration']
        Dataset2: ['duration', 'count', 'score']
        Dataset3: ['channel_1', 'channel_2', 'channel_3', 'channel_4']
- 输入：train_df、val_df、test_df（DataFrame）、config（dict）
- 输出：(归一化后的 train_df, val_df, test_df, scaler对象)
- 下一步调用：build_tabular_sequence(...)  ×3

────────────────────────────────────────────────────────────────────────────────

【步骤13】构建滑动窗口序列（三条分支共用，分别对 train/val/test 各调用一次）
- 文件：utils/sequence_builder.py
- 函数：build_tabular_sequence(df, config)
- 作用：用滑动窗口将 DataFrame 转换为三维张量 (N, T, F)，作为时序模型输入。
        特征列 = config['features']['numeric'] + config['features']['datetime']
        每个样本：取连续 window_size 行作为 X，下一时刻的 target 列值作为 y。
        步长 stride 控制窗口移动间隔：
        Dataset1: window_size=24, stride=1
        Dataset2: window_size=12, stride=1
        Dataset3: window_size=48, stride=6
- 输入：df（归一化后的 DataFrame）、config（dict）
- 输出：(X, y)，其中：
        X: np.ndarray, shape=(N, window_size, num_features)
        y: np.ndarray, shape=(N,)
- 下一步调用：_make_loader(seq_data, config, shuffle=...)

────────────────────────────────────────────────────────────────────────────────

【步骤14】封装为 PyTorch DataLoader（三条分支共用，分别对 train/val/test 各调用一次）
- 文件：data_preprocessing.py
- 函数：_make_loader(seq_data, config, shuffle=False)
- 作用：将 (X, y) numpy 数组转换为 torch.float32 张量，
        包装为 TensorDataset，再封装为 DataLoader。
        train_loader: shuffle=True；val_loader/test_loader: shuffle=False。
        batch_size 取自 config['training']['batch_size']：
        Dataset1: 64，Dataset2: 32，Dataset3: 128
- 输入：seq_data=(X, y)（numpy tuple）、config（dict）、shuffle（bool）
- 输出：torch.utils.data.DataLoader 对象
- 下一步调用：（返回给各 _load_xxx_dataset，最终回到 ExperimentRunner.run()）

════════════════════════════════════════════════════════════════════════════════
                      ▼ 模型构建与训练/评估
════════════════════════════════════════════════════════════════════════════════

【步骤15】构建模型实例
- 文件：models/model_factory.py
- 函数：get_model(model_type, input_dim, output_dim, config)
- 作用：根据 model_type 参数选择模型类并实例化：
        - 'lstm'       → LSTMModel（lstm_model.py）
        - 'gru'        → GRUModel（gru_model.py）
        - 'transformer'→ TransformerModel（transformer_model.py）
        hidden_dim 和 num_layers 从 config['training'] 中读取，
        默认值：hidden_dim=64，num_layers=2。
- 输入：model_type（str）、input_dim（int）、output_dim（int）、config（dict）
- 输出：nn.Module 子类实例（模型对象）
- 下一步调用：Trainer(model, config) 或 Evaluator(model, config)

────────────────────────────────────────────────────────────────────────────────

【步骤16-A】训练模式：执行模型训练
- 文件：utils/trainer.py
- 函数：Trainer.fit(train_loader, val_loader)
- 作用：按 config['training']['epochs'] 轮次训练模型：
        - 每轮在 train_loader 上前向传播 + 反向传播（Adam 优化器，MSELoss）
        - 每轮调用 _validate(val_loader) 计算验证损失
        - 若验证损失优于历史最优，则保存模型到 checkpoints/best_model.pt
- 输入：train_loader（DataLoader）、val_loader（DataLoader）
- 输出：无直接返回值；最优模型已保存到磁盘
- 下一步调用：
    Trainer._validate(val_loader)  ← 每轮内部校验
    Trainer.save_model()           ← 训练完成后手动保存

【步骤16-A-内部】验证损失计算
- 文件：utils/trainer.py
- 函数：Trainer._validate(val_loader)
- 作用：在验证集上以 eval 模式（no_grad）前向传播，计算平均 MSE 损失。
- 输入：val_loader（DataLoader）
- 输出：float（平均验证损失）
- 下一步调用：（结果返回给 fit()）

────────────────────────────────────────────────────────────────────────────────

【步骤16-B】评估模式：加载模型并评估
- 文件：utils/evaluator.py
- 函数：Evaluator.load_model() → Evaluator.evaluate(test_loader)
- 作用：
    load_model：从 checkpoints/best_model.pt 加载权重。
    evaluate：在测试集上推理，计算并打印 MSE 和 MAE 指标，返回指标字典。
- 输入：test_loader（DataLoader）
- 输出：{'mse': float, 'mae': float}
- 下一步调用：（无，流程结束）

================================================================================
调用链总表（按执行顺序）
================================================================================

main()                                          [run_main_experiment.py]
  └── parse_args()                              [run_main_experiment.py]
  └── ExperimentRunner.__init__()               [experiment_runner.py]
        └── get_config(dataset_name)            [config.py]
  └── ExperimentRunner.run()                    [experiment_runner.py]
        ├── load_dataset(dataset_name, config)  [data_preprocessing.py]
        │     ├── 分支 dataset1:
        │     │   └── _load_pasta_dataset(config)
        │     │         ├── pd.read_csv()
        │     │         ├── _standardize_pasta_dataset(df, config)
        │     │         ├── _ensure_base_columns(df)
        │     │         ├── extract_datetime_features(df)       [utils/feature_engineering.py]
        │     │         ├── build_source_target_split(df, config) [utils/split_utils.py]
        │     │         ├── temporal_split_by_ratio_or_dates(train_df, config) [utils/split_utils.py]
        │     │         ├── normalize_features(train_df, val_df, test_df, config) [utils/normalization.py]
        │     │         ├── build_tabular_sequence(train_data, config)  [utils/sequence_builder.py]
        │     │         ├── build_tabular_sequence(val_data, config)    [utils/sequence_builder.py]
        │     │         ├── build_tabular_sequence(test_data, config)   [utils/sequence_builder.py]
        │     │         ├── _make_loader(train_seq, config, shuffle=True)
        │     │         ├── _make_loader(val_seq, config, shuffle=False)
        │     │         └── _make_loader(test_seq, config, shuffle=False)
        │     ├── 分支 dataset2:
        │     │   └── _load_event_dataset(config)
        │     │         ├── pd.read_csv()
        │     │         ├── _standardize_event_dataset(df, config)
        │     │         ├── _filter_valid_events(df)
        │     │         ├── extract_datetime_features(df)       [utils/feature_engineering.py]
        │     │         ├── build_source_target_split(df, config) [utils/split_utils.py]
        │     │         ├── temporal_split_by_ratio_or_dates(train_df, config) [utils/split_utils.py]
        │     │         ├── normalize_features(train_df, val_df, test_df, config) [utils/normalization.py]
        │     │         ├── build_tabular_sequence(train_data, config)  [utils/sequence_builder.py]
        │     │         ├── build_tabular_sequence(val_data, config)    [utils/sequence_builder.py]
        │     │         ├── build_tabular_sequence(test_data, config)   [utils/sequence_builder.py]
        │     │         ├── _make_loader(train_seq, config, shuffle=True)
        │     │         ├── _make_loader(val_seq, config, shuffle=False)
        │     │         └── _make_loader(test_seq, config, shuffle=False)
        │     └── 分支 dataset3:
        │         └── _load_sensor_dataset(config)
        │               ├── pd.read_csv()
        │               ├── _standardize_sensor_dataset(df, config)
        │               ├── _resample_sensor_data(df, config)
        │               ├── extract_datetime_features(df)       [utils/feature_engineering.py]
        │               ├── build_source_target_split(df, config) [utils/split_utils.py]
        │               ├── temporal_split_by_ratio_or_dates(train_df, config) [utils/split_utils.py]
        │               ├── normalize_features(train_df, val_df, test_df, config) [utils/normalization.py]
        │               ├── build_tabular_sequence(train_data, config)  [utils/sequence_builder.py]
        │               ├── build_tabular_sequence(val_data, config)    [utils/sequence_builder.py]
        │               ├── build_tabular_sequence(test_data, config)   [utils/sequence_builder.py]
        │               ├── _make_loader(train_seq, config, shuffle=True)
        │               ├── _make_loader(val_seq, config, shuffle=False)
        │               └── _make_loader(test_seq, config, shuffle=False)
        ├── get_model(model_type, input_dim, output_dim, config) [models/model_factory.py]
        │     ├── 分支 lstm        → LSTMModel(...)        [models/lstm_model.py]
        │     ├── 分支 gru         → GRUModel(...)         [models/gru_model.py]
        │     └── 分支 transformer → TransformerModel(...) [models/transformer_model.py]
        ├── 分支 mode == 'train':
        │     └── Trainer(model, config)                   [utils/trainer.py]
        │           ├── Trainer.fit(train_loader, val_loader)
        │           │     └── Trainer._validate(val_loader)  ← 每 epoch 调用一次
        │           └── Trainer.save_model()
        └── 分支 mode == 'eval':
              └── Evaluator(model, config)                 [utils/evaluator.py]
                    ├── Evaluator.load_model()
                    └── Evaluator.evaluate(test_loader)

================================================================================
补充说明
================================================================================

1. 【数据拆分差异汇总】
   Dataset1 (ratio):  全量 → 后20% test，剩余再后10% val，其余 train
   Dataset2 (date):   全量 → ≥2023-10-01 test，[2023-09-01, 2023-10-01) val，其余 train
   Dataset3 (ratio):  全量 → 后15% test，剩余再后10% val，其余 train

2. 【归一化差异汇总】
   Dataset1: StandardScaler，特征 ['temp', 'pressure', 'vibration']
   Dataset2: MinMaxScaler，  特征 ['duration', 'count', 'score']
   Dataset3: StandardScaler，特征 ['channel_1', 'channel_2', 'channel_3', 'channel_4']

3. 【序列构建参数差异汇总】
   Dataset1: window_size=24, stride=1,  特征数=3数值+2时间=5, output_dim=1
   Dataset2: window_size=12, stride=1,  特征数=3数值+3时间=6, output_dim=1
   Dataset3: window_size=48, stride=6,  特征数=4数值+2时间=6, output_dim=3

4. 【模型权重保存位置】
   训练模式：每轮验证损失最优时自动保存至 checkpoints/best_model.pt
   评估模式：从 checkpoints/best_model.pt 加载权重

5. 【需要人工确认的部分】
   - models/lstm_model.py、models/gru_model.py、models/transformer_model.py
     三个模型类的内部 forward() 实现未在本文档中详细展开，
     如需了解模型结构请直接查阅对应文件。
   - config['training'] 中 hidden_dim / num_layers 字段在 CONFIGS 中未显式定义，
     model_factory.py 使用默认值 hidden_dim=64 / num_layers=2，实际以代码为准。

================================================================================
"""

with open('/Users/ming/Desktop/目前在用全新实验的副本/call_order_explanation.txt', 'w', encoding='utf-8') as f:
    f.write(content.strip())

print("✅ 文档已生成：call_order_explanation.txt")