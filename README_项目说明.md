# 项目说明文档

## 1. 项目是什么

这是一个用于复现论文 **《A multi-source multi-layer-based transfer learning approach for forecasting customer demands of newly launched products》** 的 Python 项目。

项目的核心目标不是只训练一个模型，而是完成一整套实验流程：

1. 读取原始销售数据
2. 把不同数据集整理成统一格式
3. 按论文思路划分源域（source）和目标域（target）
4. 选择与目标最相似的源数据
5. 运行多种迁移学习方法进行训练
6. 计算 RMSE、Accuracy 等指标
7. 输出结果表、图表和统计分析结果

如果你是第一次接触这个项目，可以把它理解成一个“**需求预测实验平台**”。

---

## 2. 项目整体流程

项目实际运行时，推荐走下面这条主流程：

`配置文件 -> 读取数据 -> 数据清洗与标准化 -> 时间特征生成 -> source/target 划分 -> 相似源选择 -> 模型训练/迁移学习 -> 指标评估 -> 结果保存 -> 图表/统计分析`

更具体一点，可以对应到代码中的执行顺序：

1. 入口脚本读取配置  
   推荐入口：`scripts/run_main_experiment.py`

2. 统一加载数据  
   由 `data_preprocessing.py` 负责把不同数据集整理成统一字段：
   `date`、`entity_id`、`item_id`、`sales`

3. 提取时间特征  
   自动补充 `year`、`month`、`week`、`day`

4. 构建源域和目标域  
   `build_source_target_split(...)` 会按数据集规则，把可迁移的历史序列作为 source，把要预测的序列作为 target

5. 对 source 和 target 做时间切分  
   source 通常按比例切分训练/验证/测试  
   target 按论文复现思路使用近 30 天观测窗口 + 约 180 天预测窗口

6. 进行特征归一化和滑窗构造  
   把时间序列转成模型可训练的监督学习样本

7. 选择最相似的 source  
   `source_selector.py` 会先为 target 和每个 source 构造“签名向量”，再计算距离，选出 top-k 源序列

8. 运行各类方法  
   包括：
   - `No-TL`：不做迁移学习
   - `SS-TL`：单源迁移学习
   - `MSWA-TL`：多源加权聚合
   - `MSSB-TL`：多源候选选优（model switching）
   - `MSML-TL`：多源多层迁移
   - `MSML-TL-RFE`：先做特征筛选，再进行多源多层迁移

9. 统一评估结果  
   输出 `rmse`、`accuracy`、`prediction_shape`

10. 保存结果并生成图表  
   结果会写入 `outputs/` 目录，图表和格式化表格由 `result_visualizer.py` 生成

---

## 3. 各模块作用

这个项目文件比较多，但新手最需要先理解下面这些模块。

### 3.1 入口与调度模块

- `scripts/run_main_experiment.py`
  推荐的单次实验入口。适合先跑通一个数据集、一个配置。

- `run_main_experiment.py`
  根目录下的单实验入口，功能类似，但现在更推荐使用 `scripts/` 里的版本。

- `experiment_runner.py`
  整个项目最核心的“总调度器”。负责串起数据准备、方法运行、结果汇总。

- `experiment_matrix_runner.py`
  批量实验运行器。适合一次性跑多个数据集、多个 source 数量、多个参数组合。

### 3.2 数据处理模块

- `data_preprocessing.py`
  负责数据加载、清洗、标准化、时间特征提取、source/target 划分、归一化、滑窗构造。

- `dataset_registry.py`
  负责管理数据集名称、别名和默认路径，避免不同地方写死路径。

- `数据集/`
  存放项目使用的数据文件，目前主要有：
  - `Dataset1-Challenge.csv`
  - `Dataset2-pasta.csv`
  - `Dataset3-Rossmann.csv`

### 3.3 源选择模块

- `source_selector.py`
  用于从 source pool 中选出与目标序列最相似的 source。  
  它的作用非常关键，因为多源迁移学习不是把所有源都直接拿来训练，而是先挑“更像目标”的源。

### 3.4 模型与迁移学习模块

- `cnn_model.py`
  定义基础 1D CNN 模型。很多迁移学习方法都以它作为基础网络。

- `single_source_tl.py`
  单源迁移学习实现。可以理解为先在一个 source 上预训练，再迁移到 target。

- `mswa_tl.py`
  多源加权迁移学习。核心思想是多个 source 分别训练后，再按权重融合。

- `mssb_tl.py`
  多源候选选优（model switching）。会对每个候选源分别执行单源迁移，再按验证集表现选择最佳源模型作为最终结果（不是加权融合，也不是层参数融合）。

- `msml_tl.py`
  项目的核心方法之一，多源多层迁移学习。

- `msml_tl_rfe.py`
  在 `MSML-TL` 基础上增加 RFE 特征筛选，用来减少无效特征干扰。

### 3.5 评估与展示模块

- `result_visualizer.py`
  读取实验结果 CSV，生成排序表、RMSE 图、Accuracy 图等。

- `src/evaluation/metrics.py`
  定义评估指标计算逻辑。

- `scripts/run_statistical_analysis.py`
  用于做统计显著性分析，例如 Friedman、Wilcoxon 等。

### 3.6 配置与环境模块

- `configs/default_config.json`
  最重要的实验配置文件。包含数据集、特征列、方法列表、训练轮数、source 数量等参数。

- `config.py`
  提供配置读取与封装能力。

- `environment.py`
  负责日志、随机种子、运行环境初始化等。

- `init_check.py`
  新手很推荐先运行它，用来检查依赖、配置和环境是否正常。

### 3.7 目录结构说明

项目里同时存在“根目录模块”和 `src/` 下的一套镜像实现。

- 实际主运行链路，当前主要走根目录下的模块
- `src/` 更像是按功能重新整理过的结构化版本

如果你只是想先跑通项目，优先关注根目录文件和 `scripts/` 即可。

---

## 4. 数据流动路径

这一部分是新手最容易迷糊的地方。可以把数据流理解为下面这条链路。

### 4.1 原始数据进入系统

原始 CSV 文件放在 `数据集/` 目录中。

程序启动后，会先根据配置文件中的数据集名称，去 `dataset_registry.py` 或配置路径中找到对应文件。

### 4.2 统一字段格式

不同数据集原始字段可能不一样，但进入系统后，都会被整理成统一结构：

- `date`：日期
- `entity_id`：实体编号，比如门店或品牌
- `item_id`：商品编号
- `sales`：销量

这样做的目的，是让后面的模型训练逻辑不需要为每个数据集单独写一套代码。

### 4.3 特征扩展

在基础字段上，程序会继续生成一些时间特征，例如：

- `year`
- `month`
- `week`
- `day`

这些特征会和 `sales` 一起组成模型输入特征。

### 4.4 source / target 分流

接下来，数据不会直接全部送进模型，而是先拆成两部分：

- `source_df`：源域数据，用来提供可迁移知识
- `target_df`：目标域数据，是最终真正要预测的对象

你可以把它理解为：

- source = “历史上和你相似的旧商品/旧门店”
- target = “现在真正想预测的新商品/新门店”

### 4.5 时间切分

source 和 target 之后还会继续切分：

- source：通常按训练 / 验证 / 测试比例切分
- target：按论文协议切出观测窗口和预测窗口

也就是说，target 不是随机打乱切分，而是严格按时间顺序往后预测。

### 4.6 归一化与滑窗

切分后会进行两件事：

1. 对数值特征做归一化  
   这样不同量纲的数据更容易训练

2. 构造滑动窗口样本  
   比如用前 10 个时间步去预测后 1 个时间步

经过这一步后，原始时间序列就会变成适合 CNN 输入的训练样本。

### 4.7 相似源选择

在多源迁移学习中，不会直接把所有 source 全用上，而是先：

1. 为 target 生成一个统计签名向量
2. 为每个 source 也生成签名向量
3. 计算 target 与每个 source 的距离
4. 选出距离最近的 top-k 个 source
5. 根据距离计算权重

这一步输出的是“哪些 source 最值得迁移”。

### 4.8 模型训练与迁移

不同方法会沿着不同路径处理这些数据：

- `No-TL`：只使用 target 数据训练
- `SS-TL`：先训练一个 source，再迁移到 target
- `MSWA-TL`：多个 source 分别训练后按权重融合
- `MSSB-TL`：多源候选选优（model switching）
- `MSML-TL`：多源多层迁移
- `MSML-TL-RFE`：先筛特征，再执行多源多层迁移

### 4.9 结果输出

训练结束后，统一生成结果表，常见输出包括：

- `outputs/experiment_results/`：原始实验结果 CSV
- `outputs/results_reports/`：格式化表格和图表
- `outputs/matrix_runs/`：批量实验矩阵结果
- `outputs/paper_alignment_reports/`：论文协议对齐检查结果

---

## 5. 运行步骤

下面按“新手第一次上手”的顺序来写。

### 5.1 准备环境

先确认你已经安装 Python 依赖。项目依赖主要写在 `requirements.txt` 中，包括：

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `pyyaml`
- `scipy`
- `statsmodels`

推荐命令：

```bash
pip install -r requirements.txt
```

### 5.2 先做初始化检查

第一次运行前，建议先检查环境是否完整：

```bash
python init_check.py
```

这个脚本会检查：

- 关键文件是否存在
- Python 依赖是否安装
- `config.py`、`environment.py` 是否能正常导入
- 配置文件是否能正常加载

### 5.3 查看默认配置

项目默认配置在：

- `configs/default_config.json`

新手最需要先关注这些字段：

- `single_experiment.dataset_name`：当前跑哪个数据集
- `single_experiment.k`：选多少个 source
- `single_experiment.horizon`：预测步长
- `single_experiment.source_epochs`：source 训练轮数
- `single_experiment.target_epochs`：target 训练轮数
- `methods.all_methods`：启用哪些方法

### 5.4 运行单次实验

推荐命令：

```bash
python scripts/run_main_experiment.py
```

如果你想查看更多日志，可以加上：

```bash
python scripts/run_main_experiment.py --verbose-mode full
```

如果你希望强制按严格论文模式运行，可以用：

```bash
python scripts/run_main_experiment.py --strict-paper-mode
```

### 5.5 查看结果

单次实验完成后，重点看下面这些目录：

- `outputs/experiment_results/`
  保存原始结果 CSV

- `outputs/results_reports/`
  保存格式化结果表、RMSE 图、Accuracy 图

### 5.6 运行批量实验矩阵

如果你不是只想验证一次，而是想批量跑很多实验组合，可以使用：

```bash
python scripts/run_full_experiment_matrix.py
```

它会根据 `configs/default_config.json` 中的 `matrix` 配置，自动组合：

- 数据集
- horizon
- source_count
- weight_mode
- keep_ratio
- 方法组合

运行结果会保存到：

- `outputs/matrix_runs/`

### 5.7 运行统计分析

如果你已经拿到了实验结果，想继续做论文风格统计检验，可以运行：

```bash
python scripts/run_statistical_analysis.py
```

### 5.8 新手建议的最短上手路径

如果你只想先跑通一遍，建议按下面顺序：

1. `pip install -r requirements.txt`
2. `python init_check.py`
3. `python scripts/run_main_experiment.py`
4. 打开 `outputs/results_reports/` 查看结果表和图

---

## 6. 这个项目最重要的几个理解点

### 6.1 这不是普通的单模型项目

它更像一个“实验框架”，里面包含多种迁移学习方法和统一评估流程。

### 6.2 数据处理和实验调度比单个模型更重要

真正把整个项目串起来的核心，不是 `cnn_model.py` 本身，而是：

- `experiment_runner.py`
- `data_preprocessing.py`
- `source_selector.py`

### 6.3 先理解 source 和 target，再看模型

只看模型代码很容易迷路。  
先明白“谁是 source，谁是 target，为什么要选 top-k source”，再看 `MSML-TL` 这类方法会容易很多。

### 6.4 优先使用 `scripts/` 下的入口

因为项目历史上保留了多套入口和双份代码结构，新手直接从 `scripts/` 下脚本开始最稳妥。

---

## 7. 推荐阅读顺序

如果你想真正理解项目，建议按这个顺序看：

1. 先看 `README_项目说明.md`（本文件）
2. 再看 `configs/default_config.json`
3. 再看 `scripts/run_main_experiment.py`
4. 再看 `experiment_runner.py`
5. 再看 `data_preprocessing.py`
6. 再看 `source_selector.py`
7. 最后再看 `single_source_tl.py`、`mswa_tl.py`、`mssb_tl.py`、`msml_tl.py`、`msml_tl_rfe.py`

这样会比一开始直接钻进模型代码更容易理解整个项目。

---

## 8. 一句话总结

这个项目可以理解为：

**一个面向论文复现的多源迁移学习需求预测实验平台，它负责把原始销售数据整理成统一格式，挑选最相似的源数据，运行多种迁移学习方法，并最终输出可用于论文比较的结果表、图表和统计分析。**
