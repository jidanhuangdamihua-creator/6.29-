"""
配置解析与管理模块 (Configuration Parser and Manager)
====================================================

这个模块负责：
1. 解析YAML配置文件
2. 验证配置参数的完整性和有效性
3. 提供统一的配置对象接口
4. 支持配置的覆盖和动态修改
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
import logging

try:
    import yaml
except ImportError:
    yaml = None


# 初始化日志记录器
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置数据类"""
    name: str
    path: str
    target_product_id: Optional[int] = None
    num_source_sequences: int = 3
    use_information_sharing: bool = True
    forecast_horizon: int = 12
    lookback_window: int = 24
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    normalize: bool = True
    normalization_method: str = "minmax"
    
    def __post_init__(self):
        """验证数据集配置"""
        assert 0 < self.train_ratio < 1, "train_ratio must be between 0 and 1"
        assert 0 < self.val_ratio < 1, "val_ratio must be between 0 and 1"
        assert 0 < self.test_ratio < 1, "test_ratio must be between 0 and 1"
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        assert 0.99 <= total_ratio <= 1.01, f"Sum of ratios must equal 1.0, got {total_ratio}"
        assert self.forecast_horizon > 0, "forecast_horizon must be positive"
        assert self.lookback_window > 0, "lookback_window must be positive"
        assert self.normalization_method in ["minmax", "standard", "robust"], \
            f"Unknown normalization method: {self.normalization_method}"


@dataclass
class CNNConfig:
    """CNN层配置"""
    num_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: List[int] = field(default_factory=lambda: [3, 3, 3])
    stride: int = 1
    padding: str = "same"
    activation: str = "relu"
    dropout_rate: float = 0.2


@dataclass
class LSTMConfig:
    """LSTM层配置"""
    units: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.2
    return_sequences: bool = True


@dataclass
class AttentionConfig:
    """注意力机制配置"""
    enabled: bool = True
    num_heads: int = 4
    attention_dim: int = 64


@dataclass
class OutputConfig:
    """输出层配置"""
    units: int = 1
    activation: str = "linear"


@dataclass
class ModelConfig:
    """模型配置数据类"""
    architecture: str = "cnn_lstm"
    cnn: CNNConfig = field(default_factory=CNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """验证模型配置"""
        valid_architectures = ["cnn", "lstm", "cnn_lstm", "transformer", "multihead_attention"]
        assert self.architecture in valid_architectures, \
            f"Unknown architecture: {self.architecture}. Valid options: {valid_architectures}"


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae", "mse", "r2_score"])
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 15
    lr_scheduler_enabled: bool = True
    lr_scheduler_type: str = "exponential"
    
    def __post_init__(self):
        """验证训练配置"""
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.optimizer in ["adam", "sgd", "rmsprop"], \
            f"Unknown optimizer: {self.optimizer}"
        assert self.loss in ["mse", "mae", "rmse", "huber"], \
            f"Unknown loss function: {self.loss}"


@dataclass
class SupplyChainConfig:
    """供应链配置数据类"""
    initial_inventory: Dict[str, Any] = field(default_factory=lambda: {
        "quantity": 100,
        "value": 10000,
        "location": "warehouse"
    })
    order_quantity_Q: int = 50
    lead_time_days: int = 3
    ordering_cost_per_order: float = 100.0
    unit_acquisition_cost: float = 50.0
    holding_cost_daily: float = 0.0137
    stockout_cost_per_unit: float = 25.0
    target_service_level: float = 0.95
    target_fill_rate: float = 0.95
    warehouse_capacity: int = 500
    time_period_days: int = 365
    num_simulations: int = 100
    
    def __post_init__(self):
        """验证供应链配置"""
        assert self.order_quantity_Q > 0, "order_quantity_Q must be positive"
        assert self.lead_time_days >= 0, "lead_time_days cannot be negative"
        assert self.ordering_cost_per_order > 0, "ordering_cost_per_order must be positive"
        assert self.unit_acquisition_cost > 0, "unit_acquisition_cost must be positive"
        assert 0 < self.target_service_level <= 1, "target_service_level must be between 0 and 1"
        assert 0 < self.target_fill_rate <= 1, "target_fill_rate must be between 0 and 1"


# ============================================================================
# 新增数据类：实验矩阵配置和依赖版本信息
# ============================================================================

@dataclass
class ExperimentMatrixConfig:
    """
    实验矩阵配置数据类（新增）
    用于定义多组实验的组合配置，支持矩阵化运行
    """
    enabled: bool = False
    datasets_to_run: List[str] = field(default_factory=lambda: ["demand-forecasting"])
    source_counts: List[int] = field(default_factory=lambda: [2, 3, 5])
    horizons: List[int] = field(default_factory=lambda: [7, 12, 24])
    scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enabled_methods: List[str] = field(default_factory=lambda: ["baseline", "lstm", "cnn_lstm"])
    run_control: Dict[str, Any] = field(default_factory=lambda: {
        "cross_product_mode": False,
        "save_config_snapshot": True,
        "save_env_snapshot": True,
        "snapshot_output_dir": "./outputs/run_snapshots/"
    })
    
    def __post_init__(self):
        """验证实验矩阵配置"""
        # 交叉字段验证：如果启用矩阵模式，必须有数据集和方法定义
        if self.enabled:
            assert len(self.datasets_to_run) > 0, "datasets_to_run cannot be empty when enabled"
            assert len(self.enabled_methods) > 0, "enabled_methods cannot be empty when enabled"
            assert len(self.source_counts) > 0, "source_counts cannot be empty when enabled"
            assert len(self.horizons) > 0, "horizons cannot be empty when enabled"
        
        # 验证各个参数的有效性
        assert all(s > 0 for s in self.source_counts), "All source_counts must be positive"
        assert all(h > 0 for h in self.horizons), "All horizons must be positive"
        
        # 验证快照输出目录配置
        if isinstance(self.run_control, dict):
            assert "snapshot_output_dir" in self.run_control, "snapshot_output_dir must be defined"


@dataclass
class DependencyConfig:
    """
    依赖版本配置数据类（新增）
    用于记录与验证项目依赖的推荐版本
    """
    python_version: Dict[str, Any] = field(default_factory=lambda: {
        "min": "3.8",
        "recommended": "3.9",
        "tested_on": ["3.9", "3.10"]
    })
    core: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    deeplearning: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    visualization: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timeseries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_core_packages(self) -> Dict[str, str]:
        """获取核心依赖列表（包名->版本）"""
        return {pkg: spec.get('version', '') for pkg, spec in self.core.items()}
    
    def get_tensorflow_mac_m1_spec(self) -> str:
        """获取 Mac M1 特定的 TensorFlow 安装指令"""
        tf_spec = self.deeplearning.get('tensorflow', {})
        return tf_spec.get('mac_m1_specific', 'tensorflow-macos>=2.10.0')


class Config:
    """
    统一配置管理类（扩展版）
    
    新增功能：
    - 交叉字段验证（跨参数逻辑校验）
    - 配置快照保存功能
    - 静态配置对象 vs 运行时状态的区分
    """
    
    def __init__(self, config_file: Optional[str] = None, 
                 supply_chain_file: Optional[str] = None,
                 verbose: bool = True):
        """
        初始化配置管理器
        
        Args:
            config_file: 全局配置YAML文件路径
            supply_chain_file: 供应链配置YAML文件路径
            verbose: 是否输出详细日志
        """
        self.config_file = config_file or "config.yaml"
        self.supply_chain_file = supply_chain_file or "supply_chain.yaml"
        self.verbose = verbose
        
        # 初始化配置对象
        self.experiment: Dict[str, Any] = {}
        self.dataset: DatasetConfig = None
        self.model: ModelConfig = None
        self.training: TrainingConfig = None
        self.supply_chain: SupplyChainConfig = None
        self.logging: Dict[str, Any] = {}
        self.output: Dict[str, Any] = {}
        self.methods: Dict[str, Any] = {}
        self.mac_m1: Dict[str, Any] = {}
        self.advanced: Dict[str, Any] = {}
        
        # 新增：实验矩阵和依赖版本配置（现有模块的扩展）
        self.experiment_matrix: ExperimentMatrixConfig = None
        self.dependencies: DependencyConfig = None
        
        # 新增：运行时状态对象（区别于静态配置）
        self._runtime_state: Dict[str, Any] = {
            'config_snapshot_path': None,
            'env_snapshot_path': None,
            'load_timestamp': None,
            'validation_errors': []
        }
        
        # 加载配置文件
        self._load_configs()
    
    def _load_configs(self):
        """加载所有配置文件"""
        if yaml is None:
            logger.warning("PyYAML not installed. Installing pyyaml is recommended.")
            logger.warning("You can install it with: pip install pyyaml")
        
        # 加载主配置文件
        if os.path.exists(self.config_file):
            self._load_main_config()
        else:
            logger.warning(f"Config file not found: {self.config_file}")
        
        # 加载供应链配置文件
        if os.path.exists(self.supply_chain_file):
            self._load_supply_chain_config()
        else:
            logger.warning(f"Supply chain config file not found: {self.supply_chain_file}")
    
    def _load_main_config(self):
        """加载主配置文件"""
        try:
            if yaml is None:
                logger.error("PyYAML is required. Install with: pip install pyyaml")
                return
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                logger.error(f"Empty or invalid YAML file: {self.config_file}")
                return
            
            # 解析实验配置
            self.experiment = config_data.get('experiment', {})
            
            # 解析数据集配置
            dataset_dict = config_data.get('dataset', {})
            self.dataset = DatasetConfig(**dataset_dict)
            
            # 解析模型配置
            model_dict = config_data.get('model', {})
            self._parse_model_config(model_dict)
            
            # 解析训练配置
            training_dict = config_data.get('training', {})
            self._parse_training_config(training_dict)
            
            # 新增：解析实验矩阵配置
            exp_matrix_dict = config_data.get('experiment_matrix', {})
            self._parse_experiment_matrix_config(exp_matrix_dict)
            
            # 新增：解析依赖版本配置
            dependencies_dict = config_data.get('dependencies', {})
            self._parse_dependencies_config(dependencies_dict)
            
            # 其他配置
            self.logging = config_data.get('logging', {})
            self.output = config_data.get('output', {})
            self.methods = config_data.get('methods', {})
            self.mac_m1 = config_data.get('mac_m1', {})
            self.advanced = config_data.get('advanced', {})
            
            # 新增：运行时状态记录（加载完成时间戳）
            import datetime
            self._runtime_state['load_timestamp'] = datetime.datetime.now().isoformat()
            
            if self.verbose:
                logger.info(f"✓ Successfully loaded main config from: {self.config_file}")
        
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.config_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def _parse_model_config(self, model_dict: Dict[str, Any]):
        """解析模型配置"""
        cnn_config = CNNConfig(**model_dict.get('cnn', {}))
        lstm_config = LSTMConfig(**model_dict.get('lstm', {}))
        attention_config = AttentionConfig(**model_dict.get('attention', {}))
        output_config = OutputConfig(**model_dict.get('output', {}))
        
        self.model = ModelConfig(
            architecture=model_dict.get('architecture', 'cnn_lstm'),
            cnn=cnn_config,
            lstm=lstm_config,
            attention=attention_config,
            output=output_config
        )
    
    def _parse_training_config(self, training_dict: Dict[str, Any]):
        """解析训练配置"""
        lr_scheduler = training_dict.get('lr_scheduler', {})
        early_stopping = training_dict.get('early_stopping', {})
        
        self.training = TrainingConfig(
            epochs=training_dict.get('epochs', 100),
            batch_size=training_dict.get('batch_size', 32),
            learning_rate=training_dict.get('learning_rate', 0.001),
            optimizer=training_dict.get('optimizer', 'adam'),
            loss=training_dict.get('loss', 'mse'),
            metrics=training_dict.get('metrics', ['mae', 'mse']),
            early_stopping_enabled=early_stopping.get('enabled', True),
            early_stopping_patience=early_stopping.get('patience', 15),
            lr_scheduler_enabled=lr_scheduler.get('enabled', True),
            lr_scheduler_type=lr_scheduler.get('type', 'exponential')
        )
    
    def _parse_experiment_matrix_config(self, exp_matrix_dict: Dict[str, Any]):
        """
        解析实验矩阵配置（新增）
        
        支持：
        - 多数据集配置
        - K值组合（源序列数）
        - 预测步长组合
        - 实验场景快速切换
        - 对比方法配置
        """
        self.experiment_matrix = ExperimentMatrixConfig(
            enabled=exp_matrix_dict.get('enabled', False),
            datasets_to_run=exp_matrix_dict.get('datasets_to_run', ['demand-forecasting']),
            source_counts=exp_matrix_dict.get('source_counts', [2, 3, 5]),
            horizons=exp_matrix_dict.get('horizons', [7, 12, 24]),
            scenarios=exp_matrix_dict.get('scenarios', {}),
            enabled_methods=exp_matrix_dict.get('enabled_methods', []),
            run_control=exp_matrix_dict.get('run_control', {})
        )
    
    def _parse_dependencies_config(self, dependencies_dict: Dict[str, Any]):
        """
        解析依赖版本配置（新增）
        
        用于：
        - 记录推荐依赖版本
        - Mac M1 特定说明
        - 平台兼容性信息
        """
        self.dependencies = DependencyConfig(
            python_version=dependencies_dict.get('python_version', {}),
            core=dependencies_dict.get('core', {}),
            deeplearning=dependencies_dict.get('deeplearning', {}),
            visualization=dependencies_dict.get('visualization', {}),
            timeseries=dependencies_dict.get('timeseries', {})
        )
    
    def _validate_cross_fields(self) -> List[str]:
        """
        执行跨字段验证（新增）
        
        验证多个字段之间的逻辑一致性，而不仅仅是单个字段的类型验证。
        
        Returns:
            错误信息列表（如果验证通过，列表为空）
        """
        errors = []
        
        # 交叉验证1：如果启用实验矩阵，需要确保场景有效
        if self.experiment_matrix and self.experiment_matrix.enabled:
            if not self.experiment_matrix.scenarios:
                errors.append("experiment_matrix is enabled but scenarios are empty")
            
            # 交叉验证2：检查启用的场景数量
            active_scenarios = [s for s, cfg in self.experiment_matrix.scenarios.items() 
                               if isinstance(cfg, dict) and cfg.get('enabled', False)]
            if not self.experiment_matrix.run_control.get('cross_product_mode', False):
                if not active_scenarios:
                    errors.append("experiment_matrix has no active scenarios in cross_product_mode=false")
        
        # 交叉验证3：数据集比例必须和为1
        if self.dataset:
            total_ratio = self.dataset.train_ratio + self.dataset.val_ratio + self.dataset.test_ratio
            if not (0.99 <= total_ratio <= 1.01):
                errors.append(f"Dataset ratios don't sum to 1.0: {total_ratio}")
        
        # 交叉验证4：如果使用信息共享，源序列数不能为1
        if self.dataset and self.dataset.use_information_sharing:
            if self.dataset.num_source_sequences < 2:
                errors.append("Cannot use information_sharing with less than 2 source sequences")
        
        # 交叉验证5：快照输出目录配置检查
        if self.experiment_matrix and self.experiment_matrix.run_control:
            if self.experiment_matrix.run_control.get('save_config_snapshot', False):
                snapshot_dir = self.experiment_matrix.run_control.get('snapshot_output_dir')
                if not snapshot_dir:
                    errors.append("save_config_snapshot=true but snapshot_output_dir not specified")
        
        return errors
    
    def validate_and_report(self) -> bool:
        """
        执行完整验证并报告结果（新增）
        
        Returns:
            如果验证通过返回True，否则返回False
        """
        errors = self._validate_cross_fields()
        self._runtime_state['validation_errors'] = errors
        
        if errors:
            logger.warning(f"Configuration validation found {len(errors)} issue(s):")
            for error in errors:
                logger.warning(f"  - {error}")
            return False
        else:
            logger.info("✓ Configuration validation passed")
            return True
    
    def save_config_snapshot(self, snapshot_dir: Optional[str] = None) -> str:
        """
        保存配置快照到文件（新增）
        
        Args:
            snapshot_dir: 快照目录，如果为None则使用配置中的默认路径
        
        Returns:
            快照文件路径
        """
        import json
        from datetime import datetime
        
        if snapshot_dir is None:
            if self.experiment_matrix and self.experiment_matrix.run_control:
                snapshot_dir = self.experiment_matrix.run_control.get('snapshot_output_dir', './outputs/run_snapshots/')
            else:
                snapshot_dir = './outputs/run_snapshots/'
        
        # 创建快照目录
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成快照文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_filename = f"config_snapshot_{timestamp}.json"
        snapshot_path = Path(snapshot_dir) / snapshot_filename
        
        # 转换配置为字典并保存
        config_dict = self.to_dict()
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # 记录快照路径到运行时状态
        self._runtime_state['config_snapshot_path'] = str(snapshot_path)
        
        logger.info(f"✓ Config snapshot saved: {snapshot_path}")
        return str(snapshot_path)
    
    def _load_supply_chain_config(self):
        """加载供应链配置文件"""
        try:
            if yaml is None:
                logger.error("PyYAML is required. Install with: pip install pyyaml")
                return
            
            with open(self.supply_chain_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                logger.error(f"Empty or invalid YAML file: {self.supply_chain_file}")
                return
            
            sc_dict = config_data.get('supply_chain', {})
            self.supply_chain = SupplyChainConfig(
                initial_inventory=sc_dict.get('initial_inventory', {}),
                order_quantity_Q=sc_dict.get('ordering', {}).get('order_quantity_Q', 50),
                lead_time_days=sc_dict.get('ordering', {}).get('lead_time_days', 3),
                ordering_cost_per_order=sc_dict.get('costs', {}).get('ordering_cost_per_order', 100.0),
                unit_acquisition_cost=sc_dict.get('costs', {}).get('unit_acquisition_cost', 50.0),
                holding_cost_daily=sc_dict.get('costs', {}).get('holding_cost_daily', 0.0137),
                stockout_cost_per_unit=sc_dict.get('costs', {}).get('stockout_cost_per_unit', 25.0),
                target_service_level=sc_dict.get('service_level', {}).get('target_service_level', 0.95),
                target_fill_rate=sc_dict.get('service_level', {}).get('target_fill_rate', 0.95),
                warehouse_capacity=sc_dict.get('constraints', {}).get('warehouse_capacity', 500),
                time_period_days=sc_dict.get('simulation', {}).get('time_period_days', 365),
                num_simulations=sc_dict.get('simulation', {}).get('num_simulations', 100)
            )
            
            if self.verbose:
                logger.info(f"✓ Successfully loaded supply chain config from: {self.supply_chain_file}")
        
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.supply_chain_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading supply chain config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点记法，如 'dataset.forecast_horizon'）
        
        Args:
            key: 配置键（支持点记法）
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = getattr(value, k, None)
            
            if value is None:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值（支持点记法）
        
        Args:
            key: 配置键（支持点记法）
            value: 要设置的值
        """
        keys = key.split('.')
        obj = self
        
        for k in keys[:-1]:
            if not hasattr(obj, k):
                setattr(obj, k, {})
            obj = getattr(obj, k)
        
        setattr(obj, keys[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典（扩展版）"""
        return {
            'experiment': self.experiment,
            'experiment_matrix': asdict(self.experiment_matrix) if self.experiment_matrix else {},
            'dataset': asdict(self.dataset) if self.dataset else {},
            'model': asdict(self.model) if self.model else {},
            'training': asdict(self.training) if self.training else {},
            'supply_chain': asdict(self.supply_chain) if self.supply_chain else {},
            'dependencies': asdict(self.dependencies) if self.dependencies else {},
            'logging': self.logging,
            'output': self.output,
            'methods': self.methods,
            'mac_m1': self.mac_m1,
            'advanced': self.advanced
        }
    
    def print_summary(self):
        """打印配置摘要"""
        logger.info("=" * 70)
        logger.info("实验配置摘要 (Configuration Summary)")
        logger.info("=" * 70)
        
        if self.experiment:
            logger.info(f"实验名称: {self.experiment.get('name')}")
            logger.info(f"种子值: {self.experiment.get('seed')}")
        
        if self.dataset:
            logger.info(f"\n数据集: {self.dataset.name}")
            logger.info(f"  - 预测步长: {self.dataset.forecast_horizon}")
            logger.info(f"  - 源序列数K: {self.dataset.num_source_sequences}")
            logger.info(f"  - 信息共享: {self.dataset.use_information_sharing}")
        
        if self.model:
            logger.info(f"\n模型架构: {self.model.architecture}")
            logger.info(f"  - CNN滤波器: {self.model.cnn.num_filters}")
            logger.info(f"  - LSTM单元: {self.model.lstm.units}")
            logger.info(f"  - 注意力机制: {self.model.attention.enabled}")
        
        if self.training:
            logger.info(f"\n训练配置:")
            logger.info(f"  - Epochs: {self.training.epochs}")
            logger.info(f"  - Batch Size: {self.training.batch_size}")
            logger.info(f"  - 学习率: {self.training.learning_rate}")
            logger.info(f"  - 优化器: {self.training.optimizer}")
        
        if self.mac_m1:
            logger.info(f"\nMac M1配置:")
            logger.info(f"  - 强制使用CPU: {self.mac_m1.get('tensorflow_force_cpu')}")
            logger.info(f"  - Metal加速: {self.mac_m1.get('use_metal_acceleration')}")
        
        # 新增：实验矩阵配置摘要
        if self.experiment_matrix and self.experiment_matrix.enabled:
            logger.info(f"\n🔄 实验矩阵配置:")
            logger.info(f"  - 启用: {self.experiment_matrix.enabled}")
            logger.info(f"  - 数据集: {', '.join(self.experiment_matrix.datasets_to_run)}")
            logger.info(f"  - 源序列数: {self.experiment_matrix.source_counts}")
            logger.info(f"  - 预测步长: {self.experiment_matrix.horizons}")
            logger.info(f"  - 对比方法: {len(self.experiment_matrix.enabled_methods)} 种")
            logger.info(f"  - 场景数: {len(self.experiment_matrix.scenarios)}")
        
        # 新增：依赖版本信息摘要
        if self.dependencies:
            logger.info(f"\n📦 依赖版本:")
            logger.info(f"  - Python: {self.dependencies.python_version.get('recommended', 'N/A')}")
            if self.dependencies.core:
                core_count = len(self.dependencies.core)
                logger.info(f"  - 核心依赖: {core_count} 个包")
            if self.dependencies.deeplearning:
                dl_count = len(self.dependencies.deeplearning)
                logger.info(f"  - 深度学习框架: {dl_count} 个包")
        
        # 新增：验证结果摘要
        if self._runtime_state['validation_errors']:
            logger.warning(f"\n⚠️ 验证警告 ({len(self._runtime_state['validation_errors'])} 项):")
            for error in self._runtime_state['validation_errors']:
                logger.warning(f"  - {error}")
        else:
            logger.info(f"\n✓ 配置验证通过")
        
        logger.info("=" * 70)
